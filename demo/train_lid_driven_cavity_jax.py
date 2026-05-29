#!/usr/bin/env python3
"""
JAX rewrite of the parametric trapped/lid-driven cavity PINN.

This script follows the DeepXDE formulation in CavityTrapREDepthSSB.py:
  inputs:  [x, y, ReNorm, DNorm, triNorm] in [0,1]^5
  params:  Re  in [900, 1100]
           D   in [1.9, 3.3]
           tri in [0.0, 0.01]
  raw NN:  [psi_prime, p]
  output:  [u, v, p] from a hard streamfunction/lid ansatz
  residual: weighted x/y momentum equations only; continuity is enforced by ansatz.

It also follows the attached SSBroyden usage pattern:
  Adam warmup -> ravel_pytree -> dense H0 -> blockwise Crunch SSBroyden -> export ONNX.

The default settings are deliberately small enough for a repository demo. Increase
--width, --depth, --n-pde, --adam-steps, and --ssb-total-iters for a production run.
"""

from __future__ import annotations

import argparse

import json

import math

import os

import pickle

import shutil

import sys

from dataclasses import asdict, dataclass

from functools import partial

from pathlib import Path

from typing import Any

# IMPORTANT:

# Enable JAX float64 before importing jax.numpy or creating any JAX arrays.

# Do not use setdefault; an inherited JAX_ENABLE_X64=False would otherwise win.

os.environ["JAX_ENABLE_X64"] = "True"

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax

import jax.numpy as jnp

import numpy as np

from jax.flatten_util import ravel_pytree


# Make the bundled Crunch package importable. The zip supplied in the prompt uses
# a directory named NABLA-SciML/Crunch.
THIS_DIR = Path(__file__).resolve().parent
CRUNCH_ROOT = THIS_DIR / "NABLA-SciML"
if CRUNCH_ROOT.exists():
    sys.path.insert(0, str(CRUNCH_ROOT))

try:
    from Crunch.Auxiliary.utils import static_options_SSBroyden
    from Crunch.Optimizers.minimize import minimize

    HAVE_CRUNCH = True
except Exception as exc:  # pragma: no cover - useful for environments without Crunch
    print(f"[warn] Could not import bundled Crunch SSBroyden optimizer: {exc}")
    static_options_SSBroyden = None
    minimize = None
    HAVE_CRUNCH = False


@dataclass(frozen=True)
class CavityConfig:
    re_min: float = 900.0
    re_max: float = 1100.0
    d_min: float = 1.9
    d_max: float = 3.3
    tri_min: float = 0.0
    tri_max: float = 0.01
    dx: float = math.sqrt(1.0e-3)
    dy: float = math.sqrt(1.0e-2)
    input_dim: int = 5
    raw_output_dim: int = 2


CFG = CavityConfig()


def physical_parameters(z: jnp.ndarray, cfg: CavityConfig = CFG):
    """Decode normalized input z = [x,y,ReNorm,DNorm,triNorm]."""
    re_norm = z[2]
    d_norm = z[3]
    tri_norm = z[4]
    re = cfg.re_min + (cfg.re_max - cfg.re_min) * re_norm
    depth = cfg.d_min + (cfg.d_max - cfg.d_min) * d_norm
    tri = cfg.tri_min + (cfg.tri_max - cfg.tri_min) * tri_norm
    return re, depth, tri


def init_mlp(key: jax.Array, layer_sizes: list[int], dtype=jnp.float64):
    """Glorot-normal tanh MLP parameters as a list of {'W','b'} dicts."""
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for k, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        std = jnp.sqrt(jnp.asarray(2.0 / (fan_in + fan_out), dtype=dtype))
        W = std * jax.random.normal(k, (fan_in, fan_out), dtype=dtype)
        b = jnp.zeros((fan_out,), dtype=dtype)
        params.append({"W": W, "b": b})
    return params


def raw_apply(params, z: jnp.ndarray) -> jnp.ndarray:
    """Raw network: z -> [psi_prime, p]. Single-point function."""
    h = z
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    return h @ params[-1]["W"] + params[-1]["b"]


def lid_derivatives(x, y, depth, tri, cfg: CavityConfig = CFG):
    """Analytic derivatives of the built-in lid streamfunction term.

    This is the active DeepXDE transform, including ExpB.
    """
    dx2 = cfg.dx**2
    dy2 = cfg.dy**2

    left = x - tri * (1.0 - y)
    right = 1.0 - x - tri * (1.0 - y)

    exp_l = jnp.exp(-(left**2) / dx2)
    exp_r = jnp.exp(-(right**2) / dx2)
    exp_b = jnp.exp(-((1.0 - y) ** 2) / dy2)

    common = depth * (y - 1.0) * y**2

    psi_lid_x = (
        -common * 2.0 * right / dx2 * exp_r * (1.0 - exp_l) * exp_b
        + common * (1.0 - exp_r) * 2.0 * left / dx2 * exp_l * exp_b
    )

    psi_lid_y = (
        depth * (y**2 + 2.0 * y * (y - 1.0)) * (1.0 - exp_r) * (1.0 - exp_l) * exp_b
        + common * 2.0 * tri * right / dx2 * exp_r * (1.0 - exp_l) * exp_b
        + common * (1.0 - exp_r) * (2.0 * tri * left / dx2 * exp_l) * exp_b
        + common * (1.0 - exp_r) * (1.0 - exp_l) * 2.0 * (1.0 - y) / dy2 * exp_b
    )
    return psi_lid_x, psi_lid_y


def boundary_envelope_derivatives(x, y, tri):
    left = x - tri * (1.0 - y)
    right = 1.0 - x - tri * (1.0 - y)
    bcv = 16.0 * left * right * y * (1.0 - y)
    dbcv_x = 16.0 * (1.0 - 2.0 * x) * y * (1.0 - y)
    dbcv_y = (
        16.0 * left * right * (1.0 - 2.0 * y)
        + 16.0 * tri * (1.0 - 2.0 * tri * (1.0 - y)) * y * (1.0 - y)
    )
    return bcv, dbcv_x, dbcv_y


def basis_like(z: jnp.ndarray, idx: int) -> jnp.ndarray:
    """Coordinate basis vector with the same dtype/shape as z."""
    return jnp.zeros_like(z).at[idx].set(1.0)


def jvp1(f, z: jnp.ndarray, direction: jnp.ndarray):
    """First directional derivative of f at z along direction, via forward-mode AD."""
    _, df = jax.jvp(f, (z,), (direction,))
    return df


def jvp2(f, z: jnp.ndarray, direction: jnp.ndarray):
    """Second directional derivative of f at z along direction, via nested forward-mode AD."""

    def df_fn(zz):
        return jvp1(f, zz, direction)

    _, d2f = jax.jvp(df_fn, (z,), (direction,))
    return d2f


def physical_apply_single(params, z: jnp.ndarray, cfg: CavityConfig = CFG) -> jnp.ndarray:
    """Hard-transform raw NN output to physical [u,v,p]. Single-point function.

    The velocity ansatz needs psi_prime_x and psi_prime_y. These are computed
    with forward-mode JVPs rather than reverse-mode grad so the residual graph
    remains more export-friendly for jax2onnx.
    """
    x, y = z[0], z[1]
    _, depth, tri = physical_parameters(z, cfg)

    raw = raw_apply(params, z)
    psi_prime = raw[0]
    pressure = raw[1]

    ex = basis_like(z, 0)
    ey = basis_like(z, 1)

    def psi_prime_fn(zz):
        return raw_apply(params, zz)[0]

    psi_prime_x = jvp1(psi_prime_fn, z, ex)
    psi_prime_y = jvp1(psi_prime_fn, z, ey)

    bcv, dbcv_x, dbcv_y = boundary_envelope_derivatives(x, y, tri)
    psi_lid_x, psi_lid_y = lid_derivatives(x, y, depth, tri, cfg)

    u = psi_lid_y / depth + 2.0 * bcv * dbcv_y * psi_prime / depth + bcv**2 * psi_prime_y / depth
    v = -(psi_lid_x + 2.0 * bcv * dbcv_x * psi_prime + bcv**2 * psi_prime_x)
    return jnp.stack([u, v, pressure])


physical_apply_batch = jax.jit(jax.vmap(physical_apply_single, in_axes=(None, 0)))


def residual_single(params, z: jnp.ndarray, cfg: CavityConfig = CFG) -> jnp.ndarray:
    """Weighted momentum residuals [R_u, R_v]. Single-point function.

    This implementation uses forward-mode AD only. Since the PDE residual only
    needs derivatives with respect to x and y, JVPs are a cleaner fit than full
    gradients/Hessians over all five inputs and tend to produce simpler exported
    graphs for jax2onnx.
    """
    x, y = z[0], z[1]
    re, depth, tri = physical_parameters(z, cfg)

    ex = basis_like(z, 0)
    ey = basis_like(z, 1)

    def uvp_fn(zz):
        return physical_apply_single(params, zz, cfg)

    uvp = uvp_fn(z)
    uvp_x = jvp1(uvp_fn, z, ex)
    uvp_y = jvp1(uvp_fn, z, ey)
    uvp_xx = jvp2(uvp_fn, z, ex)
    uvp_yy = jvp2(uvp_fn, z, ey)

    u = uvp[0]
    v = uvp[1]

    du_x = uvp_x[0]
    dv_x = uvp_x[1]
    dp_x = uvp_x[2]

    du_y = uvp_y[0]
    dv_y = uvp_y[1]
    dp_y = uvp_y[2]

    du_xx = uvp_xx[0]
    dv_xx = uvp_xx[1]

    du_yy = uvp_yy[0]
    dv_yy = uvp_yy[1]

    loss1 = u * du_x + (v / depth) * du_y - (1.0 / re) * (du_xx + du_yy / depth**2) + dp_x
    loss2 = u * dv_x + (v / depth) * dv_y - (1.0 / re) * (dv_xx + dv_yy / depth**2) + dp_y / depth

    env_l = 0.5 * (1.0 + jnp.tanh((x - tri * (1.0 - y)) / cfg.dx))
    env_r = 0.5 * (1.0 + jnp.tanh((1.0 - x - tri * (1.0 - y)) / cfg.dx))
    factor = 9.0 * (1.0 - y) + 1.0
    weight = factor * env_l * env_r

    return jnp.stack([weight * loss1, weight * loss2])


residual_apply_batch = jax.jit(jax.vmap(residual_single, in_axes=(None, 0)))


@jax.jit
def loss_and_terms(params, x_pde):
    r = residual_apply_batch(params, x_pde)
    loss_u = jnp.mean(r[:, 0] ** 2)
    loss_v = jnp.mean(r[:, 1] ** 2)
    return loss_u + loss_v, (loss_u, loss_v)


@jax.jit
def adam_update(params, opt_state, x_pde, lr: float, beta1: float, beta2: float, eps: float):
    step, m, v = opt_state
    loss, aux = loss_and_terms(params, x_pde)
    grads = jax.grad(lambda pp: loss_and_terms(pp, x_pde)[0])(params)
    step = step + 1
    m = jax.tree_util.tree_map(lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: beta2 * vi + (1.0 - beta2) * (gi * gi), v, grads)
    mhat = jax.tree_util.tree_map(lambda mi: mi / (1.0 - beta1**step), m)
    vhat = jax.tree_util.tree_map(lambda vi: vi / (1.0 - beta2**step), v)
    params = jax.tree_util.tree_map(lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps), params, mhat, vhat)
    return params, (step, m, v), loss, aux


def sample_pde_points(key, n: int, dtype=jnp.float64, method: str = "random"):
    """Sample points in [0,1]^5.

    The DeepXDE script used Hammersley. For simplicity and JIT-friendliness this
    script defaults to random uniform sampling; 'hammersley' is also implemented.
    """
    if method == "random":
        return jax.random.uniform(key, (n, 5), dtype=dtype)
    if method == "hammersley":
        pts = hammersley_np(n, 5).astype(np.float64 if dtype == jnp.float64 else np.float32)
        return jnp.asarray(pts, dtype=dtype)
    raise ValueError(f"Unknown sampling method: {method}")


def _van_der_corput(n: int, base: int):
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = 0.0
        denom = 1.0
        k = i + 1
        while k:
            k, rem = divmod(k, base)
            denom *= base
            x += rem / denom
        seq[i] = x
    return seq


def hammersley_np(n: int, dim: int):
    primes = [2, 3, 5, 7, 11, 13, 17]
    pts = np.empty((n, dim), dtype=np.float64)
    pts[:, 0] = (np.arange(n, dtype=np.float64) + 0.5) / n
    for j in range(1, dim):
        pts[:, j] = _van_der_corput(n, primes[j - 1])
    return pts


def train_adam(params, key, args):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt_state = (jnp.asarray(0, dtype=jnp.int64), m, v)

    for step in range(1, args.adam_steps + 1):
        if step == 1 or step % args.resample_every == 0:
            key, subkey = jax.random.split(key)
            x_pde = sample_pde_points(subkey, args.n_pde, method=args.sampling)
        params, opt_state, loss, (loss_u, loss_v) = adam_update(
            params,
            opt_state,
            x_pde,
            args.adam_lr,
            0.9,
            0.999,
            1.0e-8,
        )
        if step == 1 or step % args.print_every == 0:
            print(
                f"adam step {step:7d} loss={float(loss):.6e} "
                f"Ru={float(loss_u):.3e} Rv={float(loss_v):.3e}",
                flush=True,
            )
    return params, key


def train_ssbroyden(params, key, args):
    if not HAVE_CRUNCH:
        print("[warn] Skipping SSBroyden because Crunch could not be imported.")
        return params, key
    if args.ssb_total_iters <= 0:
        return params, key

    weights, unflatten = ravel_pytree(params)
    n_params = int(weights.shape[0])
    print(f"SSBroyden parameter count: {n_params}")
    estimated_gb = (n_params * n_params * 8) / 1.0e9
    print(f"Dense inverse-H memory estimate: {estimated_gb:.3f} GB")

    H0 = jnp.eye(weights.shape[0], dtype=weights.dtype)

    @partial(jax.jit, static_argnames=("unflatten_static",))
    def flat_loss(weights_flat, x_pde, unflatten_static):
        pp = unflatten_static(weights_flat)
        return loss_and_terms(pp, x_pde)[0]

    @partial(jax.jit, static_argnames=("unflatten_static", "static_options"))
    def ssbroyden_step(weights_in, H0_in, x_pde, unflatten_static, static_options):
        current_options = dict(static_options)
        current_options["initial_H"] = H0_in
        result = minimize(
            fun=flat_loss,
            x0=weights_in,
            args=(x_pde, unflatten_static),
            method="BFGS",
            options=current_options,
        )
        H_candidate = (result.hess_inv + result.hess_inv.T) / 2.0

        def reset_h(_):
            return jnp.eye(H_candidate.shape[0], dtype=H_candidate.dtype)

        def keep_h(h):
            return h

        # Cholesky returns NaNs if not positive definite under JAX.
        L = jnp.linalg.cholesky(H_candidate)
        bad_H = jnp.any(jnp.isnan(L)) | jnp.any(~jnp.isfinite(H_candidate))
        H_out = jax.lax.cond(bad_H, reset_h, keep_h, H_candidate)
        return result.x, H_out, result.fun, result.nit, result.success, result.status

    options = dict(static_options_SSBroyden)
    options["maxiter"] = args.ssb_block_iters
    options["gtol"] = args.ssb_gtol
    options_tuple = tuple(options.items())

    n_blocks = max(1, math.ceil(args.ssb_total_iters / args.ssb_block_iters))
    for block in range(1, n_blocks + 1):
        key, subkey = jax.random.split(key)
        x_pde = sample_pde_points(subkey, args.n_pde, method=args.sampling)
        if args.ssb_reset_h_on_resample and block > 1:
            H0 = jnp.eye(weights.shape[0], dtype=weights.dtype)

        weights, H0, loss, nit, success, status = ssbroyden_step(
            weights,
            H0,
            x_pde,
            unflatten_static=unflatten,
            static_options=options_tuple,
        )
        print(
            f"ssb block {block:4d}/{n_blocks} loss={float(loss):.6e} "
            f"nit={int(nit)} success={bool(success)} status={int(status)}",
            flush=True,
        )

    return unflatten(weights), key


def save_checkpoint(params, path: Path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x), params),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": asdict(CFG),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {path}")


def load_checkpoint(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), payload["params"])
    return params, payload


def _jax_dtype_to_ir_dtype(dtype):
    """Map JAX/NumPy dtype to onnx_ir.DataType."""
    import onnx_ir as ir

    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.float64):
        return ir.DataType.DOUBLE
    if dtype == np.dtype(np.float32):
        return ir.DataType.FLOAT
    if dtype == np.dtype(np.int64):
        return ir.DataType.INT64
    if dtype == np.dtype(np.int32):
        return ir.DataType.INT32
    if dtype == np.dtype(np.bool_):
        return ir.DataType.BOOL

    raise TypeError(f"Unsupported dtype for ONNX export: {dtype}")


def _const_value_ir_type(value):
    """Return TensorType from a Value const_value if available."""
    import onnx_ir as ir

    const_value = getattr(value, "const_value", None)
    if const_value is None:
        return None

    dtype = getattr(const_value, "dtype", None)
    if dtype is None:
        return None

    return ir.TensorType(dtype)


def _repair_onnx_ir_value_types(model_ir, default_float_dtype):
    """Repair ONNX IR Values that have shape but missing type.

    onnx_ir warns when serializing a Value with a known shape but no known type.
    This pass sets missing TensorType annotations before serialization.

    It handles:
      - constants using their const_value dtype
      - Shape/Size-like ops as int64
      - comparison/logical ops as bool
      - normal tensor ops by propagating the first known input dtype
      - remaining unknown numeric tensors as the export float dtype
    """
    import onnx_ir as ir

    default_float_type = ir.TensorType(_jax_dtype_to_ir_dtype(default_float_dtype))
    int64_type = ir.TensorType(ir.DataType.INT64)
    bool_type = ir.TensorType(ir.DataType.BOOL)

    def needs_type(value):
        return (
            value is not None
            and getattr(value, "shape", None) is not None
            and getattr(value, "type", None) is None
        )

    def set_type(value, type_):
        if needs_type(value) and type_ is not None:
            value.type = type_
            return 1
        return 0

    def first_input_type(node):
        for value in node.inputs:
            if value is not None and getattr(value, "type", None) is not None:
                return value.type
        return None

    def repair_graph(graph):
        repaired = 0

        # Graph inputs/outputs should usually already be typed, but keep this
        # defensive repair for exported graph outputs.
        for value in list(graph.inputs) + list(graph.outputs):
            repaired += set_type(value, default_float_type)

        # Initializers and constants should use their actual tensor dtype.
        for value in graph.initializers.values():
            repaired += set_type(value, _const_value_ir_type(value))

        # Iterate more than once so types can propagate through chains such as
        # Shape -> Gather -> Unsqueeze/Concat.
        for _ in range(8):
            changed = 0

            for node in graph.all_nodes():
                op = node.op_type

                for out in node.outputs:
                    if not needs_type(out):
                        continue

                    const_type = _const_value_ir_type(out)
                    if const_type is not None:
                        changed += set_type(out, const_type)
                        continue

                    if op in {
                        "Shape",
                        "Size",
                        "NonZero",
                        "ArgMax",
                        "ArgMin",
                    }:
                        changed += set_type(out, int64_type)
                        continue

                    if op in {
                        "Equal",
                        "Greater",
                        "GreaterOrEqual",
                        "Less",
                        "LessOrEqual",
                        "And",
                        "Or",
                        "Not",
                        "Xor",
                        "IsInf",
                        "IsNaN",
                    }:
                        changed += set_type(out, bool_type)
                        continue

                    input_type = first_input_type(node)
                    if input_type is not None:
                        changed += set_type(out, input_type)
                        continue

                repaired += changed

            if changed == 0:
                break

        # Final fallback: if a shaped value is still untyped, assume it is a
        # floating tensor. This should only hit numeric intermediates, not shape
        # tensors, because Shape/Gather/etc. should have propagated int64 above.
        for node in graph.all_nodes():
            for out in node.outputs:
                repaired += set_type(out, default_float_type)

        for value in graph.outputs:
            repaired += set_type(value, default_float_type)

        return repaired

    return repair_graph(model_ir.graph)


def export_onnx(params, outdir: Path):
    """Export browser/WebGPU-friendly float32 ONNX models.

    Training stays float64. Exported ONNX models are float32 only.
    """
    try:
        import onnx_ir as ir
        from jax2onnx import to_onnx
    except Exception as exc:
        print(f"[warn] ONNX export dependencies unavailable; skipping ONNX export: {exc}")
        return False, False

    outdir.mkdir(parents=True, exist_ok=True)

    export_dtype = jnp.float32

    params_export = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=export_dtype),
        params,
    )

    input_spec = [jax.ShapeDtypeStruct(("B", CFG.input_dim), export_dtype)]

    def forward_export(x):
        x = jnp.asarray(x, dtype=export_dtype)
        y = jax.vmap(physical_apply_single, in_axes=(None, 0))(params_export, x)

        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        speed = jnp.sqrt(u * u + v * v)

        return jnp.concatenate([u, v, p, speed], axis=1).astype(export_dtype)

    def residual_export(x):
        x = jnp.asarray(x, dtype=export_dtype)
        r = jax.vmap(residual_single, in_axes=(None, 0))(params_export, x)

        ru = r[:, 0:1]
        rv = r[:, 1:2]
        rmag = jnp.sqrt(ru * ru + rv * rv)

        return jnp.concatenate([ru, rv, rmag], axis=1).astype(export_dtype)

    def export_one(fn, output_path, output_name):
        model_ir = to_onnx(
            fn,
            input_spec,
            return_mode="ir",
            input_names=["x"],
            output_names=[output_name],
            export_mode="web",
            enable_double_precision=False,
        )

        repaired = _repair_onnx_ir_value_types(model_ir, export_dtype)
        print(f"repaired {repaired} ONNX IR value types before serialization")

        ir.save(model_ir, str(output_path))
        print(f"wrote {output_path}")

    ok_forward = False
    ok_residual = False

    try:
        export_one(
            forward_export,
            outdir / "lid_cavity_forward.onnx",
            "uvp_speed",
        )
        ok_forward = True
    except Exception as exc:
        print(f"[warn] forward ONNX export failed: {exc}")

    try:
        export_one(
            residual_export,
            outdir / "lid_cavity_residual.onnx",
            "ru_rv_rmag",
        )
        ok_residual = True
    except Exception as exc:
        print(f"[warn] residual ONNX export failed: {exc}")

    return ok_forward, ok_residual

def write_metadata(model_dir: Path, args, onnx_forward: bool, onnx_residual: bool):
    metadata = {
        "id": "lid_cavity_trap_re_depth_tri_v1",
        "name": "Parametric trapped lid-driven cavity PINN",
        "equations": {
            "momentum_x": "u u_x + (v/D) u_y - (1/Re)(u_xx + u_yy/D^2) + p_x = 0",
            "momentum_y": "u v_x + (v/D) v_y - (1/Re)(v_xx + v_yy/D^2) + p_y/D = 0",
            "continuity": "enforced by streamfunction ansatz",
        },
        "inputs": ["x", "y", "ReNorm", "DNorm", "triNorm"],
        "forward_outputs": ["u", "v", "p", "speed"],
        "residual_outputs": ["Ru_weighted", "Rv_weighted", "Rmag_weighted"],
        "parameter_ranges": {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "Re": [CFG.re_min, CFG.re_max],
            "D": [CFG.d_min, CFG.d_max],
            "tri": [CFG.tri_min, CFG.tri_max],
        },
        "models": {
            "forward": "lid_cavity_forward.onnx" if onnx_forward else None,
            "residual": "lid_cavity_residual.onnx" if onnx_residual else None,
        },
        "training_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": asdict(CFG),
        "recommended_execution_provider": "webgpu",
        "fallback_execution_provider": "wasm",
    }
    path = model_dir / "metadata.json"
    with path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {path}")


def copy_models_for_web(model_dir: Path, web_public_model_dir: Path):
    web_public_model_dir.mkdir(parents=True, exist_ok=True)

    for src in model_dir.glob("*.onnx"):
        shutil.copy2(src, web_public_model_dir / src.name)
        print(f"copied {src} -> {web_public_model_dir / src.name}")

    metadata = model_dir / "metadata.json"
    if metadata.exists():
        shutil.copy2(metadata, web_public_model_dir / "metadata.json")
        print(f"copied {metadata} -> {web_public_model_dir / 'metadata.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--depth", type=int, default=4, help="number of hidden layers")
    p.add_argument("--n-pde", type=int, default=4096)
    p.add_argument("--sampling", choices=["random", "hammersley"], default="random")
    p.add_argument("--adam-steps", type=int, default=1000)
    p.add_argument("--adam-lr", type=float, default=5.0e-4)
    p.add_argument("--resample-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--ssb-total-iters", type=int, default=0, help="set >0 to run Crunch SSBroyden")
    p.add_argument("--ssb-block-iters", type=int, default=100)
    p.add_argument("--ssb-gtol", type=float, default=1.0e-10)
    p.add_argument("--ssb-reset-h-on-resample", action="store_true")
    p.add_argument("--outdir", type=Path, default=THIS_DIR / "models")
    p.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    if args.resume is not None:
        params, _ = load_checkpoint(args.resume)
        print(f"loaded checkpoint {args.resume}")
    else:
        layer_sizes = [CFG.input_dim] + [args.width] * args.depth + [CFG.raw_output_dim]
        params = init_mlp(key, layer_sizes)
        print(f"initialized MLP: {layer_sizes}")

    # Compile once and report initial loss.
    key, subkey = jax.random.split(key)
    x0 = sample_pde_points(subkey, min(args.n_pde, 1024), method=args.sampling)
    initial_loss, (lu, lv) = loss_and_terms(params, x0)
    print(f"initial loss={float(initial_loss):.6e} Ru={float(lu):.3e} Rv={float(lv):.3e}")

    if args.adam_steps > 0:
        params, key = train_adam(params, key, args)

    if args.ssb_total_iters > 0:
        params, key = train_ssbroyden(params, key, args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(params, args.outdir / "lid_cavity_params.pkl", args)

    ok_forward = False
    ok_residual = False
    if args.export_onnx:
        ok_forward, ok_residual = export_onnx(params, args.outdir)

    write_metadata(args.outdir, args, ok_forward, ok_residual)
    copy_models_for_web(args.outdir, THIS_DIR / "web" / "public" / "models")


if __name__ == "__main__":
    main()
