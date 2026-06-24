# Parametric Lid-Driven Cavity PINN Demo

This directory contains a JAX rewrite of the parametric trapped/lid-driven cavity PINN from `CavityTrapREDepthSSB.py`, a bundled SSBroyden optimizer workflow, ONNX export, and a browser-native GitHub Pages demo.

## Physical/model formulation

Inputs are normalized:

```text
[x, y, ReNorm, DNorm, triNorm] in [0, 1]^5
```

The physical parameters are decoded as:

```text
Re  = 900 + 200 * ReNorm
D   = 1.9 + 1.4 * DNorm
tri = 0.01 * triNorm
```

The raw neural network predicts:

```text
[psi_prime, p]
```

The physical velocity is constructed with the same hard streamfunction/lid ansatz used in the DeepXDE script, so boundary conditions and scaled incompressibility are built into the representation. The PDE loss minimizes the two weighted momentum residuals.

## Directory layout

```text
demo/
  train_lid_driven_cavity_jax.py
  NABLA-SciML/Crunch/              # bundled SSBroyden optimizer from the attached zip
  models/                          # trained checkpoint, metadata, ONNX files
  web/                             # Vite + ONNX Runtime WebGPU app
  scripts/validate_onnx.py
  github_pages_workflow.yml         # copy to repo-root .github/workflows/deploy-pages.yml
```

## Train and export

For a quick smoke test:

```bash
cd demo
python train_lid_driven_cavity_jax.py \
  --width 16 \
  --depth 3 \
  --n-pde 512 \
  --adam-steps 10 \
  --export-onnx
```

For a more meaningful demo model, increase the settings, for example:

```bash
cd demo
python train_lid_driven_cavity_jax.py \
  --width 24 \
  --depth 4 \
  --n-pde 8192 \
  --adam-steps 5000 \
  --resample-every 250 \
  --ssb-total-iters 2000 \
  --ssb-block-iters 200 \
  --export-onnx
```

The SSBroyden step uses a dense inverse-Hessian approximation, so memory scales as `n_parameters^2`. Start small before increasing network width/depth.

The script writes:

```text
models/lid_cavity_params.pkl
models/lid_cavity_forward.onnx
models/lid_cavity_residual.onnx
models/metadata.json
web/public/models/*
```

## Run the browser demo locally

```bash
cd demo/web
npm install
npm run dev
```

Open the URL printed by Vite. The demo uses ONNX Runtime WebGPU and falls back only if you modify `src/main.js` to use WASM.

## Deploy on GitHub Pages

1. Train/export the ONNX models so `demo/models/*.onnx` exists.
2. Copy `demo/github_pages_workflow.yml` to `.github/workflows/deploy-pages.yml` at the repository root.
3. In GitHub repository settings, set Pages source to GitHub Actions.
4. Push to `main`.

The workflow builds only `demo/web` and deploys the static app.

## Validate ONNX models

```bash
cd demo
python scripts/validate_onnx.py
```
