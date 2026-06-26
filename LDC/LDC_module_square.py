import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import skopt
import pytorch_optimizer
import matplotlib.pyplot as plt
import shutil

# -------------------------
# Config & Setup
# -------------------------
SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('model', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# -------------------------
# LDC Physics parameters — single fixed Re
# -------------------------
Re = 5_000.0
dx = np.sqrt(1.e-3)
dy = np.sqrt(1.e-2)

# -------------------------
# Training hyperparameters
# -------------------------
# Stage A (SOAP + CUDA graph)
SOAP_STEPS      = 0
SOAP_WARMUP_STEPS = 30
SOAP_LR         = 2e-3
PDE_BATCH       = 2**18 - 1
FRAC_CORNER     = 0.0
CORNER_EXCL_RADIUS = 2e-2

# Stage B (SSBroyden2)
USE_SSBROYDEN2  = True
SSB_STEPS       = 10_000
SSB_LINE_SEARCH = "strong_wolfe"
SSB_LR          = 1.0
SSB_C1          = 1e-4
SSB_C2          = 0.9
SSB_LS_MAXITER  = 20
SSB_ZOOM_MAXITER= 40
SSB_AMAX        = 10
SSB_FIXED_PDE_N = PDE_BATCH

# Test / logging
TEST_PDE_N   = SSB_FIXED_PDE_N
TEST_EVERY   = 500
RECORD_EVERY = 100

# -------------------------
# Model  (in_dim=2: x, y only)
# -------------------------
class PINN(nn.Module):
    def __init__(self, in_dim=2, h_dim=50, out_dim=2):
        super().__init__()
        self.linear1    = nn.Linear(in_dim, h_dim); self.act1 = nn.Tanh()
        self.linear2    = nn.Linear(h_dim,  h_dim); self.act2 = nn.Tanh()
        self.linear3    = nn.Linear(h_dim,  h_dim); self.act3 = nn.Tanh()
        #self.linear4    = nn.Linear(h_dim,  h_dim); self.act4 = nn.Tanh()
        #self.linear5    = nn.Linear(h_dim,  h_dim); self.act5 = nn.Tanh()
        self.linearLast = nn.Linear(h_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in [self.linear1,
                  self.linear2,
                  self.linear3,
                  #self.linear4,
                  #self.linear5,
                  self.linearLast]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        x = self.act3(self.linear3(x))
        #x = self.act4(self.linear4(x))
        #x = self.act5(self.linear5(x))
        return self.linearLast(x)

# -------------------------
# Sampling  (2-D points only)
# -------------------------
def hammersley_sequence(n_samples, dim):
    sampler = skopt.sampler.Hammersly()
    space   = [(0.0, 1.0)] * dim
    return np.asarray(sampler.generate(space, n_samples + 1)[1:], dtype=np.float64)

def sobol_sequence(n_samples, dim):
    sampler = skopt.sampler.Sobol()
    space   = [(0.0, 1.0)] * dim
    return np.asarray(sampler.generate(space, n_samples + 1)[1:], dtype=np.float64)

def add_corner_points(X_train, N_corner=10000, eps=0.05, device=None):
    if device is None:
        device = X_train.device
    tl_x = torch.rand(N_corner, 1, dtype=torch.float64, device=device) * eps
    tl_y = 1.0 - torch.rand(N_corner, 1, dtype=torch.float64, device=device) * eps
    tl_points = torch.cat([tl_x, tl_y], dim=1)

    tr_x = 1.0 - torch.rand(N_corner, 1, dtype=torch.float64, device=device) * eps
    tr_y = 1.0  - torch.rand(N_corner, 1, dtype=torch.float64, device=device) * eps
    tr_points = torch.cat([tr_x, tr_y], dim=1)

    X_enhanced = torch.cat([X_train, tl_points, tr_points], dim=0)
    perm = torch.randperm(X_enhanced.shape[0], device=device)
    return X_enhanced[perm]

def _exclude_corners(X):
    corners = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]],
                            dtype=torch.float64, device=device)
    # For each point, compute min distance to any corner
    dists = torch.cdist(X, corners)          # (N, 4)
    min_dist = dists.min(dim=1).values       # (N,)
    return X[min_dist > CORNER_EXCL_RADIUS]

def sample_pde_points(n_pts):
    # Oversample to account for corner exclusions, then trim to exact size
    pts = []
    total = 0
    while total < n_pts:
        batch = np.random.rand(n_pts, 2)
        X = torch.tensor(batch, dtype=torch.float64, device=device)
        X = _exclude_corners(X)
        pts.append(X)
        total += X.shape[0]
    return torch.cat(pts, dim=0)[:n_pts]

def sample_test_points(n_pts):
    pts = []
    total = 0
    while total < n_pts:
        batch = np.random.rand(n_pts, 2)
        X = torch.tensor(batch, dtype=torch.float64, device=device)
        X = _exclude_corners(X)
        pts.append(X)
        total += X.shape[0]
    return torch.cat(pts, dim=0)[:n_pts]


"""
def sample_pde_points(n_pts):
    np_pts = sobol_sequence(n_pts, 2)          # 2-D
    X = torch.tensor(np_pts, dtype=torch.float64, device=device)
    return add_corner_points(X, N_corner=round(FRAC_CORNER * n_pts), eps=0.05, device=device)

def sample_test_points(n_pts):
    np_pts = hammersley_sequence(n_pts, 2)     # 2-D
    return torch.tensor(np_pts, dtype=torch.float64, device=device)
"""


# -------------------------
# LDC Physics & Residuals  (Re is now a global scalar)
# -------------------------
def output_transform_cavity_flow(inputs, outputs):
    """Returns (u, v, p) with BCs exactly enforced."""
    x, y = inputs.split(1, dim=1)
    bcv  = 16 * x * (1 - x) * y * (1 - y)
    ExpB = torch.exp(-(1 - y)**2 / dy**2)

    psilid   = (y-1)*y**2 * (1-torch.exp(-(x-1)**2/dx**2)) * (1-torch.exp(-x**2/dx**2)) * ExpB
    psilid_x = ((y-1)*y**2
                * (2*(x-1)/dx**2 * torch.exp(-(x-1)**2/dx**2))
                * (1-torch.exp(-x**2/dx**2)) * ExpB
                + (y-1)*y**2
                * (1-torch.exp(-(x-1)**2/dx**2))
                * (2*x/dx**2 * torch.exp(-x**2/dx**2)) * ExpB)
    psilid_y = ((y**2 + 2*y*(y-1))
                * (1-torch.exp(-(x-1)**2/dx**2))
                * (1-torch.exp(-x**2/dx**2)) * ExpB
                + psilid * 2*(1-y)/dy**2)

    dbcv_x = 16 * (1 - 2*x) * y * (1 - y)
    dbcv_y = 16 * x * (1 - x) * (1 - 2*y)

    psiprime_grad = torch.autograd.grad(
        outputs[:, 0:1], inputs,
        torch.ones_like(outputs[:, 0:1]), create_graph=True
    )[0]
    psiprime_x, psiprime_y = psiprime_grad[:, 0:1], psiprime_grad[:, 1:2]

    u = psilid_y + 2*bcv*dbcv_y * outputs[:, 0:1] + bcv**2 * psiprime_y
    v = -(psilid_x + 2*bcv*dbcv_x * outputs[:, 0:1] + bcv**2 * psiprime_x)
    p = outputs[:, 1:2]
    return u, v, p

def fp_pde(model, inputs):
    """Returns PDE residuals (momentum x, momentum y)."""
    x = inputs[:, 0:1].clone().detach().requires_grad_(True)
    y = inputs[:, 1:2].clone().detach().requires_grad_(True)
    inputs_tracked = torch.cat([x, y], dim=1)
    outputs = model(inputs_tracked)

    bcv  = 16 * x * (1 - x) * y * (1 - y)
    ExpB = torch.exp(-(1 - y)**2 / dy**2)

    psilid   = (y-1)*y**2 * (1-torch.exp(-(x-1)**2/dx**2)) * (1-torch.exp(-x**2/dx**2)) * ExpB
    psilid_x = ((y-1)*y**2
                * (2*(x-1)/dx**2 * torch.exp(-(x-1)**2/dx**2))
                * (1-torch.exp(-x**2/dx**2)) * ExpB
                + (y-1)*y**2
                * (1-torch.exp(-(x-1)**2/dx**2))
                * (2*x/dx**2 * torch.exp(-x**2/dx**2)) * ExpB)
    psilid_y = ((y**2 + 2*y*(y-1))
                * (1-torch.exp(-(x-1)**2/dx**2))
                * (1-torch.exp(-x**2/dx**2)) * ExpB
                + psilid * 2*(1-y)/dy**2)

    dbcv_x = 16 * (1 - 2*x) * y * (1 - y)
    dbcv_y = 16 * x * (1 - x) * (1 - 2*y)

    psiprime_x, psiprime_y = torch.autograd.grad(
        outputs[:, 0:1], (x, y),
        torch.ones_like(outputs[:, 0:1]), create_graph=True
    )

    u = psilid_y + 2*bcv*dbcv_y * outputs[:, 0:1] + bcv**2 * psiprime_y
    v = -(psilid_x + 2*bcv*dbcv_x * outputs[:, 0:1] + bcv**2 * psiprime_x)
    p = outputs[:, 1:2]

    du_x, du_y = torch.autograd.grad(u, (x, y), torch.ones_like(u), create_graph=True)
    dv_x, dv_y = torch.autograd.grad(v, (x, y), torch.ones_like(v), create_graph=True)
    du_xx = torch.autograd.grad(du_x, x, torch.ones_like(du_x), create_graph=True)[0]
    du_yy = torch.autograd.grad(du_y, y, torch.ones_like(du_y), create_graph=True)[0]
    dv_xx = torch.autograd.grad(dv_x, x, torch.ones_like(dv_x), create_graph=True)[0]
    dv_yy = torch.autograd.grad(dv_y, y, torch.ones_like(dv_y), create_graph=True)[0]
    dp_x, dp_y = torch.autograd.grad(p, (x, y), torch.ones_like(p), create_graph=True)

    res1 = u*du_x + v*du_y - (1/Re)*(du_xx + du_yy) + dp_x
    res2 = u*dv_x + v*dv_y - (1/Re)*(dv_xx + dv_yy) + dp_y

    # Smooth corner masks
    radius = 1e-2
    d2_tl  = x**2 + (1.0 - y)**2
    d2_tr  = (1.0 - x)**2 + (1.0 - y)**2
    mask_tl = 1.0 - torch.exp(-d2_tl / radius**2)
    mask_tr = 1.0 - torch.exp(-d2_tr / radius**2)
    return mask_tl*mask_tr*res1, mask_tl*mask_tr*res2

def compute_test_loss(model, X_test, max_batch=65536):
    model.eval()
    res1_sq, res2_sq, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(True):
        for s in range(0, X_test.shape[0], max_batch):
            X_b = X_test[s:s+max_batch]
            r1, r2 = fp_pde(model, X_b)
            res1_sq += r1.pow(2).sum().item()
            res2_sq += r2.pow(2).sum().item()
            count   += r1.numel()
    model.train()
    tl1 = res1_sq / max(1, count)
    tl2 = res2_sq / max(1, count)
    return float(tl1 + tl2), float(tl1), float(tl2)

# -------------------------
# Line Search & SSBroyden2  (unchanged from original)
# -------------------------
def _phi_and_derphi(eval_fg, xk, pk, alpha: float):
    x = xk + alpha * pk
    f, g = eval_fg(x)
    return f, g, torch.dot(g, pk)

def _zoom(eval_fg, xk, pk, phi0, derphi0, c1, c2,
          alo, ahi, phi_alo, derphi_alo, maxiter=20):
    for _ in range(maxiter):
        aj = 0.5 * (alo + ahi)
        phi_aj, g_aj, derphi_aj = _phi_and_derphi(eval_fg, xk, pk, aj)
        if (phi_aj > phi0 + c1*aj*derphi0) or (phi_aj >= phi_alo):
            ahi = aj
        else:
            if abs(float(derphi_aj)) <= -c2 * float(derphi0):
                return aj, phi_aj, g_aj
            if float(derphi_aj) * (ahi - alo) >= 0:
                ahi = alo
            alo, phi_alo, derphi_alo = aj, phi_aj, derphi_aj
    phi_alo2, g_alo2, _ = _phi_and_derphi(eval_fg, xk, pk, alo)
    return alo, phi_alo2, g_alo2

def strong_wolfe_line_search(eval_fg, xk, pk, gfk, old_fval,
                              c1=1e-4, c2=0.9, amax=None,
                              maxiter=10, zoom_maxiter=20, alpha1=1.0):
    derphi0 = torch.dot(gfk, pk)
    if (not torch.isfinite(derphi0)) or float(derphi0) >= 0.0:
        pk      = -gfk
        derphi0 = -torch.dot(gfk, gfk)
    phi0   = old_fval
    alpha0 = 0.0
    alpha1 = float(alpha1)
    if amax is not None:
        alpha1 = min(alpha1, float(amax))
    phi_a0, derphi_a0 = phi0, derphi0
    phi_a1, g_a1, derphi_a1 = _phi_and_derphi(eval_fg, xk, pk, alpha1)
    for i in range(maxiter):
        if (phi_a1 > phi0 + c1*alpha1*derphi0) or (i > 0 and phi_a1 >= phi_a0):
            a_s, p_s, g_s = _zoom(eval_fg, xk, pk, phi0, derphi0,
                                   c1, c2, alpha0, alpha1, phi_a0, derphi_a0,
                                   maxiter=zoom_maxiter)
            return float(a_s), xk + a_s*pk, p_s, g_s
        if abs(float(derphi_a1)) <= -c2 * float(derphi0):
            return float(alpha1), xk + alpha1*pk, phi_a1, g_a1
        if float(derphi_a1) >= 0.0:
            a_s, p_s, g_s = _zoom(eval_fg, xk, pk, phi0, derphi0,
                                   c1, c2, alpha1, alpha0, phi_a1, derphi_a1,
                                   maxiter=zoom_maxiter)
            return float(a_s), xk + a_s*pk, p_s, g_s
        alpha0, phi_a0, derphi_a0 = alpha1, phi_a1, derphi_a1
        alpha1 *= 2.0
        if amax is not None:
            alpha1 = min(alpha1, float(amax))
        phi_a1, g_a1, derphi_a1 = _phi_and_derphi(eval_fg, xk, pk, alpha1)
    return float(alpha1), xk + alpha1*pk, phi_a1, g_a1

class SSBroyden2(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gtol=1e-10, xrtol=0.0,
                 line_search="strong_wolfe", c1=1e-4, c2=0.9, backtrack=0.5,
                 ls_max_steps=25, wolfe_maxiter=10, zoom_maxiter=20,
                 amax=None, initial_scale=False, eps=1e-30,
                 dtype=torch.float64, device=None):
        super().__init__(params, defaults={})
        self.lr, self.gtol, self.xrtol           = lr, gtol, xrtol
        self.line_search, self.c1, self.c2       = line_search, c1, c2
        self.backtrack                           = backtrack
        self.ls_max_steps, self.wolfe_maxiter    = ls_max_steps, wolfe_maxiter
        self.zoom_maxiter                        = zoom_maxiter
        self.amax, self.initial_scale, self.eps  = amax, initial_scale, eps
        self.dtype                               = dtype
        self._params  = list(self.param_groups[0]["params"])
        self.device   = device if device is not None else self._params[0].device
        self._numels  = [p.numel() for p in self._params]
        self._P       = int(sum(self._numels))
        self.state["H"] = torch.eye(self._P, device=self.device, dtype=self.dtype)
        self.state["x"] = self._gather_flat_params().detach().clone()
        self.state["k"] = 0

    def _gather_flat_params(self):
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    def _set_flat_params_(self, flat):
        with torch.no_grad():
            off = 0
            for p, n in zip(self._params, self._numels):
                p.copy_(flat[off:off+n].view_as(p)); off += n

    def _gather_flat_grad(self):
        return torch.cat([
            p.grad.detach().reshape(-1) if p.grad is not None
            else torch.zeros(n, device=self.device, dtype=self.dtype)
            for p, n in zip(self._params, self._numels)
        ])

    def _eval_loss_and_grad(self, closure):
        loss = closure()
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=self.device, dtype=self.dtype)
        return loss.detach(), self._gather_flat_grad()

    def step(self, closure):
        self.zero_grad(set_to_none=True)
        fk, gk   = self._eval_loss_and_grad(closure)
        xk, Hk, N = self._gather_flat_params().detach(), self.state["H"], self._P
        if torch.linalg.vector_norm(gk).item() <= self.gtol:
            return fk
        pk = -(Hk @ gk)

        def eval_fg_at(x_flat):
            self._set_flat_params_(x_flat)
            self.zero_grad(set_to_none=True)
            return self._eval_loss_and_grad(closure)

        alpha, xkp1, fkp1, gkp1 = strong_wolfe_line_search(
            eval_fg_at, xk, pk, gk, fk,
            c1=self.c1, c2=self.c2, amax=self.amax,
            maxiter=self.wolfe_maxiter, zoom_maxiter=self.zoom_maxiter,
            alpha1=float(self.lr)
        )
        self._set_flat_params_(xkp1)

        sk, yk = xkp1 - xk, gkp1 - gk
        ys = torch.dot(yk, sk).item()
        if abs(ys) < self.eps:
            self.state["x"] = xkp1.detach().clone(); self.state["k"] += 1
            return fkp1

        rhok   = 1.0 / ys
        Hkyk   = Hk @ yk
        ykHkyk = torch.dot(yk, Hkyk).item()
        if abs(ykHkyk) < self.eps:
            self.state["x"] = xkp1.detach().clone(); self.state["k"] += 1
            return fkp1

        hk    = ykHkyk * rhok
        bk    = -alpha * rhok * torch.dot(sk, gk).item()
        ak    = bk * hk - 1.0
        denom = (1.0 + ak) or self.eps
        rad   = max(abs(ak) / denom, 0.0)
        rhokm = min(1.0, hk * (1.0 - math.sqrt(rad)))

        thetakm = 0.0 if abs(ak) < self.eps else (rhokm - 1.0) / ak
        thetakp = (1.0 / rhokm) if abs(rhokm) > self.eps else (1.0 / self.eps)
        inner   = thetakp if abs(bk) < self.eps else (1.0 - bk) / bk
        thetak  = max(thetakm, min(thetakp, inner))

        rhokk    = min(1.0, (1.0 / bk) if abs(bk) > self.eps else (1.0 / self.eps))
        sigmak   = 1.0 + thetak * ak
        exp      = 1.0 / (1.0 - float(N))
        sigmaknm1= abs(sigmak)**exp if abs(sigmak) > 0 else 0.0

        tauk = (min(rhokk * sigmaknm1, sigmak) if thetak <= 0.0
                else rhokk * min(sigmaknm1, 1.0 / thetak))

        vk     = (sk * rhok) - (Hkyk / ykHkyk)
        denom3 = (1.0 + ak * thetak) or self.eps
        phik   = (1.0 - thetak) / denom3
        tauk   = tauk if abs(tauk) > self.eps else (self.eps if tauk >= 0 else -self.eps)

        H_term  = Hk - torch.outer(Hkyk, Hkyk)/ykHkyk + (phik*ykHkyk)*torch.outer(vk, vk)
        Hk_new  = (H_term / tauk) + (rhok * torch.outer(sk, sk))

        self.state["H"] = Hk_new
        self.state["x"] = xkp1.detach().clone()
        self.state["k"] += 1
        return fkp1

# -------------------------
# Plotting
# -------------------------
def generate_plots(model, device, loss_data=None, suffix=""):
    plt.rcParams.update({'font.size': 14})

    # ── 1. Loss history ──────────────────────────────────────────────────────
    if loss_data and len(loss_data.get('StepsTraining', [])) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.set_tight_layout(True)
        ax.semilogy(loss_data['StepsTraining'], loss_data['TrainLoss1'],
                    color='blue', lw=2, label='Train – Eq1 (x-mom)')
        ax.semilogy(loss_data['StepsTraining'], loss_data['TrainLoss2'],
                    color='red',  lw=2, label='Train – Eq2 (y-mom)')
        ax.semilogy(loss_data['EpochsTest'],    loss_data['TestLoss1'],
                    'xb', ms=6, label='Test – Eq1')
        ax.semilogy(loss_data['EpochsTest'],    loss_data['TestLoss2'],
                    'xr', ms=6, label='Test – Eq2')
        ax.set_xlabel('Iteration'); ax.set_ylabel('MSE Loss')
        ax.set_title(f'Loss History  (Re = {Re:.0f})')
        ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
        plt.savefig(f"figures/LossHistory{suffix}.png", dpi=150)
        plt.close()

    # ── Build evaluation grid ────────────────────────────────────────────────
    n_plot  = 200
    xVals   = np.linspace(0, 1, n_plot)
    yVals   = np.linspace(0, 1, n_plot)
    xx, yy  = np.meshgrid(xVals, yVals, indexing="ij")

    x_grid = torch.tensor(xx, dtype=torch.float64, device=device).reshape(-1, 1)
    y_grid = torch.tensor(yy, dtype=torch.float64, device=device).reshape(-1, 1)

    model.eval()
    with torch.set_grad_enabled(True):
        inputs  = torch.cat([x_grid, y_grid], dim=1).requires_grad_(True)
        outputs = model(inputs)
        u, v, p = output_transform_cavity_flow(inputs, outputs)
        res1, res2 = fp_pde(model, inputs)

        # Stream function (psi)
        x_in, y_in = inputs[:, 0:1], inputs[:, 1:2]
        bcv  = 16 * x_in * (1 - x_in) * y_in * (1 - y_in)
        ExpB = torch.exp(-(1 - y_in)**2 / dy**2)
        psilid = ((y_in-1)*y_in**2
                  * (1-torch.exp(-(x_in-1)**2/dx**2))
                  * (1-torch.exp(-x_in**2/dx**2)) * ExpB)
        psi = psilid + bcv**2 * outputs[:, 0:1]

    u_plot    = u.detach().cpu().numpy().reshape(xx.shape)
    v_plot    = v.detach().cpu().numpy().reshape(xx.shape)
    p_plot    = p.detach().cpu().numpy().reshape(xx.shape)
    psi_plot  = psi.detach().cpu().numpy().reshape(xx.shape)
    res1_plot = res1.detach().cpu().numpy().reshape(xx.shape)
    res2_plot = res2.detach().cpu().numpy().reshape(xx.shape)
    vMag_plot = np.sqrt(u_plot**2 + v_plot**2)

    def _save_contourf(data, title, filename, cmap='jet', n_levels=60):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.set_tight_layout(True)
        cf = ax.contourf(xx, yy, data, n_levels, cmap=cmap)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.axis("scaled")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'{title}  (Re = {Re:.0f})')
        plt.savefig(f"figures/{filename}", dpi=150)
        plt.close()

    # ── 2. Velocity components ───────────────────────────────────────────────
    _save_contourf(u_plot,    'Velocity  u',  f'Velocity_u{suffix}.png')
    _save_contourf(v_plot,    'Velocity  v',  f'Velocity_v{suffix}.png')

    # ── 3. Pressure ──────────────────────────────────────────────────────────
    # Mean-subtract so pressure is relative (removes arbitrary constant)
    p_centered = p_plot - p_plot.mean()
    _save_contourf(p_centered, 'Pressure',
                   f'Pressure_p{suffix}.png', cmap='RdBu_r')

    # ── 4. PDE residuals (log |residual|) ────────────────────────────────────
    eps_r = 1e-16
    _save_contourf(np.log10(np.abs(res1_plot) + eps_r),
                   r'log$_{10}|$Res. x-mom$|$',
                   f'Residual_Eq1{suffix}.png', cmap='hot_r')
    _save_contourf(np.log10(np.abs(res2_plot) + eps_r),
                   r'log$_{10}|$Res. y-mom$|$',
                   f'Residual_Eq2{suffix}.png', cmap='hot_r')

    # ── 5. Streamline plot (matplotlib streamplot ─────────
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_tight_layout(True)

    # Background: velocity magnitude
    #cf = ax.contourf(xx, yy, vMag_plot, 60, cmap='Blues')
    #fig.colorbar(cf, ax=ax, shrink=0.85, label='|U|')

    # streamplot expects arrays ordered by (y, x) when indexing='ij'
    # so we transpose u and v
    speed = vMag_plot.T                                         # shape (n_plot, n_plot)
    skip = max(1, n_plot // 20)
    lw    = 2.0 * speed / (speed.max() + 1e-10) + 0.5         # thicker where flow is faster

    strm = ax.streamplot(
        xVals, yVals,                   # 1-D coordinate vectors
        u_plot.T, v_plot.T,             # transpose because meshgrid uses indexing='ij'
        color=speed,
        cmap='hot',
        linewidth=lw,
        density=2.0,                    # increase for denser lines near corners
        arrowsize=1.2,
        arrowstyle='->',
    )
    fig.colorbar(strm.lines, ax=ax, shrink=0.85, label='|U|')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Streamlines  (Re = {Re:.0f})')
    plt.savefig(f"figures/Streamlines{suffix}.png", dpi=150)
    plt.close()


    # ── 6. Flow magnitude + quiver (colour = speed) ──────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_tight_layout(True)
    cf = ax.contourf(xx, yy, vMag_plot, 60, cmap='jet')
    fig.colorbar(cf, ax=ax, shrink=0.85, label='|U|')
    ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
              u_plot[::skip, ::skip], v_plot[::skip, ::skip],
              color='white', alpha=0.8)
    ax.axis("scaled")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Flow Magnitude (Re = {Re:.0f})')
    plt.savefig(f"figures/Flow_Magnitude_Quiver{suffix}.png", dpi=150)
    plt.close()

    # ── 7. Centreline velocity profiles (u along x=0.5, v along y=0.5) ──────
    mid = n_plot // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.set_tight_layout(True)
    axes[0].plot(u_plot[mid, :], yVals, 'b-', lw=2)
    axes[0].axvline(0, color='k', lw=0.8, ls='--')
    axes[0].set_xlabel('u'); axes[0].set_ylabel('y')
    axes[0].set_title(f'u-velocity  at  x = 0.5  (Re = {Re:.0f})')
    axes[0].grid(True, ls='--', alpha=0.4)

    axes[1].plot(xVals, v_plot[:, mid], 'r-', lw=2)
    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('v')
    axes[1].set_title(f'v-velocity  at  y = 0.5  (Re = {Re:.0f})')
    axes[1].grid(True, ls='--', alpha=0.4)

    plt.savefig(f"figures/Centreline_Profiles{suffix}.png", dpi=150)
    plt.close()

    print(f"Figures saved to figures/ (suffix='{suffix}')")

# -------------------------
# Training Flow
# -------------------------
def train_pinn(model, device, save_prefix="model_"):
    X_test = sample_test_points(TEST_PDE_N)
    loss_hist, lp_hist, lb_hist, steps_hist            = [], [], [], []
    test_loss_hist, test_lp_hist, test_lb_hist, test_steps = [], [], [], []

    # Save a copy of this script inside the model directory
    src = os.path.realpath(__file__)
    dst = os.path.join('model', os.path.basename(src))
    shutil.copy2(src, dst)

    global_step = 0

    # ── Stage A: SOAP + CUDA graph ───────────────────────────────────────────
    if SOAP_STEPS > 0:
        opt = pytorch_optimizer.optimizer.soap.SOAP(
            model.parameters(), lr=SOAP_LR,
            betas=(0.99, 0.999), weight_decay=0.0, precondition_frequency=1
        )
        #Xpde_static = torch.empty(
        #    (PDE_BATCH + round(FRAC_CORNER*PDE_BATCH)*2, 2),   # 2-D
        #    device=device, dtype=torch.float64, requires_grad=True
        #)
        Xpde_static = torch.empty(
            (PDE_BATCH, 2),   # 2-D
            device=device, dtype=torch.float64, requires_grad=True
        )
        lp_static   = torch.empty((), device=device, dtype=torch.float64)
        lb_static   = torch.empty((), device=device, dtype=torch.float64)
        loss_static = torch.empty((), device=device, dtype=torch.float64)

        Xpde_static.data.copy_(sample_pde_points(PDE_BATCH))

        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()

        with torch.cuda.stream(s):
            a = torch.randn((512, 512), device=device, dtype=torch.float64, requires_grad=True)
            (a @ a).sum().backward()

            for _ in range(SOAP_WARMUP_STEPS):
                Xpde_static.data.copy_(sample_pde_points(PDE_BATCH))
                opt.zero_grad(set_to_none=True)
                res1, res2 = fp_pde(model, Xpde_static)
                loss = torch.mean(res1**2) + torch.mean(res2**2)
                loss.backward(); opt.step()

            g.capture_begin()
            opt.zero_grad(set_to_none=True)
            res1, res2 = fp_pde(model, Xpde_static)
            loss1 = torch.mean(res1**2)
            loss2 = torch.mean(res2**2)
            loss  = loss1 + loss2
            lp_static.copy_(loss1); lb_static.copy_(loss2); loss_static.copy_(loss)
            loss.backward(); opt.step()
            g.capture_end()

        s.synchronize(); torch.cuda.synchronize()
        print("Captured CUDA graph. Starting SOAP replay training...")

        for step in range(SOAP_STEPS):
            if step % 50 == 0:
                Xpde_static.data.copy_(sample_pde_points(PDE_BATCH))
            g.replay()

            if step % RECORD_EVERY == 0:
                torch.cuda.synchronize()
                steps_hist.append(global_step)
                loss_hist.append(float(loss_static.detach().cpu()))
                lp_hist.append(float(lp_static.detach().cpu()))
                lb_hist.append(float(lb_static.detach().cpu()))

            if step % TEST_EVERY == 0:
                torch.cuda.synchronize()
                tl, tlp, tlb = compute_test_loss(model, X_test)
                test_steps.append(global_step)
                test_loss_hist.append(tl); test_lp_hist.append(tlp); test_lb_hist.append(tlb)

                generate_plots(model, device)
                torch.save(model.state_dict(), f"model/{save_prefix}soap_{step:05d}.pth")

            if step % RECORD_EVERY == 0:
                torch.cuda.synchronize()
                msg = (f"[SOAP] step={step:5d} loss={loss_hist[-1]:.6e} "
                       f"Eq1={lp_hist[-1]:.6e} Eq2={lb_hist[-1]:.6e}")
                if test_steps and test_steps[-1] == global_step:
                    msg += f" | test={test_loss_hist[-1]:.6e}"
                print(msg)
            global_step += 1

        torch.save(model.state_dict(), f"model/{save_prefix}soap_final.pth")

    # ── Stage B: SSBroyden2 ──────────────────────────────────────────────────
    if USE_SSBROYDEN2:
        try: del g; del s
        except NameError: pass
        torch.cuda.empty_cache()
        print("Starting SSBroyden2 refinement...")

        Xpde_qn_base = sample_pde_points(SSB_FIXED_PDE_N)

        with torch.enable_grad():
            r1, r2    = fp_pde(model, Xpde_qn_base)
            loss_init = torch.mean(r1**2) + torch.mean(r2**2)
            print(f"Loss BEFORE Step 0: {loss_init.item():.6e}")

        global_step = SOAP_STEPS
        opt_ssb = SSBroyden2(
            model.parameters(), lr=SSB_LR, gtol=1e-12,
            line_search=SSB_LINE_SEARCH, c1=SSB_C1, c2=SSB_C2,
            wolfe_maxiter=SSB_LS_MAXITER, zoom_maxiter=SSB_ZOOM_MAXITER,
            amax=SSB_AMAX, dtype=torch.float64, device=device
        )

        for k in range(SSB_STEPS):
            def closure():
                opt_ssb.zero_grad(set_to_none=True)
                r1, r2 = fp_pde(model, Xpde_qn_base)
                loss   = torch.mean(r1**2) + torch.mean(r2**2)
                loss.backward()
                return loss

            loss_val = opt_ssb.step(closure)

            with torch.enable_grad():
                r1, r2 = fp_pde(model, Xpde_qn_base)
                loss1  = torch.mean(r1**2)
                loss2  = torch.mean(r2**2)
                loss   = loss1 + loss2

            steps_hist.append(global_step)
            loss_hist.append(float(loss.detach().cpu()))
            lp_hist.append(float(loss1.detach().cpu()))
            lb_hist.append(float(loss2.detach().cpu()))
            global_step += 1

            if k % TEST_EVERY == 0:
                tl, tlp, tlb = compute_test_loss(model, X_test)
                test_steps.append(global_step)
                test_loss_hist.append(tl); test_lp_hist.append(tlp); test_lb_hist.append(tlb)
                generate_plots(model, device)
                torch.save(model.state_dict(), f"model/{save_prefix}ssbroyden_{k:05d}.pth")

            if k % RECORD_EVERY == 0:
                msg = (f"[SSBroyden2] iter={k:5d} loss={loss_hist[-1]:.6e} "
                       f"Eq1={lp_hist[-1]:.6e} Eq2={lb_hist[-1]:.6e}")
                if test_steps and test_steps[-1] == global_step:
                    msg += f" | test={test_loss_hist[-1]:.6e}"
                print(msg)

        torch.save(model.state_dict(), f"model/{save_prefix}ssbroyden_final.pth")

    loss_data = {
        'StepsTraining': steps_hist, 'TrainLoss1': lp_hist, 'TrainLoss2': lb_hist,
        'EpochsTest': test_steps,    'TestLoss1':  test_lp_hist, 'TestLoss2': test_lb_hist,
    }
    generate_plots(model, device, loss_data)
    return model

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    LoadModel       = False
    TrainModel      = True
    ModelToLoadPath = 'old_model/base_ssbroyden_final.pth'

    model = PINN().to(device)

    if LoadModel:
        if os.path.exists(ModelToLoadPath):
            print(f"Loading model state from {ModelToLoadPath}...")
            model.load_state_dict(torch.load(ModelToLoadPath, map_location=device))
        else:
            print(f"Warning: {ModelToLoadPath} not found. Starting from scratch.")

    if TrainModel:
        train_pinn(model, device, save_prefix="base_")
    else:
        print("Skipping training. Generating plots from loaded model...")
        generate_plots(model, device, suffix="_loaded")
