"""
Binning of saved particles → f(v, xi, X, Z) per step → TXT outputs.

Assumes files written like:
    <base_dir>/<run_id>/device_0/step_000025.h5
    <base_dir>/<run_id>/device_1/step_000025.h5
    ...

Each file contains datasets: "v", "xi", "X", "Z", "ids", "tauEsc", "XEsc", "ZEsc".
"""

from pathlib import Path
import re
import h5py
import numpy as np
import tqdm
import os
import glob
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({'font.size': 18})

# ====================
# CONFIG
# ====================
base_dir   = "./particles"   # same as generator
run_id     = "run_0001"
out_dir    = "./binned_txt"  # where to write txt outputs

# grid (nodes, inclusive)
vMin, vMax, nv = 0.01, 15.0, 32
xiMin, xiMax, nxi  = -1.0,  1.0, 32
XMin, XMax, nX  = -1.0,  1.0, 128
ZMin, ZMax, nZ  = -1.0,  1.0, 128

# Physics constants
EnergyeV = 2e4
Ts = 1e3
Te = Ts
q = 2
Ephi0 = 0.0
aMinor = 0.5  # minor radius in meters
R0 = 3*aMinor # major radius in meters
B0Tesla = 2 # magnetic field in Tesla
ni = 1e20 # ion density in 1/m^3
ne = ni
Zs = 1 # charge of energetic ion species
Zbulk = Zs # charge of bulk plamsa species
ms = 2 # mass of energetic particle in units of proton mass
mbulk = ms # mass of bulk ion species
CouLog = 15 # Coulomb logarithm

## physical constants ##
re = 2.8179e-15 # classical electron radius in meters
clight = 2.998e8 # speed of light
meOverms = 1/1836.15/ms # electron to energetic ion mass
mecSq = 511e3 # electron rest mass in eV

## derived quantities ##
epsilon = aMinor / R0
B0Gauss = 10**4 * B0Tesla
wc0a = 9.58e3 * Zs / ms * B0Gauss
rhostar = np.sqrt(2) * np.sqrt(meOverms) * np.sqrt(Ts/mecSq) * clight/wc0a/aMinor
nu_D = np.pi * (aMinor*re**2*ni) * Zs**2 * Zbulk**2 * (mecSq/Te)**2 * (ms/mbulk)**2 * CouLog

# time length and steps. Needs to be identical as used in particle pusher
tfinal = 100_000
NumOutputs = 11
tplot = int(tfinal/(NumOutputs-1)) # interval over which to plot solution
dt_ = 1.e-1 # time step
n_steps = int(tfinal/dt_)

output_steps = []
output_steps.append(0)
plot_every = (tplot/dt_)
for step in range(1, n_steps + 1):
    if step % plot_every == 0:
        output_steps.append(step)

output_steps_np = np.asarray(output_steps, dtype=np.int64)
TimeArray = output_steps_np * dt_

nrbins = 40 # number of radial bins for performing statistics on particles

xi_val = 0.0

# ---------- grid & jacobian ----------
def make_grids():
    dv  = (vMax - vMin) / (nv - 1)
    dxi = (xiMax - xiMin) / (nxi - 1)
    dX = (XMax - XMin) / (nX - 1)
    dZ = (ZMax - ZMin) / (nZ - 1)
    v_nodes  = vMin + np.arange(nv,  dtype=np.float32) * dv
    xi_nodes = xiMin + np.arange(nxi, dtype=np.float32) * dxi
    X_nodes = XMin + np.arange(nX, dtype=np.float32) * dX
    Z_nodes = ZMin + np.arange(nZ, dtype=np.float32) * dZ
    return v_nodes, xi_nodes, X_nodes, Z_nodes, dv, dxi, dX, dZ


def jacobian_matrix_vxi(v_nodes, dv, dxi):
    """J[ii,jj] with edge factors 1/2 (edges) and 1/4 (corners)."""
    base = (2.0 * np.pi * v_nodes**2 * dv * dxi).astype(np.float32)  # (nv,)
    edge_v  = np.ones((nv,),  np.float32); edge_v[[0, nv-1]] = 0.5
    edge_xi = np.ones((nxi,), np.float32); edge_xi[[0, nxi-1]] = 0.5
    J = (base[:, None] * edge_v[:, None] * edge_xi[None, :]).astype(np.float32)  # (ng,nxi)
    return J


def jacobian_matrix_XZ(X_nodes, dX, dZ):
    """J[ii,jj] with edge factors 1/2 (edges) and 1/4 (corners)."""
    R_nodes = 1/epsilon + X_nodes
    base = (R_nodes * dX * dZ).astype(np.float32)  # (nX,)
    edge_X  = np.ones((nX,),  np.float32); edge_X[[0, nX-1]] = 0.5
    edge_Z = np.ones((nZ,), np.float32); edge_Z[[0, nZ-1]] = 0.5
    J = (base[:, None] * edge_X[:, None] * edge_Z[None, :]).astype(np.float32)  # (nX,nZ)
    return J


# ---------- Bin the particle distribution in momentum space  ----------
def BinMomentum(v, xi, X, Z, v_nodes, xi_nodes, dv, dxi, J, weights=None):
    fDist = np.zeros((nv, nxi), dtype=np.float32)

    # inside open bounds & finite
    m = (
        np.isfinite(v) & np.isfinite(xi) &
        np.isfinite(X) & np.isfinite(Z) &
        (v > vMin) & (v < vMax) &
        (xi > xiMin) & (xi < xiMax) &
        (X > XMin) & (X < XMax) &
        (Z > ZMin) & (Z < ZMax)
    )

    # return early if no particles in domain
    if not np.any(m):
        return fDist

    v  = v[m].astype(np.float32)
    xi = xi[m].astype(np.float32)
    w  = (np.ones_like(v, np.float32) if weights is None else np.asarray(weights, np.float32)[m])

    # cell coords
    vcoord = (v - vMin) / dv
    xicoord = (xi - xiMin) / dxi
    i = np.floor(vcoord).astype(np.int64)
    j = np.floor(xicoord).astype(np.int64)
    i = np.clip(i, 0, nv-2); j = np.clip(j, 0, nxi-2)
    ip1 = i + 1; jp1 = j + 1

    vi   = v_nodes[i];    vip1 = v_nodes[ip1]
    xj   = xi_nodes[j];   xjp1 = xi_nodes[jp1]

    wx_i   = 1.0 - np.abs(v - vi)   / dv
    wx_ip1 = 1.0 - np.abs(v - vip1) / dv
    wy_j   = 1.0 - np.abs(xi - xj)   / dxi
    wy_jp1 = 1.0 - np.abs(xi - xjp1) / dxi

    w_ij     = (wx_i   * wy_j   ) * w
    w_ijp1   = (wx_i   * wy_jp1 ) * w
    w_ip1j   = (wx_ip1 * wy_j   ) * w
    w_ip1jp1 = (wx_ip1 * wy_jp1 ) * w

    J_ij     = J[i,   j  ]
    J_ijp1   = J[i,   jp1]
    J_ip1j   = J[ip1, j  ]
    J_ip1jp1 = J[ip1, jp1]

    # scatter-add (CIC / Jacobian)
    np.add.at(fDist, (i,   j  ), w_ij     / J_ij)
    np.add.at(fDist, (i,   jp1), w_ijp1   / J_ijp1)
    np.add.at(fDist, (ip1, j  ), w_ip1j   / J_ip1j)
    np.add.at(fDist, (ip1, jp1), w_ip1jp1 / J_ip1jp1)

    return fDist  # (nv, nxi)


# ---------- Bin the particle distribution in momentum space  ----------
def BinSpatialDistXZ(v, xi, X, Z, X_nodes, Z_nodes, dX, dZ, J, weights=None):
    fDist = np.zeros((nX, nZ), dtype=np.float32)

    # inside open bounds & finite
    m = (
        np.isfinite(v) & np.isfinite(xi) &
        np.isfinite(X) & np.isfinite(Z) &
        (v > vMin) & (v < vMax) &
        absorb_mask(v, xi, X, Z) == False
        #&
        #(xi > xiMin) & (xi < xiMax) &
        #(X > XMin) & (X < XMax) &
        #(Z > ZMin) & (Z < ZMax)
    )

    # return early if no particles in domain
    if not np.any(m):
        return fDist

    X  = X[m].astype(np.float32)
    Z  = Z[m].astype(np.float32)
    w  = (np.ones_like(X, np.float32) if weights is None else np.asarray(weights, np.float32)[m])

    # cell coords
    Xcoord = (X - XMin) / dX
    Zcoord = (Z - ZMin) / dZ
    i = np.floor(Xcoord).astype(np.int64)
    j = np.floor(Zcoord).astype(np.int64)
    i = np.clip(i, 0, nX-2); j = np.clip(j, 0, nZ-2)
    ip1 = i + 1; jp1 = j + 1

    Xi   = X_nodes[i];    Xip1 = X_nodes[ip1]
    Zj   = Z_nodes[j];    Zjp1 = Z_nodes[jp1]

    wx_i   = 1.0 - np.abs(X - Xi)   / dX
    wx_ip1 = 1.0 - np.abs(X - Xip1) / dX
    wy_j   = 1.0 - np.abs(Z - Zj)   / dZ
    wy_jp1 = 1.0 - np.abs(Z - Zjp1) / dZ

    w_ij     = (wx_i   * wy_j   ) * w
    w_ijp1   = (wx_i   * wy_jp1 ) * w
    w_ip1j   = (wx_ip1 * wy_j   ) * w
    w_ip1jp1 = (wx_ip1 * wy_jp1 ) * w

    J_ij     = J[i,   j  ]
    J_ijp1   = J[i,   jp1]
    J_ip1j   = J[ip1, j  ]
    J_ip1jp1 = J[ip1, jp1]

    # scatter-add (CIC / Jacobian)
    np.add.at(fDist, (i,   j  ), w_ij     / J_ij)
    np.add.at(fDist, (i,   jp1), w_ijp1   / J_ijp1)
    np.add.at(fDist, (ip1, j  ), w_ip1j   / J_ip1j)
    np.add.at(fDist, (ip1, jp1), w_ip1jp1 / J_ip1jp1)

    return fDist  # (nX, nZ)


def absorb_mask(v, xi, X, Z):
    """Boolean mask of survivors (p >= p_th)."""
    rSQ = X**2 + Z**2
    
    return rSQ > 1.0


# ---------- Particle Statistics --------
def radial_msd_binned(
    X_init: np.ndarray,
    Z_init: np.ndarray,
    X_now: np.ndarray,
    Z_now: np.ndarray,
    r_edges: np.ndarray,
    *,
    min_count: int = 20,
):
    """
    Compute binned mean-squared radial displacement relative to each particle's *initial* minor radius.

    For each particle i:
      r0_i = r_init_i
      r_i  = r_now_i
      Δr_i = r_i - r0_i
      contribution = (Δr_i)^2

    Then bin by r0_i using r_edges and compute per-bin:
      count, mean(Δr^2), stderr_of_mean(Δr^2)

    Parameters
    ----------
    X_init, Z_init, X_now, Z_now : (N,) arrays
        Initial and current coordinates for N particles.
    r_edges : (Nbins+1,) array
        Bin edges in minor radius r (if use_r=True) or in r^2 (if use_r=False).
    min_count : int
        Bins with fewer than min_count particles will get NaN for mean/stderr.

    Returns
    -------
    out : dict
        Keys: 'bin_edges', 'bin_centers', 'count', 'msd', 'msd_stderr',
        where msd is <Δr^2> in each initial-radius bin.
    """

    X_init = np.asarray(X_init)
    Z_init = np.asarray(Z_init)
    X_now = np.asarray(X_now)
    Z_now = np.asarray(Z_now)
    r_edges = np.asarray(r_edges)

    if not (X_init.shape == Z_init.shape == X_now.shape == Z_now.shape):
        raise ValueError("X_init, Z_init, X_now, Z_now must all have the same shape (N,)")

    r0_sq = X_init * X_init + Z_init * Z_init
    r_sq = X_now * X_now + Z_now * Z_now


    r0 = np.sqrt(r0_sq)
    r = np.sqrt(r_sq)
    bin_var = r0
    bin_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    dr = r - r0
    dr2 = dr * dr

    # Bin by initial radius
    bin_index = np.digitize(bin_var, r_edges) - 1  # Identify bin index for each particle. bins: 0...Nbins-1
    nbins = len(r_edges) - 1
    valid = (bin_index >= 0) & (bin_index < nbins) & np.isfinite(dr2) # check that particle is in domain

    count = np.zeros(nbins, dtype=int) # sum used to count particles in each bin
    sum1 = np.zeros(nbins, dtype=float) # sum used for mean
    sum2 = np.zeros(nbins, dtype=float) # sum used for stderr

    # Accumulate with bincount for speed
    bi = bin_index[valid] # only include particle in domain
    v = dr2[valid]
    count += np.bincount(bi, minlength=nbins).astype(int)
    sum1 += np.bincount(bi, weights=v, minlength=nbins)
    sum2 += np.bincount(bi, weights=v * v, minlength=nbins)

    msd = np.full(nbins, np.nan, dtype=float)
    msd_stderr = np.full(nbins, np.nan, dtype=float)

    good = count >= max(1, int(min_count))
    msd[good] = sum1[good] / count[good]

    # Standard error of the mean from per-bin sample variance:
    # var = (E[v^2] - E[v]^2) * n/(n-1), stderr = sqrt(var/n)
    n = count[good].astype(float)
    ev = msd[good]
    ev2 = sum2[good] / n
    var_unbiased = (ev2 - ev * ev) * (n / (n - 1.0))
    var_unbiased = np.maximum(var_unbiased, 0.0)
    msd_stderr[good] = np.sqrt(var_unbiased / n)

    return dict(
        bin_edges=r_edges,
        bin_centers=bin_centers,
        count=count,
        msd=msd,
        msd_stderr=msd_stderr,
    )


def psi_tor_msd_binned(
    X_init: np.ndarray,
    Z_init: np.ndarray,
    X_now: np.ndarray,
    Z_now: np.ndarray,
    psit_edges: np.ndarray,
    *,
    min_count: int = 20,
):

    X_init = np.asarray(X_init)
    Z_init = np.asarray(Z_init)
    X_now = np.asarray(X_now)
    Z_now = np.asarray(Z_now)
    psit_edges = np.asarray(psit_edges)

    if not (X_init.shape == Z_init.shape == X_now.shape == Z_now.shape):
        raise ValueError("X_init, Z_init, X_now, Z_now must all have the same shape (N,)")

    r0_sq = X_init * X_init + Z_init * Z_init
    r_sq = X_now * X_now + Z_now * Z_now
    psit0 = 0.5 * r0_sq
    psit = 0.5 * r_sq

    #r0 = np.sqrt(r0_sq)
    #r = np.sqrt(r_sq)
    bin_var = psit
    bin_centers = 0.5 * (psit_edges[:-1] + psit_edges[1:])

    dpsit = psit - psit0
    dpsit2 = dpsit * dpsit

    # Bin by initial radius
    bin_index = np.digitize(bin_var, psit_edges) - 1  # Identify bin index for each particle. bins: 0...Nbins-1
    nbins = len(psit_edges) - 1
    valid = (bin_index >= 0) & (bin_index < nbins) & np.isfinite(dpsit2) # check that particle is in domain

    count = np.zeros(nbins, dtype=int) # sum used to count particles in each bin
    sum1 = np.zeros(nbins, dtype=float) # sum used for mean
    sum2 = np.zeros(nbins, dtype=float) # sum used for stderr

    # Accumulate with bincount for speed
    bi = bin_index[valid] # only include particle in domain
    v = dpsit2[valid]
    count += np.bincount(bi, minlength=nbins).astype(int)
    sum1 += np.bincount(bi, weights=v, minlength=nbins)
    sum2 += np.bincount(bi, weights=v * v, minlength=nbins)

    msd = np.full(nbins, np.nan, dtype=float)
    msd_stderr = np.full(nbins, np.nan, dtype=float)

    good = count >= max(1, int(min_count))
    msd[good] = sum1[good] / count[good]

    # Standard error of the mean from per-bin sample variance:
    # var = (E[v^2] - E[v]^2) * n/(n-1), stderr = sqrt(var/n)
    n = count[good].astype(float)
    ev = msd[good]
    ev2 = sum2[good] / n
    var_unbiased = (ev2 - ev * ev) * (n / (n - 1.0))
    var_unbiased = np.maximum(var_unbiased, 0.0)
    msd_stderr[good] = np.sqrt(var_unbiased / n)

    return dict(
        bin_edges=psit_edges,
        bin_centers=bin_centers,
        count=count,
        msd=msd,
        msd_stderr=msd_stderr,
    )


def fit_diffusivity_from_msd(
        t: np.ndarray,
        msd_t: np.ndarray,
        *,
        msd_stderr_t: np.ndarray | None = None,
        dim: int = 1,
        t_min: float | None = None,
        t_max: float | None = None,
):
    """
    Fit a line to MSD(t) to infer diffusivity D from <Δr^2> ≈ 2*dim*D*t + C.

    Uses weighted least squares if msd_stderr_t is provided; otherwise unweighted.
    Returns D and its 1-sigma uncertainty propagated from the slope uncertainty.

    Parameters
    ----------
    t : (M,) array
        Times corresponding to MSD samples.
    msd_t : (M,) array
        MSD values (e.g., one radial bin's <Δr^2> vs time).
    msd_stderr_t : (M,) array or None
        1-sigma uncertainty in msd_t. If None, fit is unweighted.
    dim : int
        Effective diffusion dimension. For 1D radial diffusion, use dim=1:
          <Δr^2> = 2 D t + C
        For 2D, <Δr^2> = 4 D t + C, etc.
    t_min, t_max : float or None
        Optional time window for the fit.

    Returns
    -------
    out : dict
        Keys: 'slope', 'slope_stderr', 'intercept', 'intercept_stderr',
              'D', 'D_stderr', 'r2', 'mask'
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(msd_t, dtype=float)

    if msd_stderr_t is None:
        w = None
    else:
        s = np.asarray(msd_stderr_t, dtype=float)
        w = np.where(np.isfinite(s) & (s > 0), 1.0 / (s * s), 0.0)

    mask = np.isfinite(t) & np.isfinite(y)
    if t_min is not None:
        mask &= t >= t_min
    if t_max is not None:
        mask &= t <= t_max
    if w is not None:
        mask &= w > 0

    tt = t[mask]
    yy = y[mask]
    if w is None:
        ww = None
    else:
        ww = w[mask]

    if tt.size < 2:
        raise ValueError("Not enough valid points to fit.")

    # Weighted linear regression with intercept using normal equations
    if ww is None:
        ww = np.ones_like(tt)

    Sw = np.sum(ww)
    Sx = np.sum(ww * tt)
    Sy = np.sum(ww * yy)
    Sxx = np.sum(ww * tt * tt)
    Sxy = np.sum(ww * tt * yy)

    Delta = Sw * Sxx - Sx * Sx
    if Delta <= 0:
        raise ValueError("Degenerate fit (check times/weights).")

    m = (Sw * Sxy - Sx * Sy) / Delta
    b = (Sxx * Sy - Sx * Sxy) / Delta

    resid = yy - (m * tt + b)
    dof = max(tt.size - 2, 1)
    s2 = np.sum(ww * resid * resid) / dof  # weighted residual variance estimate

    var_m = s2 * (Sw / Delta)
    var_b = s2 * (Sxx / Delta)

    slope = m
    slope_stderr = np.sqrt(max(var_m, 0.0))
    intercept = b
    intercept_stderr = np.sqrt(max(var_b, 0.0))

    # Diffusivity: slope = 2*dim*D
    D = slope / (2.0 * float(dim))
    D_stderr = slope_stderr / (2.0 * float(dim))

    # Weighted R^2 (using weights if provided)
    if ww is None:
        ybar = np.mean(yy)
        ss_res = np.sum((yy - (slope * tt + intercept)) ** 2)
        ss_tot = np.sum((yy - ybar) ** 2)
    else:
        ybar = np.sum(ww * yy) / np.sum(ww)
        ss_res = np.sum(ww * (yy - (slope * tt + intercept)) ** 2)
        ss_tot = np.sum(ww * (yy - ybar) ** 2)
    r2 = np.nan if ss_tot == 0 else (1.0 - ss_res / ss_tot)

    return dict(
        slope=slope,
        slope_stderr=slope_stderr,
        intercept=intercept,
        intercept_stderr=intercept_stderr,
        D=D,
        D_stderr=D_stderr,
        r2=r2,
        mask=mask,
    )


# ---------- utilities ---------
#_step_pat = re.compile(r"step_(\d{6})\.h5$")

def discover_devices(run_dir: Path):
    dev_dirs = sorted([p for p in run_dir.glob("device_*") if p.is_dir()])
    if not dev_dirs:
        raise RuntimeError(f"No device_* folders found under {run_dir}")
    return dev_dirs

def list_steps_for_device(dev_dir: Path):
    steps = []
    for f in dev_dir.glob("step_*.h5"):
        m = _step_pat.search(f.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)

def intersect_steps(dev_dirs):
    sets = []
    for d in dev_dirs:
        steps = list_steps_for_device(d)
        if not steps:
            raise RuntimeError(f"No step_*.h5 in {d}")
        sets.append(set(steps))
    common = sorted(set.intersection(*sets))
    return common


def plot_distribution_vxi(output_file, v_nodes, xi_nodes, z_values):
    # Create a figure and an axes object
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create a meshgrid for the contour plot axes
    X, Y = np.meshgrid(v_nodes, xi_nodes)

    # Reshape the 1D distribution data into a 2D array for the Z values
    # The shape must match the meshgrid: (number of xi_nodes, number of g_nodes)
    Z = z_values.reshape(len(xi_nodes), len(v_nodes)).T

    # Create the filled contour plot
    # A logarithmic color scale is often useful for distributions
    contour = ax.contourf(X, Y, Z, levels=50, cmap='jet')

    # Add a color bar to the plot to show the scale of the distribution values
    #fig.colorbar(contour, ax=ax, label='Distribution Value')
    fig.colorbar(contour, ax=ax)

    # Set plot titles and labels for clarity
    #ax.set_title('Particle Distribution')
    ax.set_xlabel('$v/v_{Ti}$')
    ax.set_ylabel(r'$\xi$')

    # Use a logarithmic scale for the y-axis, which is common for
    # distribution functions to show a wide dynamic range.
    # ax.set_yscale('log')

    # Add a grid for better readability
    #ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Add a legend
    #ax.legend()

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

    # Display the plot
    plt.show()


def plot_distribution_XZ(output_file, X_nodes, Z_nodes, c_values, PlotLogScale = True):
    # Create a figure and an axes object
    #fig, ax = plt.subplots(figsize=(10, 7))
    fig, ax = plt.subplots()

    # Create a meshgrid for the contour plot axes
    X, Z = np.meshgrid(X_nodes, Z_nodes)

    PlasmaBoundary = X**2 + Z**2 - 1
    # Reshape the 1D distribution data into a 2D array for the C values
    # The shape must match the meshgrid: (number of xi_nodes, number of g_nodes)
    C = c_values.reshape(len(Z_nodes), len(X_nodes)).T

    #print(f"Maximum value = {np.nanmax(C)}")
    
    vmin = 0.0
    vmax = np.log10(1.0 + 5.6e5)
    levels = np.linspace(vmin, vmax, 100)
    norm = Normalize(vmin=vmin, vmax=vmax)

    rsq = X**2 + Z**2
    r = np.sqrt(rsq)
    BMax = 1 / ( 1 - r*epsilon )
    Bprime = 1 / (1 + epsilon*X)
    TrapRegion = BMax/Bprime * ( 1 - xi_val**2 ) - 1
    
    # Create the filled contour plot
    # A logarithmic color scale is often useful for distributions
    if PlotLogScale == True:
        #contour = ax.contourf(X, Z, np.log10(1+C), levels=50, cmap='jet')
        contour = ax.contourf(X, Z, np.log10(1+C), levels=levels, cmap='jet', norm=norm)
    else:
        contour = ax.contourf(X, Z, C, levels=50, cmap='jet')
        
    #contourtmp = ax.contour(X, Z, PlasmaBoundary, levels=[0], colors='white', linestyles='-', linewidths=2)
    cs1tmp = ax.contour(X, Z, TrapRegion, levels=[0], colors='white', linestyles='-', linewidths=2)
    
    # Add a color bar to the plot to show the scale of the distribution values
    #fig.colorbar(contour, ax=ax, label='Distribution Value')
    fig.colorbar(contour, ax=ax)

    # Set plot titles and labels for clarity
    #ax.set_title('Particle Distribution')
    ax.set_xlabel('$(R-R_0)/a$')
    ax.set_ylabel(r'$Z/a$')

    # Use a logarithmic scale for the y-axis, which is common for
    # distribution functions to show a wide dynamic range.
    # ax.set_yscale('log')

    # Add a grid for better readability
    #ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Add a legend
    #ax.legend()

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

    # Display the plot
    plt.show()
    

def plot1D(output_file, values, time, ylabel, PlotLogScale = True):
    #fig, ax = plt.subplots(figsize=(10, 7))
    fig, ax = plt.subplots()
    
    ax.plot(time, values, label='r"\Delta p_\varphi / p_\varphi"', linestyle='-',color='blue',linewidth=2)

    #ax.set_title('Particle Distribution')
    ax.set_xlabel(r"$tv_{Ti}/a$")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # force scientific notation on x-axis
    if PlotLogScale == True:
        ax.set_yscale("log")
    #ax.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    plt.show()


def plot1D_multiple_curves(output_file, plot_info, time, xlabel, ylabel, PlotLogScale = True):
    fig, ax = plt.subplots()

    for label, (values, color, style) in plot_info.items():
        # We pass 'style' as the third positional argument (fmt)
        # This allows it to be a line ('-') or a marker ('o')
        ax.plot(time, values, style, color=color, label=label, linewidth=2)
        
    #for label, values in values_dict.items():
    #    ax.plot(time, values, label=label, linestyle='-', linewidth=2)   
    #ax.plot(time, values, label='r"\Delta p_\varphi / p_\varphi"', linestyle='-',color='blue',linewidth=2)

    #ax.set_title('Particle Distribution')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # force scientific notation on x-axis
    if PlotLogScale == True:
        ax.set_yscale("log")
    #ax.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    plt.show()
    

# ---------- main ----------
def main():
    run_dir = Path(base_dir) / run_id
    print(run_dir)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # grids & Jacobian
    v_nodes, xi_nodes, X_nodes, Z_nodes, dv, dxi, dX, dZ = make_grids()
    Jvxi = jacobian_matrix_vxi(v_nodes, dv, dxi)
    JXZ = jacobian_matrix_XZ(X_nodes, dX, dZ)

    # discover devices and common steps
    dev_dirs = discover_devices(run_dir)

    # write grids once
    np.savetxt(outp / "v_nodes.txt",  v_nodes,  fmt="%.8e")
    np.savetxt(outp / "xi_nodes.txt", xi_nodes, fmt="%.8e")
    np.savetxt(outp / "X_nodes.txt", X_nodes, fmt="%.8e")
    np.savetxt(outp / "Z_nodes.txt", Z_nodes, fmt="%.8e")

    # loop over steps
    Pphis      = []
    mus        = []
    r_edges = np.linspace(0,1,nrbins+1)
    psit_edges = np.linspace(0,0.5,nrbins+1)
    msd_vs_time  = np.zeros([len(output_steps_np),nrbins])
    msd_err_vs_time  = np.zeros([len(output_steps_np),nrbins])
    counter = 0
    for step in tqdm.tqdm(output_steps, desc="Binning"):
        # gather particles from all devices for this step
        vs         = []
        xis        = []
        Xs         = []
        Zs         = []
        idsList    = []
        tauEscs    = []
        XEscs      = []
        ZEscs      = []
        print(f"step={step}")
        for ddir in dev_dirs:
            fpath = ddir / f"step_{step:06d}.h5"
            with h5py.File(fpath, "r") as f:
                v      = f["v"][:]
                xi     = f["xi"][:]
                X      = f["X"][:]
                Z      = f["Z"][:]
                ids    = f["ids"][:]
                tauEsc = f["tauEsc"][:]
                XEsc   = f["XEsc"][:]
                ZEsc   = f["ZEsc"][:]
            # keep only alive particles
            # m_alive = (ids >= 0)
            # if m_alive.any():
                vs.append(v)
                xis.append(xi)
                Xs.append(X)
                Zs.append(Z)
                idsList.append(ids)
                tauEscs.append(tauEsc)
                XEscs.append(XEsc)
                ZEscs.append(ZEsc)
        if not vs:
            # no alive particles anywhere at this step
            fDistvxi = np.zeros((nv, nxi), dtype=np.float32)
            fDistXZ = np.zeros((nX, nZ), dtype=np.float32)
        else:
            v_all         = np.concatenate(vs).astype(np.float32, copy=False)
            xi_all        = np.concatenate(xis).astype(np.float32, copy=False)
            X_all         = np.concatenate(Xs).astype(np.float32, copy=False)
            Z_all         = np.concatenate(Zs).astype(np.float32, copy=False)
            tauEsc_all    = np.concatenate(tauEscs).astype(np.float32, copy=False)
            XEsc_all      = np.concatenate(XEscs).astype(np.float32, copy=False)
            ZEsc_all      = np.concatenate(ZEscs).astype(np.float32, copy=False)
            fDistvxi      = BinMomentum(v_all, xi_all, X_all, Z_all, v_nodes, xi_nodes, dv, dxi, Jvxi)
            fDistXZ       = BinSpatialDistXZ(v_all, xi_all, X_all, Z_all, X_nodes, Z_nodes, dX, dZ, JXZ)

            if step == 0:
                fInitvxi = fDistvxi
                vInit = v_all
                xiInit = xi_all
                XInit = X_all
                ZInit = Z_all
                rSqInit = np.sqrt(XInit**2+ZInit**2)
                maskinDomain = rSqInit < 1.0
                #maskinDomain = xiInit**2 > 0.9
                
            # Check toroidal canonical momentum conservation
            #mask = rSqInit < 1
            R = 1 + epsilon*X_all
            rSq = X_all**2 + Z_all**2
            Bphi = 1 / R
            B = Bphi
            #B = Bphi * np.sqrt(1+rSq*epsilon**2/q**2)
            #bphi = Bphi / B
            pPhi = np.sum(R[maskinDomain]*xi_all[maskinDomain]*v_all[maskinDomain] - 0.5/rhostar*epsilon*rSq[maskinDomain]/q)
            mu = 0.5*sum( v_all[maskinDomain]**2*(1.-xi_all[maskinDomain]**2)/B[maskinDomain] )
            print(f"p_phi = {pPhi}, step = {step}")
            print(f"mu = {mu}, step = {step}")
            Pphis.append(pPhi)
            mus.append(mu)
            
            #SaveSteps.append(step)

            #############################
            # Perform particle statistics
            #############################

            # Compute mean-squared-deviation
            out = radial_msd_binned(XInit, ZInit, X_all, Z_all, r_edges)
            #out = psi_tor_msd_binned(XInit, ZInit, X_all, Z_all, psit_edges)
            # Per-bin results
            r_centers = out["bin_centers"]   # (Nbins,)
            msd = out["msd"]                 # <Δr^2>(bin)
            msd_err = out["msd_stderr"]      # standard error of the mean
            counts = out["count"]            # particles per bin

            msd_vs_time[counter, :] = msd[:]
            msd_err_vs_time[counter, :] = msd_err[:]

            #####################################
            # Compute escape time and probability
            #####################################
            # Determine which particles have escaped
            masktauEsc = tauEsc_all > 0
            print(f"Number of absorbed electrons = {int(masktauEsc.sum())}")
            
            # Identify initial position of escapted particles
            vAbsorbed = vInit[masktauEsc]
            xiAbsorbed = xiInit[masktauEsc]
            XAbsorbed = XInit[masktauEsc]
            ZAbsorbed = ZInit[masktauEsc]
            tauAbsorbed = tauEsc_all[masktauEsc]
            XEscAbsorbed = XEsc_all[masktauEsc]
            ZEscAbsorbed = ZEsc_all[masktauEsc]

            """
            Zcrit = -0.1
            Xcrit = 0.9
            # From the particles that have escaped
            # identify which are lost to a given region
            # of the wall
            Captured = ((ZEscAbsorbed < Zcrit) & (XEscAbsorbed > Xcrit)).astype(np.float32)
            """
            # Bin electrons who escaped
            fEscInitvxi = BinMomentum(vAbsorbed, xiAbsorbed, XAbsorbed, ZAbsorbed, v_nodes, xi_nodes, dv, dxi, Jvxi)
            tauEscDistvxi = BinMomentum(vAbsorbed, xiAbsorbed, XAbsorbed, ZAbsorbed, v_nodes, xi_nodes, dv, dxi, Jvxi, weights=tauAbsorbed)
            tauEscAvgvxi = tauEscDistvxi / fEscInitvxi
            plot_distribution_vxi(f"figures/tauEscvxi_step_{step:06d}.png", v_nodes, xi_nodes, tauEscAvgvxi)

            fEscInitXZ = BinSpatialDistXZ(vAbsorbed, xiAbsorbed, XAbsorbed, ZAbsorbed, X_nodes, Z_nodes, dX, dZ, JXZ)
            tauEscDistXZ = BinSpatialDistXZ(vAbsorbed, xiAbsorbed, XAbsorbed, ZAbsorbed, X_nodes, Z_nodes, dX, dZ, JXZ, weights=tauAbsorbed)
            tauEscAvgXZ = tauEscDistXZ / fEscInitXZ
            print("Computed Escape time in XZ plane")
            plot_distribution_XZ(f"figures/tauEscXZ_step_{step:06d}.png", X_nodes, Z_nodes, tauEscAvgXZ)

            """
            CaptureDistXZ = BinSpatialDistXZ(vAbsorbed, xiAbsorbed, XAbsorbed, ZAbsorbed, X_nodes, Z_nodes, dX, dZ, JXZ, weights=Captured)
            CaptureProbXZ = CaptureDistXZ / fEscInitXZ
            plot_distribution_XZ(f"figures/CapProbXZ_step_{step:06d}.png", X_nodes, Z_nodes, CaptureProbXZ, PlotLogScale = False)
            """
            ############################
            # Plot particle distribution
            ############################
            # plot results and save figure
            plot_distribution_vxi(f"figures/fDistvxi_step_{step:06d}.png", v_nodes, xi_nodes, fDistvxi)
            plot_distribution_XZ(f"figures/fDistXZ_step_{step:06d}.png", X_nodes, Z_nodes, fDistXZ, PlotLogScale = False)
        
            # save per-step distribution as TXT (shape: ng rows × nxi cols)
            np.savetxt(outp / f"fDistvxi_step_{step:06d}.txt", fDistvxi, fmt="%.8e")

            # save per-step distribution as TXT (shape: nX rows × nZ cols)
            np.savetxt(outp / f"fDistXZ_step_{step:06d}.txt", fDistXZ, fmt="%.8e")

            ############################
            # Output data to text file
            ############################
            Xg, Zg = np.meshgrid(X_nodes, Z_nodes, indexing="ij")  # Xg,Zg shape: (Nx, Nz)

            # Ensure tau has shape (Nx, Nz)
            tau = np.asarray(tauEscAvgXZ)
            if tau.shape == (nX, nZ):
                pass
            elif tau.shape == (nZ, nX):
                tau = tau.T
            else:
                raise ValueError(f"tauEscAvgXZ shape {tau.shape} is not compatible with (Nx,Nz)=({Nx},{Nz})")

            xi_grid = np.full((nX, nZ), xi_val, dtype=tau.dtype)

            # Flatten to columns and save
            X_col  = Xg.ravel()
            Z_col  = Zg.ravel()
            xi_col = xi_grid.ravel()
            tau_col = tau.ravel()

            out = np.column_stack((X_col, Z_col, xi_col, tau_col))  # shape: (Nx*Nz, 4)
            mask = ~np.isnan(out).any(axis=1)           # keep rows with no NaNs
            out_clean = out[mask]
            
            # Save as a whitespace-delimited text file
            np.savetxt(f"./data/tauEsc_table.txt", out_clean, fmt="%.8e", header="X Z xi tauEscAvgXZ")
            counter += 1
            

    # Set time step
    print(f"[ok] wrote binned distributions to {outp}")
    PphiArray = np.array(Pphis)
    MuArray = np.array(mus)
    #StepsArray = np.array(SaveSteps)
    #TimeArray = dt * StepsArray
    out_Pphi = np.column_stack((TimeArray, PphiArray))  # shape: (num steps, 2)
    out_mu = np.column_stack((TimeArray, MuArray))
    mask_Pphi = ~np.isnan(out_Pphi).any(axis=1)           # keep rows with no NaNs
    mask_mu = ~np.isnan(out_mu).any(axis=1)
    out_clean_Pphi = out_Pphi[mask_Pphi]
    out_clean_mu = out_Pphi[mask_mu]
    np.savetxt(f"./data/Pphi_table.txt", out_clean_Pphi, fmt="%.12e", header="Time Pphi")
    np.savetxt(f"./data/mu_table.txt", out_clean_mu, fmt="%.12e", header="Time mu")
    # Plot time evolution of toroidal canonical momentum

    PphiInit = PphiArray[0]
    FracChangePphi = abs(PphiInit-PphiArray) / abs(PphiInit)
    plot1D(f"figures/Pphi.png", FracChangePphi[1:], TimeArray[1:], r'$|\Delta p_\varphi / p_\varphi|$')
    
    MuInit = MuArray[0]
    FracChangeMu = abs(MuInit-MuArray) / abs(MuInit)
    plot1D(f"figures/mu.png", FracChangeMu[1:], TimeArray[1:], r'$|\Delta \mu / \mu|$')

    loc = 20
    print(f"rcenter = {r_centers[loc]}")
    Dbin = np.zeros(nrbins)
    slopeBin = np.zeros(nrbins)
    interceptBin = np.zeros(nrbins)
    DneoBin = np.zeros(nrbins)
    for i in range(0,nrbins-1):
        r = r_centers[i]
        ep = r * epsilon
        ft = np.sqrt(2*ep)
        v = np.sqrt(EnergyeV/Ts)
        nu_D_v = nu_D / v**3
        Dneo = (0.689/2) * ft * q**2/epsilon**2/r**2 * (EnergyeV/Ts) * rhostar**2 * nu_D_v
        DneoBin[i] = Dneo
        
        FlagNaN = np.isnan(msd_vs_time[:, i]).any()
        if FlagNaN == False:
            fit = fit_diffusivity_from_msd(
                t=TimeArray,
                msd_t=msd_vs_time[:, i],
                msd_stderr_t=msd_err_vs_time[:, i],   # weighted fit
                dim=1,                    # <Δr^2> = 2 D t + C (radial 1D)
                t_min=0.5 * TimeArray.max(),      # optional: fit late-time window
                t_max=TimeArray.max(),
)
            Dbin[i] = fit["D"]
            slopeBin[i] = fit["slope"]
            interceptBin[i] = fit["intercept"]
        
            print(f"Dneo = {Dneo}")
            print("D =", fit["D"], "+/-", fit["D_stderr"])
            print("R^2 =", fit["r2"])

    fit_curve = 2*Dbin[loc]*TimeArray+interceptBin[loc]

    data_to_plot = {
        r"$<\Delta r^2>$": (msd_vs_time[:, loc], "red", "-"),
        "Fit": (fit_curve, "blue", "--")
    }
    
    plot1D_multiple_curves(f"figures/fitCompare.png", data_to_plot, TimeArray, r"$tv_{Ti}/a$", r'$<\Delta r^2>$', PlotLogScale=False)

    FirstIndex, LastIndex = 10, nrbins - 7
    data_to_plot = {
        "Jonta": (Dbin[FirstIndex: LastIndex], "red", "x"),
        "Analytic": (DneoBin[FirstIndex: LastIndex], "blue", "-")
    }
    print(f"Dneo/Danalytic = {DneoBin[FirstIndex: LastIndex]/Dbin[FirstIndex: LastIndex]}")    
    plot1D_multiple_curves(f"figures/Dneo.png", data_to_plot, r_centers[FirstIndex: LastIndex], '$r/a$', r'$D_{neo}$', PlotLogScale=True)

    ratio = DneoBin/Dbin
    data_to_plot = {
       r"$D_{analytic}/D_{JONTA}$": (ratio[FirstIndex: LastIndex], "red", "x"),
    }    
    plot1D_multiple_curves(f"figures/ratioDneo.png", data_to_plot, r_centers[FirstIndex: LastIndex], '$r/a$', r"$D_{analytic}/D_{JONTA}$", PlotLogScale=False)
    
if __name__ == "__main__":
    main()
