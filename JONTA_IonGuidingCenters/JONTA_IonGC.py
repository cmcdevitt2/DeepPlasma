import os,time,threading,queue
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import tqdm
from functools import partial
from dataclasses import dataclass
import h5py

jax.config.update("jax_enable_x64", False)

plt.rcParams.update({'font.size': 25})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# Set initial energy and pitch distribution of ions
EnergyeV = 2e4 # energy where ions will be centered about
DE = 1.e-6 * EnergyeV # spread in energy of ions
xiInit = 0.0
Dxi = 1.0
Ts = 1e3

vInit = np.sqrt(EnergyeV/Ts)
Dv = np.sqrt(DE/Ts)

vMin, vMax = vInit-Dv, vInit+Dv
xiMin, xiMax = xiInit-Dxi, xiInit+Dxi
Xmin, Xmax = -1.0, 1.0
Zmin, Zmax = -1.0, 1.0
Te = Ts
q = 2
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

# time length and steps
tfinal = 100_000
NumOutputs = 11
tplot = int(tfinal/(NumOutputs-1)) # interval over which to plot solution
dt_ = jnp.array(1.e-1) # time step
NumMarkerPart = 1_000_000

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

# Flags
PitchAngleScatOn = True

if PitchAngleScatOn == False:
    nu_D = 0.0

# ---------------- Params ----------------
@jax.tree_util.register_pytree_node_class
@dataclass
class McDevittParams:
    epsilon: jnp.ndarray
    q:       jnp.ndarray
    nu_D:    jnp.ndarray
    rhostar: jnp.ndarray
    def tree_flatten(self):
        return (self.epsilon, self.q, self.nu_D, self.rhostar), None
    @classmethod
    def tree_unflatten(cls, aux, children):
        epsilon, q, nu_D, rhostar = children
        return cls(epsilon, q, nu_D, rhostar)


# ---------------- Helpers ----------------
def IonGuidingCenterEqs(v, xi, X, Z, params: McDevittParams):
    epsilon, q, rhostar = params.epsilon, params.q, params.rhostar

    rSq = X**2 + Z**2
    B = 1 / (1+epsilon*X)
    
    vdot = 0.0
    xidot = -0.5 * (1-xi**2) * v / q * epsilon**2 * B * Z
    Xdot = -xi*v*Z/q*epsilon
    Zdot = xi*v*X/q*epsilon - 0.5 * rhostar * (1+xi**2) * v**2 * epsilon
    
    return v, xi, X, Z, vdot, xidot, Xdot, Zdot


def nuD_from_v(v, params: McDevittParams):
    nu_D = params.nu_D
    return nu_D / (v**3 + 1e-30)

def absorb_mask(v, xi, X, Z, params: McDevittParams):
    """Boolean mask of survivors (p >= p_th)."""
    rSQ = X**2 + Z**2
    
    return rSQ > 1.0


# ---------------- One RK4 + scatter step with fixed dt ----------------
@jit
def step_rk4_scatter_fixeddt(key, v, xi, X, Z, tauEsc, XEsc, ZEsc, t_k, dt, params: McDevittParams):
    # Ensure scalar dt with correct dtype
    dt = jnp.asarray(dt, dtype=v.dtype)

    # k1
    _, _, _, _, v1, xi1, X1, Z1 = IonGuidingCenterEqs(v, xi, X, Z, params)
    # k2
    v2_state  = v    + 0.5*dt*v1
    xi2_state = xi   + 0.5*dt*xi1
    X2_state  = X    + 0.5*dt*X1
    Z2_state  = Z    + 0.5*dt*Z1
    _, _, _, _, v2, xi2, X2, Z2 = IonGuidingCenterEqs(v2_state, xi2_state, X2_state, Z2_state, params)
    # k3
    v3_state  = v    + 0.5*dt*v2
    xi3_state = xi   + 0.5*dt*xi2
    X3_state  = X    + 0.5*dt*X2
    Z3_state  = Z    + 0.5*dt*Z2
    _, _, _, _, v3, xi3, X3, Z3 = IonGuidingCenterEqs(v3_state, xi3_state, X3_state, Z3_state, params)
    # k4
    v4_state  = v    + dt*v3
    xi4_state = xi   + dt*xi3
    X4_state = X     + dt*X3
    Z4_state = Z     + dt*Z3
    _, _, _, _, v4, xi4, X4, Z4 = IonGuidingCenterEqs(v4_state, xi4_state, X4_state, Z4_state, params)

    # update quantities
    v_det    = v     + (dt/6.0) * (v1 + 2*v2 + 2*v3 + v4)
    xi_det   = xi    + (dt/6.0) * (xi1 + 2*xi2 + 2*xi3 + xi4)
    X_det    = X     + (dt/6.0) * (X1 + 2*X2 + 2*X3 + X4)
    Z_det    = Z     + (dt/6.0) * (Z1 + 2*Z2 + 2*Z3 + Z4)
    # xi_det    = jnp.clip(xi_det, -1.0, 1.0)

    # ν_D at updated gamma
    nuD = nuD_from_v(v_det, params)

    # Stochastic pitch-angle kick (same Rademacher noise style you used)
    key, k_u = random.split(key)
    randu = random.uniform(k_u, shape=xi.shape)
    signs = jnp.where(randu < 0.5, -1.0, 1.0)

    one_minus_xi2 = jnp.clip(1.0 - xi_det*xi_det, 0.0, 1.0)
    sigma = jnp.sqrt(jnp.maximum(one_minus_xi2 * nuD * dt, 0.0))
    xi_new = xi_det * (1.0 - nuD * dt) + signs * sigma

    # Only a deterministic step for these unknowns for now
    v_new = v_det
    X_new = X_det
    Z_new = Z_det

    # absorbing boundary (park absorbed to keep shapes static)
    alive = tauEsc < 0 # Check if particle was in domain
    escaped_now = absorb_mask(v_new, xi_new, X_new, Z_new, params) # See if it just escaped
    
    # If the particle was alive and just escaped, return the current time
    # Add 0.5dt to take the midpoint of the step
    tauEsc_new = jnp.where(escaped_now & alive, jnp.asarray(t_k+0.5*dt, dtype=jnp.float32), tauEsc)
    XEsc_new = jnp.where(escaped_now & alive, 0.5*(X_new+X), XEsc)
    ZEsc_new = jnp.where(escaped_now & alive, 0.5*(Z_new+Z), ZEsc)
    
    return key, v_new, xi_new, X_new, Z_new, tauEsc_new, XEsc_new, ZEsc_new


# ---------------- Initialization ----------------
def initialize_particles_jax(
        key,
        N,
        *,
        init_mode="uniform",
        vmin=0.2, vmax=2.0,
        ximin=-1.0, ximax=1.0,
        Xmin=-1.0, Xmax=1.0,
        Zmin=-1.0, Zmax=1.0,
        v0=1.0, xi0=0.0, X0=0.0, Z0=0.0,
        sigma_v=0.2, sigma_xi=0.2, sigma_X=0.2, sigma_Z=0.2,
        start_id=0,
        device_unique=False,
        axis_name=None,  # e.g., set to "devices" when pmapping
):

    # Synchronous, deterministic IDs (no races, no in-place writes).
    if device_unique and axis_name is not None:
        # Make IDs unique per device (assumes equal N on each device).
        dev_idx = lax.axis_index(axis_name)
        ids = start_id + dev_idx * N + jnp.arange(N, dtype=jnp.int32)
    else:
        ids = start_id + jnp.arange(N, dtype=jnp.int32)

    # Initialize escape times to -1.0, since none of have escaped yet
    tauEsc = jnp.full((N,), -1.0, dtype=jnp.float32)
    XEsc = jnp.full((N,), 0.0, dtype=jnp.float32)
    ZEsc = jnp.full((N,), 0.0, dtype=jnp.float32)
    
    if init_mode == "uniform":
        key, kv, kxi, kX, kZ  = random.split(key, 5)
        v  = random.uniform(kv, (N,)) * (vmax - vmin) + vmin
        xi = random.uniform(kxi, (N,)) * (ximax - ximin) + ximin
        X = random.uniform(kX, (N,)) * (Xmax - Xmin) + Xmin
        Z = random.uniform(kZ, (N,)) * (Zmax - Zmin) + Zmin
    elif init_mode == "gaussian":
        key, kv, kxi, kX, kZ  = random.split(key, 5)
        v  = random.normal(kp, (N,)) * sigma_v + v0
        xi = random.normal(kx, (N,)) * sigma_xi + xi0
        X  = random.normal(kX, (N,)) * sigma_X + X0
        Z  = random.normal(kZ, (N,)) * sigma_Z + Z0
        v  = jnp.clip(v,  vmin, vmax)
        xi = jnp.clip(xi, ximin, ximax)
        X  = jnp.clip(X,  Xmin, Xmax)
        Z  = jnp.clip(Z,  Zmin, Zmax)
    else:
        raise ValueError("init_mode must be 'uniform' or 'gaussian'")
    return key, v, xi, X, Z, ids, tauEsc, XEsc, ZEsc


def _writer_thread_fn(q: "queue.Queue", device_dir: str):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
            
        step_idx, v, xi, X, Z, ids, tauEsc, XEsc, ZEsc = item
        fname = os.path.join(device_dir, f"step_{step_idx:06d}.h5")

        with h5py.File(fname, "w") as f:
            f.create_dataset("v",      data=v,      compression="lzf")
            f.create_dataset("xi",     data=xi,     compression="lzf")
            f.create_dataset("X",      data=X,      compression="lzf")
            f.create_dataset("Z",      data=Z,      compression="lzf")
            f.create_dataset("ids",    data=ids,    compression="lzf")
            f.create_dataset("tauEsc", data=tauEsc, compression="lzf")
            f.create_dataset("XEsc", data=XEsc, compression="lzf")
            f.create_dataset("ZEsc", data=ZEsc, compression="lzf")

        q.task_done()


# ---------------- Fixed-dt runner ----------------
def run_with_fixed_dt(
        *,
        key, v, xi, X, Z,
        ids, tauEsc, XEsc, ZEsc,
        params,
        dt: float,         # fixed timestep
        n_steps: int,
        plot_every: int = 0,
        plot_fn1=None,
        plot_fn2=None,
):
    # JIT a single fixed-dt step to get maximum speed in the loop
    step_jit = jax.jit(lambda k, v, xi, X, Z, tauEsc, XEsc, ZEsc, t_k: step_rk4_scatter_fixeddt(k, v, xi, X, Z, tauEsc, XEsc, ZEsc, t_k, dt, params))

    t_k = jnp.asarray(0.0, dtype=jnp.float32)
    # Warm-up compilation
    k_tmp, v_tmp, xi_tmp, X_tmp, Z_tmp, tauEsc_tmp, XEsc_tmp, ZEsc_tmp = step_jit(key, v, xi, X, Z, tauEsc, XEsc, ZEsc, t_k)
    v_tmp.block_until_ready(); xi_tmp.block_until_ready();
    X_tmp.block_until_ready(); Z_tmp.block_until_ready();
    tauEsc_tmp.block_until_ready();
    XEsc_tmp.block_until_ready(); ZEsc_tmp.block_until_ready();

    # Define output location
    base_dir: str = "./particles"
    run_id: str = "run_0001"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    di = 0
    ddir = os.path.join(run_dir, f"device_{di}")
    os.makedirs(ddir, exist_ok=True)
    q = queue.Queue(maxsize=9)  # buffer; tune if needed
    t = threading.Thread(target=_writer_thread_fn, args=(q, ddir), daemon=True)
    t.start()

    q.put((0, v, xi, X, Z, ids, tauEsc, XEsc, ZEsc))
    for step in tqdm.trange(1, n_steps + 1):
        t_k = jnp.float32(step) * jnp.float32(dt)
        key, v, xi, X, Z, tauEsc, XEsc, ZEsc = step_jit(key, v, xi, X, Z, tauEsc, XEsc, ZEsc, t_k)
        if plot_every and plot_fn1 and plot_fn2 and (step % plot_every == 0):
            v_host  = np.asarray(v)
            xi_host = np.asarray(xi)
            X_host = np.asarray(X)
            Z_host = np.asarray(Z)
            plot_fn1(v_host, xi_host, t_k)
            plot_fn2(X_host, Z_host, t_k)

            q.put((step, v, xi, X, Z, ids, tauEsc, XEsc, ZEsc))
    q.put(None)
    q.join()
    t.join()
    
    return key, v, xi, X, Z, ids, tauEsc, XEsc, ZEsc


# ---------------- Simple plotting ----------------
def plot_scatter_vxi(v, xi, t):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(v, xi, '.', ms=0.5, alpha=0.1, color='black')
    ax.set_xlabel('$v_{Ti}$')
    ax.set_ylabel(r'$\xi$')
    ax.set_xlim(0,10)
    ax.set_ylim(-1, 1)
    ax.set_title(rf'$t = {t:.2f}$')
    if t==0:
        fig.savefig(f'figures/jaxtestvxi_Init')
    else:
        fig.savefig(f'figures/jaxtestvxi_{int(t)}')
    plt.close(fig)
    

def plot_scatter_XZ(X, Z, t):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(X, Z, '.', ms=0.5, alpha=0.1, color='black')
    ax.set_xlabel('$(R-R_0/a)$')
    ax.set_ylabel(r'$Z$')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1, 1)
    ax.set_title(rf'$t = {t:.2f}$')
    if t==0:
        fig.savefig(f'figures/jaxtestXZ_Init')
    else:
        fig.savefig(f'figures/jaxtestXZ_{int(t)}')
    plt.close(fig)

    
def plot_fn_vxi(v_host, xi_host, t):
    plot_scatter_vxi(v_host, xi_host, t)


def plot_fn_XZ(X_host, Z_host, t):
    plot_scatter_XZ(X_host, Z_host, t)


# ---------------- Example call ----------------
params = McDevittParams(
    epsilon =jnp.array(epsilon),
    q       = jnp.array(q),
    nu_D    = jnp.array(nu_D),
    rhostar = jnp.array(rhostar)
)
key = random.PRNGKey(123)
key, v, xi, X, Z, ids, tauEsc, XEsc, ZEsc = initialize_particles_jax(
    key, N=NumMarkerPart, init_mode="uniform",
    vmin=vMin, vmax=vMax, ximin=xiInit-Dxi, ximax=xiInit+Dxi,
    Xmin=Xmin, Xmax=Xmax, Zmin=Zmin, Zmax=Zmax
)
v = np.asarray(v)
xi = np.asarray(xi)
X = np.asarray(X)
Z = np.asarray(Z)
plot_fn_vxi(v, xi, 0.0)
plot_fn_XZ(X, Z, 0.0)

# Fixed timestep
dt = dt_
n_steps = int(tfinal/dt)

key, v, xi, X, Z, ids, tauEsc, XEsc, ZEsc = run_with_fixed_dt(
    key=key, v=v, xi=xi, X=X, Z=Z,
    ids=ids, tauEsc=tauEsc, XEsc=XEsc, ZEsc=ZEsc,
    params=params, dt=dt, n_steps=n_steps,
    plot_every=int(tplot/dt), plot_fn1=plot_fn_vxi, plot_fn2=plot_fn_XZ
)

