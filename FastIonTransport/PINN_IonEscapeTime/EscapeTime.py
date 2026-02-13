import torch
import torch.nn as nn
import torch.utils.checkpoint
import numpy as np
from scipy.optimize import minimize
import pytorch_optimizer
from torch.optim import lr_scheduler
import skopt
import matplotlib.pylab as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib import ticker
import os, shutil


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

torch.manual_seed(1234)
np.random.seed(1234)
torch.set_default_dtype(torch.float64)

# Physics constants
EnergyeV = 1e4
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
nu_SD = mbulk/ms * nu_D

print(f"rhostar = {rhostar}")
print(f"nu_D = {nu_D}")

v = np.sqrt(EnergyeV/Ts) # normalized speed
rhostar_v = rhostar * v
wc_v = 1 / rhostar_v

rMin, rMax = 0, 1
thetaMin, thetaMax = -np.pi, np.pi
xiMin, xiMax = -1, 1


alpha = 1
A = 100
wPDE = 100
wBC = 1
lr = 5.e-4
N = 2_000_000
Nbc = 1_000_000
N_test = 1_000_000
epochSOAP = 20_001
resample_every = 8000 # residual based or random reshuffling of training points
resample_train_points = True
k, c = 1.0, 1.0
frac = 0.02
pool_mult = 25

LoadModel = False # indicate whether a pretrained model should be loaded
LoadTrainingPts = False # indicate if training points should be loaded from a file
NameModelToLoad = 'model/pinn_weights_epoch_45000.pth'
NamePointsToLoad = 'model/train_points.pth'
NameTrainingLossToLoad = 'model/TrainingLosses.dat'
NameTestLossToLoad = 'model/TestLosses.dat'
TrainSOAP = True
TrainSSBroyden = TrainSOAP

epochSSBroyden = 5000
InputDim = 3

# Points at which cross sections of data will be plotted
r_val = 0.9999
theta_val = 0
xi_val = -0.8


r_valNorm = (r_val-rMin) / (rMax-rMin)
theta_valNorm = (theta_val-thetaMin) / (thetaMax-thetaMin)
xi_valNorm = (xi_val-xiMin) / (xiMax-xiMin)

# Save python script inside model directory
if TrainSOAP==True or TrainSSBroyden == True:
    src = os.path.realpath(__file__)
    dst = os.path.join('model', os.path.basename(src))
    shutil.copy2(src, dst)


# -----------------
# PINN architecture
# -----------------
class PINN(nn.Module):
    def __init__(self, in_dim=InputDim, h_dim=50, out_dim=1):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, h_dim); self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(h_dim, h_dim);  self.act2 = nn.Tanh()
        self.linear3 = nn.Linear(h_dim, h_dim);  self.act3 = nn.Tanh()
        self.linear4 = nn.Linear(h_dim, h_dim);  self.act4 = nn.Tanh()
        self.linear5 = nn.Linear(h_dim, h_dim);  self.act5 = nn.Tanh()
        self.linear6 = nn.Linear(h_dim, h_dim);  self.act6 = nn.Tanh()
        self.linear7 = nn.Linear(h_dim, h_dim);  self.act7 = nn.Tanh()
        self.linearLast = nn.Linear(h_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in [
                self.linear1,
                self.linear2,
                self.linear3,
                self.linear4,
                self.linear5,
                self.linear6,
                self.linear7,
                self.linearLast
        ]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        x = self.act3(self.linear3(x))
        x = self.act4(self.linear4(x))
        x = self.act5(self.linear5(x))
        x = self.act6(self.linear6(x))
        x = self.act7(self.linear7(x))
        x = self.linearLast(x)
        return x
   
# ----------------
# Output transform
# ----------------
def output_transform(TNN):
    Tesc = torch.exp(TNN[:,0:1])
    
    return Tesc


def input_transform(inputs):
    rNorm, thetaNorm, xiNorm = inputs.split(1,dim=1)

    r  = rMin + (rMax  - rMin ) * rNorm
    theta = thetaMin + ( thetaMax - thetaMin ) * thetaNorm
    xi = xiMin + (xiMax - xiMin) * xiNorm
    
    cos1 = torch.cos(theta)
    sin1 = torch.sin(theta)

    R = 1 + epsilon*r*cos1 # normalized to R_0

    Z = r*sin1
    X = R - 1
    
    return torch.cat([X, Z, xi], dim=1)


# ----------------
#       PDE
# ----------------
def fp_pde(model, rNorm, thetaNorm, xiNorm):
    rNorm  = rNorm.clone().detach().requires_grad_(True)
    thetaNorm = thetaNorm.clone().detach().requires_grad_(True)
    xiNorm = xiNorm.clone().detach().requires_grad_(True)

    inputs = torch.cat([rNorm, thetaNorm, xiNorm], dim=1)
    X = input_transform(inputs) 

    r  = rMin + (rMax  - rMin ) * rNorm
    theta = thetaMin + ( thetaMax - thetaMin ) * thetaNorm
    xi = xiMin + (xiMax - xiMin) * xiNorm
    
    cos1 = torch.cos(theta)
    sin1 = torch.sin(theta)

    R = 1 + epsilon*r*cos1 # normalized to R_0
    
    TNN = model(X)
    T = output_transform(TNN)
    
    dy_r, dy_theta, dy_xi    = torch.autograd.grad(T, (rNorm, thetaNorm, xiNorm) , grad_outputs=torch.ones_like(T), create_graph=True)

    dy_r = dy_r / (rMax-rMin)
    dy_theta = dy_theta / (thetaMax-thetaMin)
    dy_xi = dy_xi / (xiMax-xiMin)
    dy_xixi = torch.autograd.grad(dy_xi, xiNorm, grad_outputs=torch.ones_like(T), create_graph=True)[0] / (xiMax-xiMin)

    B = 1/R
    
    CollisionalTerms = -r * 0.5*(nu_D/v**3)*( (1-xi**2)*dy_xixi - 2*xi*dy_xi )
    ParallelStreaming = -r * v*xi/q*epsilon * dy_theta
    PerpDrifts = 0.5*epsilon*v**2*rhostar * (1+xi**2) * ( r*sin1*dy_r + cos1*dy_theta )
    MirrorForce = r**2 * 0.5 * v*(1-xi**2)/q*epsilon**2*B * sin1 * dy_xi

    condtheta1 = torch.where(torch.logical_and(torch.abs(theta) < 0.1, r > 0.99), torch.zeros_like(T), torch.ones_like(T))
    condtheta2 = torch.where(torch.logical_and(torch.abs(theta) > np.pi - 0.1, r > 0.99), torch.zeros_like(T), torch.ones_like(T))
    FrontFactor = A / (A + T**alpha)

    residual = condtheta1*condtheta2 * FrontFactor * ( ParallelStreaming + PerpDrifts + MirrorForce + CollisionalTerms - r )
    
    return residual


def bc_rMax(inputs, outputs):
    XX, Z, xi = inputs.split(1,dim=1)
    T = output_transform(outputs)
    
    cond = torch.where(Z<0.0, torch.ones_like(T), torch.zeros_like(T))

    return cond * (T-0.0)**2


# ----------------------
# Wrapper for SSBroyden
# ----------------------
def get_flat_params(model):
   params = []
   for param in model.parameters():
       params.append(param.detach().cpu().numpy().reshape(-1))
   return np.concatenate(params)

def set_flat_params(model, flat_params):
   idx = 0
   for param in model.parameters():
       numel = param.numel()
       param_np = flat_params[idx:idx+numel].reshape(param.shape)
       param.data.copy_(torch.tensor(param_np, dtype=param.dtype, device=param.device))
       idx += numel


def lbfgs_loss_and_grad(flat_params, model, loss_fn, rNorm_train, thetaNorm_train, xiNorm_train):
   set_flat_params(model, flat_params)
   model.zero_grad()
   loss = loss_fn(model, rNorm_train, thetaNorm_train, xiNorm_train)
   loss.backward()
   grads = []
   for param in model.parameters():
       grads.append(param.grad.detach().cpu().numpy().reshape(-1))
   flat_grad = np.concatenate(grads)
   return loss.item(), flat_grad


def pinn_loss(model, rNorm, thetaNorm, xiNorm):
    res = fp_pde(model, rNorm, thetaNorm, xiNorm)
    loss_pde = wPDE * torch.mean(res**2)
    loss_bc  = wBC * torch.mean( bc_rMax(X_bound, model(X_bound)) )
    TrainingLossPDE.append(loss_pde.detach().cpu().numpy())
    TrainingLossBC.append(loss_bc.detach().cpu().numpy())
    EpochLocal = StepsTraining[-1]+1
    StepsTraining.append(EpochLocal)
    if EpochLocal % 500 == 0: # save model occasionally
        torch.save(model.state_dict(), f"model/pinn_weights_epoch_{EpochLocal:05d}.pth")

    return loss_pde + loss_bc


def fit_lbfgsb(model, loss_fn, rNorm_train, thetaNorm_train, xiNorm_train, maxiter, print_every):
   flat_params_init = get_flat_params(model)
   nfeval = [1]
   def callback(params):
       if nfeval[0] % print_every == 0:
           set_flat_params(model, params)
           loss = loss_fn(model, rNorm_train, thetaNorm_train, xiNorm_train)
           test_PDE, test_BC = compute_test_loss()
           PrevGlobalEpoch = epochGlobalList[-1] if epochGlobalList else 0
           TestPDE.append(test_PDE)
           TestBC.append(test_BC)
           epochGlobalList.append(PrevGlobalEpoch+print_every)
           print(f"Epoch {nfeval[0]}: Training Loss = {loss.item():.4e}, Test PDE Loss = {test_PDE:.4e}, Test BC Loss = {test_BC:.4e}")
       nfeval[0] += 1
   res = minimize(
       lbfgs_loss_and_grad,
       flat_params_init,
       args=(model, loss_fn, rNorm_train, thetaNorm_train, xiNorm_train),
       method='BFGS',
       jac=True,
       callback=callback,
       options={'maxiter': maxiter, 'disp': None}
   )
   set_flat_params(model, res.x)
   return res


# memory‑friendly test loss (chunked; accumulate on CPU)
def compute_test_loss(max_batch: int = 65536) -> float:
    was_training = model.training
    model.eval()
    res_sq_err = 0.0
    count = 0
    with torch.set_grad_enabled(True):  # need gradients to compute residual
        for s in range(0, N_test, max_batch):
            r = rNorm_test[s:s+max_batch]
            theta = thetaNorm_test[s:s+max_batch]
            xi = xiNorm_test[s:s+max_batch]
            res = fp_pde(model, r, theta, xi)        # residuals on GPU
            res_sq_err += res.pow(2).sum().item()  # move scalar to CPU immediately
            count += res.numel() # sum number of points

            del r, theta, xi, res  # free references promptly

    loss_bc = torch.mean( bc_rMax(X_bound_test, model(X_bound_test)) ).item()
            
    if was_training:
        model.train()
    return wPDE * res_sq_err / max(1, count), wBC * loss_bc


# Hammersly sampling
def hammersley_sequence(n_samples, dim):
    skip = 0
    if dim == 1:
        sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
    else:
        sampler = skopt.sampler.Hammersly()
        skip = 1
    space = [(0.0, 1.0)] * dim
    points = np.asarray(sampler.generate(space, n_samples + skip)[skip:],dtype=np.float64)
    return points


# Adaptive collocation resampling based on residual magnitude (DeepXDE-style)
# Selects more points where |residual| is large; k controls sharpness, c adds a uniform floor.
def adaptive_resample(
    model,
    fp_pde,
    N_target: int,
    *,
    k: float = 1.0,
    c: float = 1.0,
    frac: float = 0.1,
    pool_mult: int = 25,
    chunk_size: int = 65536,
    device=None,
):
    assert 0 < frac <= 1.0
    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # number of increments and per-increment quota
    increments = max(1, int(round(1.0 / frac)))
    per_inc = int(round(N_target * frac))
    # adjust last increment to hit N_target exactly
    sizes = [per_inc] * (increments - 1) + [N_target - per_inc * (increments - 1)]

    selected_r = []
    selected_theta = []
    selected_xi = []

    model_was_training = model.training
    model.eval()

    for n_select in sizes:
        if n_select <= 0:
            continue
        M = pool_mult * n_select  # candidate pool size

        # draw candidate points uniformly on the domain [0,1]^2 (adjust if needed)
        r_pool = torch.rand(M, 1, device=device, dtype=dtype)
        theta_pool = torch.rand(M, 1, device=device, dtype=dtype)
        xi_pool = torch.rand(M, 1, device=device, dtype=dtype)

        # evaluate residual magnitude in chunks (enable grad: fp_pde uses autograd w.r.t. inputs)
        f_list = []
        with torch.set_grad_enabled(True):
            for s in range(0, M, chunk_size):
                r = r_pool[s:s+chunk_size].detach().requires_grad_(True)
                theta = theta_pool[s:s+chunk_size].detach().requires_grad_(True)
                xi = xi_pool[s:s+chunk_size].detach().requires_grad_(True)
                res = fp_pde(model, r, theta, xi)
                f_list.append(res.detach().abs().flatten())  # store |residual|
                del r, theta, xi, res
        f = torch.cat(f_list)  # shape [M]

        # importance weights: (f^k / mean(f^k)) + c, then normalize
        fk = torch.clamp(f, min=0).pow(k)
        mean_fk = fk.mean().clamp(min=torch.finfo(fk.dtype).tiny)
        w = fk / mean_fk + c
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w_sum = w.sum().clamp(min=torch.finfo(w.dtype).tiny)
        probs = w / w_sum

        # sample without replacement according to probs
        idx = torch.multinomial(probs, n_select, replacement=False)
        selected_r.append(r_pool.index_select(0, idx))
        selected_theta.append(theta_pool.index_select(0, idx))
        selected_xi.append(xi_pool.index_select(0, idx))

        del r_pool, theta_pool, xi_pool, f_list, f, fk, mean_fk, w, w_sum, probs, idx

    if model_was_training:
        model.train()

    r_sel = torch.cat(selected_r, dim=0)
    theta_sel = torch.cat(selected_theta, dim=0)
    xi_sel = torch.cat(selected_xi, dim=0)
    # return tensors shaped [N_target, 1] on the same device/dtype as the model
    return r_sel, theta_sel, xi_sel


def save_losses(steps, msePDE, mseBC, out_path='losses.dat'):
    # Convert inputs to 1D NumPy arrays (handles Python lists and torch.Tensors)
    def to_np(x):
        try:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().reshape(-1)
        except Exception:
            pass
        return np.asarray(x, dtype=np.float64).reshape(-1)

    s  = to_np(steps)
    mPDE = to_np(msePDE)
    mBC = to_np(mseBC)

    if not (len(s) == len(mPDE) == len(mBC)):
        raise ValueError(f'All inputs must have the same length, got: steps={len(s)}, mse1={len(mPDE)}, mse2={len(mBC)}')

    data = np.column_stack([s, mPDE, mBC])
    # text file with header
    np.savetxt(out_path, data, delimiter=',', comments='', fmt='%.8e')


def load_losses(path='losses'):
    # Try CSV with a header line first; fall back to no header
    data = np.loadtxt(path, delimiter=',')
    """
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except Exception:
        data = np.loadtxt(path, delimiter=',')
    """
    Steps = data[:, 0].tolist()
    msePDE  = data[:, 1].tolist()
    mseBC  = data[:, 2].tolist()
    return Steps, msePDE, mseBC

def GenerateTrainingPts():
    rMinEdge, rMaxEdge = 0.95, 1
    Nedge = int(np.round(0.25*N))
    Ninterior = int(N - Nedge)

    # Hammersley
    X_trainInterior = torch.tensor(hammersley_sequence(Ninterior, InputDim), dtype=torch.float64, device=device).requires_grad_(True)
    X_trainEdge = torch.tensor(hammersley_sequence(Nedge, InputDim), dtype=torch.float64, device=device).requires_grad_(True)
    rNorm_trainInterior, thetaNorm_trainInterior, xiNorm_trainInterior = X_trainInterior[:,0:1], X_trainInterior[:,1:2], X_trainInterior[:,2:3]
    rNorm_trainEdge, thetaNorm_trainEdge, xiNorm_trainEdge = X_trainEdge[:,0:1], X_trainEdge[:,1:2], X_trainEdge[:,2:3]
    rNorm_trainEdge = rMinEdge + (rMaxEdge - rMinEdge) * rNorm_trainEdge

    # Need to append training points for rNorm_trainInterior, thetaNorm_trainInterior, xiNorm_trainInterior and rNorm_trainEdge, thetaNorm_trainEdge, xiNorm_trainEdge
    X_train = torch.cat([
        torch.cat([rNorm_trainInterior, thetaNorm_trainInterior, xiNorm_trainInterior], dim=1),
        torch.cat([rNorm_trainEdge, thetaNorm_trainEdge, xiNorm_trainEdge], dim=1),
    ], dim=0)

    # Reshuffle interior and edge points
    perm = torch.randperm(X_train.shape[0], device=device)
    X_train = X_train[perm]
    
    rNorm_bc = torch.ones((Nbc, 1), device=device) # set r = 1, since the boundary condition is only applied at r=r_Max
    thetaNorm_bc  = torch.rand(Nbc,1,device=device)
    xiNorm_bc = torch.rand(Nbc,1,device=device)

    thetaNorm_bc = 0.5 * thetaNorm_bc # only sample negative theta, so new values range from -pi to 0
    
    inputs_bc = torch.cat([rNorm_bc, thetaNorm_bc, xiNorm_bc], dim=1)
    X_bound = input_transform(inputs_bc)

    return X_train, X_bound


def GenerateTestPts():
    rNorm_test = torch.rand(N_test, 1, device=device)
    thetaNorm_test = torch.rand(N_test, 1, device=device)
    xiNorm_test = torch.rand(N_test, 1, device=device)
    X_test = torch.cat([rNorm_test, thetaNorm_test, xiNorm_test], dim=1)
    
    rNorm_bc_test = torch.ones((Nbc, 1), device=device) # set r = 1, since the boundary condition is only applied at r=r_Max
    thetaNorm_bc_test = torch.rand(Nbc,1,device=device)
    xiNorm_bc_test = torch.rand(Nbc,1,device=device)

    thetaNorm_bc_test = 0.5 * thetaNorm_bc_test # only sample negative theta, so new values range from -pi to 0

    inputs_bc_test = torch.cat([rNorm_bc_test, thetaNorm_bc_test, xiNorm_bc_test], dim=1)
    X_bound_test = input_transform(inputs_bc_test)

    return X_test, X_bound_test


TrainingLossPDE = []
TrainingLossBC = []
StepsTraining = []

epochGlobal = 0
epochGlobalList = []
TestPDE = []
TestBC = []

# -----------
# Training
# -----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN().to(device)
if LoadModel == True:
    state = torch.load(NameModelToLoad, map_location=device)  # device='cpu' or 'cuda'
    model.load_state_dict(state)
    model.eval()
    if os.path.exists(NamePointsToLoad) and LoadTrainingPts == True:
        checkpoint = torch.load(NamePointsToLoad, map_location=device)
        rNorm_train = checkpoint['rNorm_train']
        thetaNorm_train = checkpoint['thetaNorm_train']
        xiNorm_train = checkpoint['xiNorm_train']

        _, X_bound = GenerateTrainingPts() # generate boundary pts
    else:
        X_train, X_bound = GenerateTrainingPts()
        rNorm_train, thetaNorm_train, xiNorm_train = X_train[:,0:1], X_train[:,1:2], X_train[:,2:3]
    if os.path.exists(NameTrainingLossToLoad):
        StepsTraining, TrainingLossPDE, TrainingLossBC = load_losses(NameTrainingLossToLoad)
    if os.path.exists(NameTestLossToLoad):
        epochGlobalList, TestPDE, TestBC = load_losses(NameTestLossToLoad)
else:
    X_train, X_bound = GenerateTrainingPts()
    rNorm_train, thetaNorm_train, xiNorm_train = X_train[:,0:1], X_train[:,1:2], X_train[:,2:3]

X_test, X_bound_test = GenerateTestPts()
rNorm_test, thetaNorm_test, xiNorm_test = X_test[:,0:1], X_test[:,1:2], X_test[:,2:3]
'''
If you want to train with SOAP/ADAM/LBFGS
'''
if TrainSOAP == True:
    optimizer = pytorch_optimizer.optimizer.soap.SOAP(model.parameters(), lr=lr, betas=(.999, .999), weight_decay=0e-2, precondition_frequency=1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999995)

    for epoch in range(epochSOAP):
        # periodic resampling (in-place to minimize memory)
        if resample_train_points and (epoch % resample_every == 0) and epoch > 0:
            rNorm_train, thetaNorm_train, xiNorm_train = adaptive_resample(
                model, fp_pde, N_target=N,
                k=k, c=c, frac=frac, pool_mult=pool_mult,
                chunk_size=65536, device=device
            )
        
        optimizer.zero_grad(set_to_none=True)
        res = fp_pde(model, rNorm_train, thetaNorm_train, xiNorm_train)
        loss_pde = wPDE * torch.mean(res**2)
        loss_bc = wBC * torch.mean( bc_rMax(X_bound, model(X_bound)) )
        loss = loss_pde + loss_bc
        TrainingLossPDE.append(loss_pde.detach().cpu().numpy())
        TrainingLossBC.append(loss_bc.detach().cpu().numpy())
        StepsTraining.append(epoch)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total Training Loss = {loss.item():.2e}, PDE loss = {loss_pde.item():.2e}, boundary loss = {loss_bc.item():.2e}")
            # checkpoint and compute test loss
        if epoch % 5000 == 0: # save model occasionally
            test_PDE, test_BC = compute_test_loss()
            TestPDE.append(test_PDE)
            TestBC.append(test_BC)
            epochGlobalList.append(epochGlobal)
            print(f"Epoch {epoch}: PDE Test Loss = {test_PDE:.2e}, BC Test Loss = {test_BC:.2e}")
            torch.save(model.state_dict(), f"model/pinn_weights_epoch_{epoch:05d}.pth")
        scheduler.step()  # step once per epoch
        epochGlobal += 1
        


    print(f"Epoch {epoch}: Total Training Loss = {loss.item():.2e}, PDE loss = {loss_pde.item():.2e}, boundary loss = {loss_bc.item():.2e}")
    final_test_PDE, final_test_BC = compute_test_loss()
    TestPDE.append(final_test_PDE)
    TestBC.append(final_test_BC)
    epochGlobalList.append(epochGlobal)
    print(f"Test PDE Loss after SOAP = {final_test_PDE:.2e}, Test BC Loss after SOAP = {final_test_BC:.2e}")
    torch.save(model.state_dict(), f"model/pinn_weights_epoch_{epoch:05d}.pth")
    torch.save({
        'rNorm_train': rNorm_train,
        'thetaNorm_train': thetaNorm_train,
        'xiNorm_train': xiNorm_train
    }, 'model/train_points.pth')

    
'''
If you want to train with SSBroyden
'''
if TrainSSBroyden == True:
    fit_lbfgsb(model, pinn_loss, rNorm_train, thetaNorm_train, xiNorm_train,epochSSBroyden,100)

    final_test_loss, test_BC = compute_test_loss()
    print(f"Final Test Loss = {final_test_loss:.2e}")

    # Save trained model
    torch.save(model.state_dict(), 'model/pinn_weights.pth')

if TrainSOAP==True or TrainSSBroyden == True:
    save_losses(StepsTraining, TrainingLossPDE, TrainingLossBC, out_path='model/TrainingLosses.dat')
    save_losses(epochGlobalList, TestPDE, TestBC, out_path='model/TestLosses.dat')


# -----------
# Plotting
# -----------
plt.rcParams.update({'font.size': 18})

n_r_plot = 150
n_theta_plot = 110
n_xi_plot = 160
rValsNorm = np.linspace(0, 0.999, n_r_plot, dtype=np.float64) # avoid r=1
thetaValsNorm = np.linspace(0, 1, n_theta_plot, dtype=np.float64)
xiValsNorm = np.linspace(0, 1, n_xi_plot, dtype=np.float64)

xiNorm = xi_valNorm*np.ones(n_r_plot*n_theta_plot)
rrNorm, ththNorm = np.meshgrid(rValsNorm, thetaValsNorm, indexing="ij")

r  = rMin + (rMax  - rMin ) * rrNorm
theta = thetaMin + ( thetaMax - thetaMin ) * ththNorm
xi = xiMin + (xiMax - xiMin) * xiNorm
    
cos1 = np.cos(theta)
sin1 = np.sin(theta)

R = 1 + epsilon*r*cos1 # normalized to R_0
Z = r*sin1
X = R - 1

r_grid  = torch.tensor(rrNorm, device=device).reshape(-1,1)
theta_grid  = torch.tensor(ththNorm, device=device).reshape(-1,1)
xi_grid  = torch.tensor(xiNorm, device=device).reshape(-1,1)


model.eval()
with torch.no_grad():
    inputs = torch.cat([r_grid, theta_grid, xi_grid], dim=1)
    XX = input_transform(inputs)
    TNN = model(XX)
    T = output_transform(TNN)
    T_plot = T.cpu().numpy().reshape(rrNorm.shape)
   
# R-Z plot of Tesc
fig, ax = plt.subplots()
fig.set_tight_layout(True)

BMax = 1 / ( 1 - r*epsilon )
Bprime = 1 / (1 + epsilon*r*np.cos(theta) )
TrapRegion = BMax/Bprime * ( 1 - xi_val**2 ) - 1

vmin = 0.0
vmax = np.log10(1.0 + 1.8e5)
levels = np.linspace(vmin, vmax, 100)
norm = Normalize(vmin=vmin, vmax=vmax)

cf = plt.contourf(R/epsilon, Z, np.log10(1+T_plot), 100, cmap='jet')
cs1tmp = ax.contour(R/epsilon, Z, TrapRegion, levels=[0], colors='white', linestyles='-', linewidths=2)

plt.colorbar(cf)
plt.xlabel('$R/a$')
plt.ylabel('$Z/a$')
plt.title('Escape Time')
plt.savefig("figures/T_Z_R.png")


r_grid_norm_res  = torch.tensor(rrNorm, device=device).reshape(-1,1)

theta_grid_norm_res  = torch.tensor(ththNorm, device=device).reshape(-1,1)
xi_grid_norm_res  = torch.tensor(xiNorm, device=device).reshape(-1,1)

res_flat = fp_pde(model, r_grid_norm_res, theta_grid_norm_res, xi_grid_norm_res)
Residual_plot = res_flat.detach().cpu().numpy().reshape(rrNorm.shape)

# R-Z plot of residual
fig, ax = plt.subplots()
fig.set_tight_layout(True)

cf_res = plt.contourf(R/epsilon, Z, Residual_plot, 100, cmap='jet')

plt.colorbar(cf_res)
plt.xlabel('$R/a$')
plt.ylabel('$Z/a$')
plt.title('PDE Residual')
plt.savefig("figures/Residual_Z_R.png")


rrNorm2, xiNorm2 = np.meshgrid(rValsNorm, xiValsNorm, indexing="ij")
ththNorm2 = theta_valNorm*np.ones(n_r_plot*n_xi_plot)

r2  = rMin + (rMax  - rMin ) * rrNorm2
xi2 = xiMin + (xiMax - xiMin) * xiNorm2

r_grid2  = torch.tensor(rrNorm2, device=device).reshape(-1,1)
theta_grid2  = torch.tensor(ththNorm2, device=device).reshape(-1,1)
xi_grid2  = torch.tensor(xiNorm2, device=device).reshape(-1,1)

model.eval()
with torch.no_grad():
    inputs = torch.cat([r_grid2, theta_grid2, xi_grid2], dim=1)
    XX = input_transform(inputs)
    TNN2 = model(XX)
    T2 = output_transform(TNN2)
    T_plot2 = T2.cpu().numpy().reshape(rrNorm2.shape)

# r-\xi plot Tesc
fig, ax = plt.subplots()
fig.set_tight_layout(True)

cs3 = ax.contourf(r2, xi2, np.log10(1+T_plot2), levels=100, cmap='jet')

xiTrap = np.sqrt(2*r2*epsilon/(1+r2*epsilon))
ax.plot(r2, xiTrap, linestyle='-',color='white',linewidth=2)
ax.plot(r2, -xiTrap, linestyle='-',color='white',linewidth=2)

fig.colorbar(cs3,ax=ax)
ax.set_ylabel("$\\xi$")
ax.set_xlabel("$r/a$")
ax.set_title("Escape Time")
plt.savefig("figures/T_xi_r.png")


r_grid_norm_res2  = torch.tensor(rrNorm2, device=device).reshape(-1,1)
theta_grid_norm_res2  = torch.tensor(ththNorm2, device=device).reshape(-1,1)
xi_grid_norm_res2  = torch.tensor(xiNorm2, device=device).reshape(-1,1)

res_flat2 = fp_pde(model, r_grid_norm_res2, theta_grid_norm_res2, xi_grid_norm_res2)
Residual_plot2 = res_flat2.detach().cpu().numpy().reshape(rrNorm2.shape)

# r-\xi plot of residual
fig, ax = plt.subplots()
fig.set_tight_layout(True)

cs4 = ax.contourf(r2, xi2, Residual_plot2, 50, cmap='jet')

fig.colorbar(cs4,ax=ax)
ax.set_ylabel("$\\xi$")
ax.set_xlabel("$r/a$")
ax.set_title("PDE residual")
plt.savefig("figures/Residual_xi_r.png")


thetaNorm3, xiNorm3 = np.meshgrid(thetaValsNorm, xiValsNorm, indexing="ij")
rNorm3 = r_valNorm*np.ones(n_theta_plot*n_xi_plot)

r3  = rMin + (rMax  - rMin ) * rNorm3
theta3 = thetaMin + ( thetaMax - thetaMin ) * thetaNorm3
xi3 = xiMin + (xiMax - xiMin) * xiNorm3

r_grid3  = torch.tensor(rNorm3, device=device).reshape(-1,1)
theta_grid3  = torch.tensor(thetaNorm3, device=device).reshape(-1,1)
xi_grid3  = torch.tensor(xiNorm3, device=device).reshape(-1,1)

model.eval()
with torch.no_grad():
    inputs = torch.cat([r_grid3, theta_grid3, xi_grid3], dim=1)
    XX = input_transform(inputs)
    TNN3 = model(XX)
    T3 = output_transform(TNN3)
    T_plot3 = T3.cpu().numpy().reshape(thetaNorm3.shape)

# \theta-\xi plot Tesc
fig, ax = plt.subplots()
fig.set_tight_layout(True)

BMax = 1 / ( 1 - r_val*epsilon )
Bprime = 1 / (1 + epsilon*r_val*np.cos(theta3) )
TrapRegion3 = BMax/Bprime * ( 1 - xi3**2 ) - 1

cs5 = ax.contourf(theta3, xi3, np.log10(1+T_plot3), levels=50, cmap='jet')

cs5tmp = ax.contour(theta3, xi3, TrapRegion3, levels=[0], colors='white', linewidths=2)

fig.colorbar(cs5,ax=ax)
ax.set_ylabel("$\\xi$")
ax.set_xlabel("$\\theta$")
ax.set_title("Escape Time")
plt.savefig("figures/T_theta_xi.png")


r_grid_norm_res3  = torch.tensor(rNorm3, device=device).reshape(-1,1)
theta_grid_norm_res3  = torch.tensor(thetaNorm3, device=device).reshape(-1,1)
xi_grid_norm_res3  = torch.tensor(xiNorm3, device=device).reshape(-1,1)

res_flat3 = fp_pde(model, r_grid_norm_res3, theta_grid_norm_res3, xi_grid_norm_res3)
Residual_plot3 = res_flat3.detach().cpu().numpy().reshape(thetaNorm3.shape)


# \theta-\xi plot of residual
fig, ax = plt.subplots()
fig.set_tight_layout(True)


cs6 = ax.contourf(theta3, xi3, Residual_plot3, levels=50, cmap='jet')
cs6tmp = ax.contour(theta3, xi3, TrapRegion3, levels=[0], colors='white', linewidths=2)

fig.colorbar(cs6,ax=ax)
ax.set_ylabel("$\\xi$")
ax.set_xlabel("$\\theta$")
ax.set_title("PDE residual")
plt.savefig("figures/Residual_theta_xi.png")

# Loss history plot
fig, ax = plt.subplots()
fig.set_tight_layout(True)

StepsTraining     = np.asarray(StepsTraining)
TrainingLossPDE   = np.asarray(TrainingLossPDE)
TrainingLossBC    = np.asarray(TrainingLossBC)

epochs_int = np.rint(StepsTraining).astype(np.int64)  # robust to float formatting like 1.000e+02
maskEpoch = (epochs_int % 100 == 0)

StepsReduced = StepsTraining[maskEpoch]
TrainingLossPDEReduced = TrainingLossPDE[maskEpoch]
TrainingLossBCReduced = TrainingLossBC[maskEpoch]

ax.plot(StepsReduced, TrainingLossPDEReduced, label='Train PDE', linestyle='-',color='blue',linewidth=2)
ax.plot(StepsReduced, TrainingLossBCReduced, label='Train BC', linestyle='-',color='red',linewidth=2)
ax.plot(epochGlobalList, TestPDE, 'xb', label='Test PDE')
ax.plot(epochGlobalList, TestBC, 'xr', label='Test BC')

plt.yscale('log')
plt.xlabel('Epochs')
plt.title('Losses')
plt.legend(fontsize=12)
plt.savefig("figures/LossHistory.png")

# residual magnitude loss of test points
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.set_tight_layout(True)

res_flat3 = fp_pde(model, rNorm_test, thetaNorm_test, xiNorm_test)
res_plot3 = res_flat3.abs().detach().cpu().numpy()

SizeDist = 0.2*res_plot3

rNorm_test = rNorm_test.detach().cpu().numpy()
thetaNorm_test = thetaNorm_test.detach().cpu().numpy()
xiNorm_test = xiNorm_test.detach().cpu().numpy()

rTest  = rMin + (rMax  - rMin ) * rNorm_test
thetaTest = thetaMin + ( thetaMax - thetaMin ) * thetaNorm_test
xiTest = xiMin + (xiMax - xiMin) * xiNorm_test

idx = np.argmax(res_plot3)
print(f"Max error = {res_plot3[idx]} at r={rTest[idx]}, theta={thetaTest[idx]}, xi={xiTest[idx]}")

ax[0,0].scatter(rTest, thetaTest, s=SizeDist, color='black')
ax[0,0].set_ylabel("$\\theta$")
ax[0,0].set_xlabel("$r/a$")
ax[0,0].set_title("Test Points")

ax[0,1].scatter(rTest, xiTest, s=SizeDist, color='black')
ax[0,1].set_ylabel("$\\xi$")
ax[0,1].set_xlabel("$r/a$")
ax[0,1].set_title("Test Points")

ax[1,0].scatter(thetaTest, xiTest, s=SizeDist, color='black')
ax[1,0].set_ylabel("$\\xi$")
ax[1,0].set_xlabel("$\\theta$")
ax[1,0].set_title("Test Points")

ax[1,1].scatter(rTest, res_plot3, s=SizeDist, color='black')
ax[1,1].set_ylabel("Magnitude")
ax[1,1].set_xlabel("$r/a$")
ax[1,1].set_title("Test Points")
ax[1,1].set_yscale('log')

plt.savefig("figures/TestDist.png")

# residual magnitude loss of training points
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.set_tight_layout(True)

res_flat3 = fp_pde(model, rNorm_train, thetaNorm_train, xiNorm_train)
res_plot3 = res_flat3.abs().detach().cpu().numpy()

SizeDist = 0.1*res_plot3

rNorm_train = rNorm_train.detach().cpu().numpy()
thetaNorm_train = thetaNorm_train.detach().cpu().numpy()
xiNorm_train = xiNorm_train.detach().cpu().numpy()

rTrain  = rMin + (rMax  - rMin ) * rNorm_train
thetaTrain = thetaMin + ( thetaMax - thetaMin ) * thetaNorm_train
xiTrain = xiMin + (xiMax - xiMin) * xiNorm_train

ax[0,0].scatter(rTrain, thetaTrain, s=SizeDist, color='black')
ax[0,0].set_ylabel("$\\theta$")
ax[0,0].set_xlabel("$r/a$")
ax[0,0].set_title("Training Points")

ax[0,1].scatter(rTrain, xiTrain, s=SizeDist, color='black')
ax[0,1].set_ylabel("$\\xi$")
ax[0,1].set_xlabel("$r/a$")
ax[0,1].set_title("Training Points")

ax[1,0].scatter(thetaTrain, xiTrain, s=SizeDist, color='black')
ax[1,0].set_ylabel("$\\xi$")
ax[1,0].set_xlabel("$\\theta$")
ax[1,0].set_title("Training Points")

ax[1,1].scatter(rTrain, res_plot3, s=SizeDist, color='black')
ax[1,1].set_ylabel("Magnitude")
ax[1,1].set_xlabel("$r/a$")
ax[1,1].set_title("Training Points")
ax[1,1].set_yscale('log')

plt.savefig("figures/TrainDist.png")
