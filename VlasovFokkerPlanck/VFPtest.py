import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
import pytorch_optimizer
from torch.optim import lr_scheduler
import skopt
import matplotlib.pyplot as plt
import os
import time

# Set seeds and precision
torch.manual_seed(1234)
np.random.seed(1234)
torch.set_default_dtype(torch.float64)

# ------------------------
# Physics Constants
# ------------------------
clight = 2.99792e10      # speed of light in cm/s
CLASSICALER = 2.8179e-13 # classical electron radius in units of cm
mecSQ = 511e3            # electron rest mass in units eV
IA = 0.017045e6          # Alfven current in Amperes

# Domain definitions
EMax = 15.0
EMin = 0.01
xiMax = 1
xiMin = -1
xMax = 1
xMin = -1
tMax = 1
tMin = 0

# Physics parameters
KnMax = 0.1
Zeff = 1
n1 = 9
A = 0.5
xleft = -0.5
xright = 0.5
Dx = 0.1
ma = 3   # normalized mass (particle)
ms = 2.5 # normalized mass (species collided with)
md = 2   # deuterium mass
mt = 3   # tritium mass

# Training Hyperparameters
epochsSOAP = 50000
epochsBFGS = 10000 # Max iter for SSBroyden/L-BFGS
lr = 1.e-3
pts = 100000
batch_size = pts # Full batch
InputDim = 5     # Feature transform results in 5 inputs: E, xi^2, x^2, x*xi, t

E_val = 8
xi_val =0.8
x_val = 0
t_val = 0.5

# --------------------------------
# PINN Architecture
# --------------------------------
class PINN(nn.Module):
    def __init__(self, in_dim=InputDim, h_dim=32, out_dim=1):
        super().__init__()
        # Feature transform changes input dim to 5
        self.linear1 = nn.Linear(in_dim, h_dim); self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(h_dim, h_dim);  self.act2 = nn.Tanh()
        self.linear3 = nn.Linear(h_dim, h_dim);  self.act3 = nn.Tanh()
        self.linear4 = nn.Linear(h_dim, h_dim);  self.act4 = nn.Tanh()
        self.linearLast = nn.Linear(h_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in [self.linear1, self.linear2, self.linear3, self.linear4, self.linearLast]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        x = self.act3(self.linear3(x))
        x = self.act4(self.linear4(x))
        x = self.linearLast(x)
        return x

# -----------------------------------------
# Transforms
# -----------------------------------------
def feature_transform(Energy, xi, x, t):
    # Note: Inputs here are normalized [0,1] or physical depending on usage.
    xiSq = xi * xi
    xSq = x * x
    xxi = x * xi
    return torch.cat((Energy, xiSq, xSq, xxi, t), axis=1)


def output_transform(outputs, Energy, xi, x, t):
    # Enforces the Hard Constraints (Boundary/Initial Conditions).

    # Physics profiles derived from coordinates
    ne = 1 + 0.5 * n1 * (2 + torch.tanh((xleft - x) / Dx) + torch.tanh((x - xright) / Dx))
    nd = (1 - A) * ne
    nt = A * ne
    Te = 1 / ne
    Ebar = Energy / Te
    DE = 0.005 * EMax
    Tmin = 1 / (1 + n1)
    
    # Base distribution fa0
    # The original code calculates fa based on network output 'outputs'
    
    # Original TF logic:
    # fa = 1/(...) * exp( -Ebar - DE**2 / (...) * ... + (complex boundary term) * NN_output )
    
    term1 = -Ebar
    term2 = - (DE**2 / ((EMax - Energy)**2 + DE**2)) * Energy * (Te - Tmin) / Tmin / Te
    
    # The boundary enforcing term that multiplies the Neural Network output
    boundary_scaling = (Energy - EMin) * (EMax - Energy) / (EMax - EMin)**2 * \
                       (x - xMin) * (xMax - x) / (xMax - xMin)**2
    
    exponent = term1 + term2 + boundary_scaling * outputs
    
    prefactor = 1 / (2 * np.pi)**1.5 * nt / Te**1.5
    fa = prefactor * torch.exp(exponent)
    
    return fa

# -----------------------------------------
# PDE Definition
# -----------------------------------------
def fp_pde(model, EnergyNorm, xiNorm, xNorm, tNorm):
    # Enable gradients for inputs
    EnergyNorm = EnergyNorm.clone().detach().requires_grad_(True)
    xiNorm     = xiNorm.clone().detach().requires_grad_(True)
    xNorm      = xNorm.clone().detach().requires_grad_(True)
    tNorm      = tNorm.clone().detach().requires_grad_(True)

    # Scale to physical units
    Energy = EMin + (EMax - EMin) * EnergyNorm
    xi     = xiMin + (xiMax - xiMin) * xiNorm
    x      = xMin + (xMax - xMin) * xNorm
    t      = tMin + (tMax - tMin) * tNorm

    # Prepare input for network (Feature Transform)
    # The network takes normalized inputs transformed into features
    network_input = feature_transform(EnergyNorm, xiNorm, xNorm, tNorm)
    
    # Get Network Output
    nn_output = model(network_input)
    
    # Apply Output Transform (Hard Constraints) to get physical distribution 'fa'
    fa = output_transform(nn_output, Energy, xi, x, t)

    # -----------------------------------------
    # Compute Gradients using Autograd
    # Note: We must divide by the scaling factor (Max-Min) because
    # d/dx = d/dxNorm * d(xNorm)/dx = d/dxNorm * (1/(Max-Min))
    # -----------------------------------------
    
    # First Derivatives
    grads = torch.autograd.grad(fa, (EnergyNorm, xiNorm, xNorm, tNorm), 
                                grad_outputs=torch.ones_like(fa), create_graph=True)
    dy_ENorm, dy_xiNorm, dy_xNorm, dy_tNorm = grads

    dy_E  = dy_ENorm  / (EMax - EMin)
    dy_xi = dy_xiNorm / (xiMax - xiMin)
    dy_x  = dy_xNorm  / (xMax - xMin)
    dy_t  = dy_tNorm  / (tMax - tMin)

    # Second Derivatives (Hessians)
    dy_EE_Norm  = torch.autograd.grad(dy_ENorm, EnergyNorm, grad_outputs=torch.ones_like(fa), create_graph=True)[0]
    dy_xixi_Norm = torch.autograd.grad(dy_xiNorm, xiNorm, grad_outputs=torch.ones_like(fa), create_graph=True)[0]

    dy_EE   = dy_EE_Norm   / (EMax - EMin)**2
    dy_xixi = dy_xixi_Norm / (xiMax - xiMin)**2

    # -----------------------------------------
    # Physics Equation Terms
    # -----------------------------------------
    # Re-calculate plasma profiles for coefficients
    ne = 1 + 0.5 * n1 * (2 + torch.tanh((xleft - x) / Dx) + torch.tanh((x - xright) / Dx))
    nd = (1 - A) * ne
    nt = A * ne
    Te = 1 / ne
    
    v = torch.sqrt(Energy)

    Kn = KnMax * Te*Te / ne
    nu_E = 2.0 / Kn * torch.sqrt(Te) * (nd/ne * ma/md + nt/ne * ma/mt)
    nu_xi = 0.25 * nu_E * md/ma * (1 + nt/nd) / (1 + nt/nd * md/mt)

    Vx = xi * Energy
    VE = nu_E * Te**1.5
    dVE_E = 0.0
    Dxi = nu_xi * Te**1.5 / Energy * (1.0 - xi*xi)
    Dxi_xi = -2.0 * nu_xi * Te**1.5 / Energy * xi
    DE = nu_E * Te**2.5
    DE_E = 0.0

    # Normalization terms for Loss
    Ebar = Energy / Te
    fa0 = 1 / (2 * np.pi)**1.5 * nt / Te**1.5
    
    DeltaE = 0.05 * EMax
    ECut = 0.8 * EMax
    GE = 0.5 * (1 - torch.tanh((Energy - ECut) / DeltaE))

    # The Residual Equation
    # loss = GE / (fa0 + ...) * sqrt(...) * ( PDE_Terms )
    
    pde_lhs = torch.sqrt(Energy) * dy_t + Vx * dy_x
    pde_rhs = VE * dy_E + dVE_E * fa + Dxi * dy_xixi + Dxi_xi * dy_xi + DE * dy_EE + DE_E * dy_E
    
    residual_core = pde_lhs - pde_rhs
    
    # Scaling factor from original code
    scaling = GE / (fa0 + 1e-2 * fa) * torch.sqrt(Energy / (1.0 + Energy))
    
    loss = scaling * residual_core

    return loss


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


def lbfgs_loss_and_grad(flat_params, model, loss_fn, E_train, xi_train, x_train, t_train):
   set_flat_params(model, flat_params)
   model.zero_grad()
   loss = loss_fn(model, E_train, xi_train, x_train, t_train)
   loss.backward()
   grads = []
   for param in model.parameters():
       grads.append(param.grad.detach().cpu().numpy().reshape(-1))
   flat_grad = np.concatenate(grads)
   return loss.item(), flat_grad


def pinn_loss(model, E_train, xi_train, x_train, t_train):
    res = fp_pde(model, E_train, xi_train, x_train, t_train)
    loss_pde = torch.mean(res**2)
    #TrainingLossPDE.append(loss_pde.detach().cpu().numpy())
    #EpochLocal = StepsTraining[-1]+1
    #StepsTraining.append(EpochLocal)
    #if EpochLocal % 500 == 0: # save model occasionally
    #    torch.save(model.state_dict(), f"model/pinn_weights_epoch_{EpochLocal:05d}.pth")

    return loss_pde


def fit_lbfgsb(model, loss_fn, E_train, xi_train, x_train, t_train, maxiter, print_every):
   flat_params_init = get_flat_params(model)
   nfeval = [1]
   def callback(params):
       if nfeval[0] % print_every == 0:
           set_flat_params(model, params)
           loss = loss_fn(model, E_train, xi_train, x_train, t_train)
           #test_PDE = compute_test_loss()
           #PrevGlobalEpoch = epochGlobalList[-1] if epochGlobalList else 0
           #TestPDE.append(test_PDE)
           #epochGlobalList.append(PrevGlobalEpoch+print_every)
           print(f"Epoch {nfeval[0]}: Training Loss = {loss.item():.4e}")
       nfeval[0] += 1
   res = minimize(
       lbfgs_loss_and_grad,
       flat_params_init,
       args=(model, loss_fn, E_train, xi_train, x_train, t_train),
       method='BFGS',
       jac=True,
       callback=callback,
       options={'maxiter': maxiter, 'disp': None}
   )
   set_flat_params(model, res.x)
   return res



# -----------------------------------------
# Utilities
# -----------------------------------------
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

def hammersley_sequence(n_samples, dim):
    sampler = skopt.sampler.Hammersly()
    # Hammersley gen returns [0,1]
    return np.asarray(sampler.generate([(0.0, 1.0)]*dim, n_samples), dtype=np.float64)

# -----------------------------------------
# Training Logic
# -----------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Model
    model = PINN().to(device)
    
    # Generate Data (Hammersley Sampling)
    # Normalized between 0 and 1
    # 4 dimensions: Energy, xi, x, t
    print(f"Generating {pts} Hammersley points...")
    raw_points = hammersley_sequence(pts, 4)
    X_train = torch.tensor(raw_points, dtype=torch.float64, device=device)
    
    # Split columns
    E_train  = X_train[:, 0:1]
    xi_train = X_train[:, 1:2]
    x_train  = X_train[:, 2:3]
    t_train  = X_train[:, 3:4]
    
    # Loss lists
    loss_history = []
    TrainingLossPDE = []
    StepsTraining = []

    # -----------------------------------------
    # Phase 1: SOAP Optimizer
    # -----------------------------------------
    print("Starting SOAP training...")
    optimizer = pytorch_optimizer.optimizer.soap.SOAP(
        model.parameters(), 
        lr=lr, 
        betas=(.999, .999), 
        weight_decay=0e-2, 
        precondition_frequency=1
    )
    # Using ExponentialLR as in EscapeTime.py logic
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999995) 

    model.train()
    st_time = time.time()

    for epoch in range(epochsSOAP):
        optimizer.zero_grad()
        
        # Calculate Residual
        loss = pinn_loss(model, E_train, xi_train, x_train, t_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch % 100 == 0:
            print(f"SOAP Epoch {epoch}/{epochsSOAP}: Loss = {loss_val:.4e}")

    print(f"SOAP training finished. Time: {time.time()-st_time:.2f}s")

    # -----------------------
    # Phase 2: SSBroyden
    # -----------------------
    print("Starting SSBroyden training...")
    # SSBroyden is a memory-optimized BFGS variant. L-BFGS-B is the closest standard SciPy equivalent.
    fit_lbfgsb(model, pinn_loss, E_train, xi_train, x_train, t_train, epochsBFGS, 100)
    
    print("L-BFGS-B training finished.")

    # -----------------------------------------
    # Saving
    # -----------------------------------------
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model_pytorch.ckpt')
    np.savetxt('./model/loss_history.txt', np.array(loss_history))

    n_E_plot = 60
    n_xi_plot = 70
    n_x_plot = 80
    n_t_plot = 90
    EValsNorm = np.linspace(0, 1, n_E_plot, dtype=np.float64)
    xiValsNorm = np.linspace(0, 1, n_xi_plot, dtype=np.float64)
    xValsNorm = np.linspace(0, 1, n_x_plot, dtype=np.float64)
    tValsNorm = np.linspace(0, 1, n_t_plot, dtype=np.float64)

    xiNorm = xi_valNorm*np.ones(n_E_plot*n_xi_plot)
    tNorm = t_valNorm*np.ones(n_E_plot*n_xi_plot)
    EENorm, xixiNorm = np.meshgrid(EValsNorm, xiValsNorm, indexing="ij")

    
    # Loss history
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss History')
    plt.savefig('./model/loss.png')
    print("Model and plots saved.")

if __name__ == "__main__":
    main()
