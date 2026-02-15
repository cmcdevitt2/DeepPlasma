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
import argparse
from scipy.integrate import simps

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

# Plotting slice defaults
E_val = 8
xi_val =0.8
x_val = 0
t_val = 0.5
# Normalized equivalents for plotting
E_valNorm = (E_val - EMin) / (EMax - EMin)
xi_valNorm = (xi_val - xiMin) / (xiMax - xiMin)
x_valNorm = (x_val - xMin) / (xMax - xMin)
t_valNorm = (t_val - tMin) / (tMax - tMin)

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

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='PINN Training and Plotting')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--plot', action='store_true', help='Plot results from existing model')
    args = parser.parse_args()

    if not args.train and not args.plot:
        print("Please specify either --train or --plot")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Model
    model = PINN().to(device)
    model_path = './model/model_pytorch.ckpt'

    # --------------------------------------
    # TRAINING
    # --------------------------------------
    if args.train:
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

        # Phase 1: SOAP Optimizer
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

        # Phase 2: SSBroyden
        print("Starting SSBroyden training...")
        # SSBroyden is a memory-optimized BFGS variant. L-BFGS-B is the closest standard SciPy equivalent.
        fit_lbfgsb(model, pinn_loss, E_train, xi_train, x_train, t_train, epochsBFGS, 100)
    
        print("L-BFGS-B training finished.")

        # Saving
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), model_path)
        np.savetxt('./model/loss_history.txt', np.array(loss_history))
    
        # Loss history
        plt.figure()
        plt.plot(loss_history)
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss History')
        plt.savefig('./model/loss.png')
        print("Model and plots saved.")


    # -----------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------
    if args.plot:
        print("Loading model for plotting...")
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Run --train first.")
            return
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        os.makedirs('./figures', exist_ok=True)
        plt.rcParams.update({'font.size': 18})

        # Grids definition matches PlotFPHotSpot.py logic
        numE = 200
        numxi = 150
        numx = 100
        
        Energygrid = np.logspace(np.log10(EMin), np.log10(EMax), numE)
        xigrid = np.linspace(xiMin, xiMax, numxi)
        xgrid = np.linspace(xMin, xMax, numx)

        # ------------------------------------------------------------------
        # Plot 1: f(x, xi) at constant Energy and Time
        # ------------------------------------------------------------------
        print("Generating f(x, xi) contour...")
        xinew, xnew = np.meshgrid(xigrid, xgrid) # shape (numx, numxi)
        
        # Flatten for prediction
        flat_xi = xinew.ravel()
        flat_x = xnew.ravel()
        
        # Normalize inputs
        xi_in = (torch.tensor(flat_xi, device=device).unsqueeze(1) - xiMin) / (xiMax - xiMin)
        x_in  = (torch.tensor(flat_x, device=device).unsqueeze(1) - xMin) / (xMax - xMin)
        
        # Constant inputs
        E_in = torch.ones_like(xi_in) * E_valNorm
        t_in = torch.ones_like(xi_in) * t_valNorm
        
        with torch.no_grad():
            # Features
            features = feature_transform(E_in, xi_in, x_in, t_in)
            nn_out = model(features)
            
            # Physical coordinates for output transform
            E_phys = torch.tensor(E_val, device=device).expand_as(E_in)
            xi_phys = torch.tensor(flat_xi, device=device).unsqueeze(1)
            x_phys = torch.tensor(flat_x, device=device).unsqueeze(1)
            t_phys = torch.tensor(t_val, device=device).expand_as(E_in)
            
            fa_pred = output_transform(nn_out, E_phys, xi_phys, x_phys, t_phys)
            
        fxVsxinew = fa_pred.cpu().numpy().reshape(xinew.shape)

        # Calculate Residual for this slice
        # requires grad
        with torch.enable_grad():
            res_tens, scaling = fp_pde(model, E_in, xi_in, x_in, t_in)
        # Note: fp_pde returns (scaling * residual). DeepXDE code often plots just residual?
        # PlotFPHotSpot plots "residual of fa" but uses model.predict(operator=pde) which returns the loss form usually.
        # We will plot the scaled residual (the actual loss contribution).
        resnew = (scaling * res_tens).detach().cpu().numpy().reshape(xinew.shape)

        # FIG 1: f_a(x, xi)
        fig1, ax1 = plt.subplots(num=1, clear=True)
        fig1.set_tight_layout(True)
        cs1 = ax1.contourf(xgrid, xigrid, fxVsxinew.T, levels=50, cmap='jet')
        fig1.colorbar(cs1, ax=ax1)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$\\xi$")
        ax1.set_title(f"$f_a$ (E={E_val}, t={t_val})")
        fig1.savefig("./figures/fa_x_xi.png")

        # FIG 2: Residual(x, xi)
        fig2, ax2 = plt.subplots(num=2, clear=True)
        fig2.set_tight_layout(True)
        cs2 = ax2.contourf(xgrid, xigrid, resnew.T, 50, cmap='jet')
        fig2.colorbar(cs2, ax=ax2)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$\\xi$")
        ax2.set_title("Residual")
        fig2.savefig("./figures/residual_x_xi.png")

        # ------------------------------------------------------------------
        # Plot 2: f(E, xi) at constant x and Time
        # ------------------------------------------------------------------
        print("Generating f(E, xi) contour...")
        Enew, xinew = np.meshgrid(Energygrid, xigrid) # shape (numxi, numE)
        
        flat_E = Enew.ravel()
        flat_xi = xinew.ravel()
        
        E_in = (torch.tensor(flat_E, device=device).unsqueeze(1) - EMin) / (EMax - EMin)
        xi_in = (torch.tensor(flat_xi, device=device).unsqueeze(1) - xiMin) / (xiMax - xiMin)
        x_in = torch.ones_like(E_in) * x_valNorm
        t_in = torch.ones_like(E_in) * t_valNorm
        
        with torch.no_grad():
            features = feature_transform(E_in, xi_in, x_in, t_in)
            nn_out = model(features)
            
            E_phys = torch.tensor(flat_E, device=device).unsqueeze(1)
            xi_phys = torch.tensor(flat_xi, device=device).unsqueeze(1)
            x_phys = torch.tensor(x_val, device=device).expand_as(E_in)
            t_phys = torch.tensor(t_val, device=device).expand_as(E_in)
            
            fa_pred = output_transform(nn_out, E_phys, xi_phys, x_phys, t_phys)
            
        fEVsxinew = fa_pred.cpu().numpy().reshape(Enew.shape)
        
        # Calculate Maxwellian baseline for normalization (fMax)
        # Replicating logic from PlotFPHotSpot:
        # ne, nd, nt, Tprof calculated at x_val
        ne_val = 1 + 0.5 * n1 * (2 + np.tanh((xleft - x_val) / Dx) + np.tanh((x_val - xright) / Dx))
        nt_val = A * ne_val
        Te_val = 1 / ne_val
        
        fMaxnew = nt_val / Te_val**1.5 * np.exp(-Enew / Te_val)
        
        # Residual for this slice
        with torch.enable_grad():
            res_tens, scaling = fp_pde(model, E_in, xi_in, x_in, t_in)
        resExinew = (scaling * res_tens).detach().cpu().numpy().reshape(Enew.shape)

        # FIG 3: fa / fMax
        fig3, ax3 = plt.subplots(num=3, clear=True)
        fig3.set_tight_layout(True)
        # PlotFPHotSpot plots xinew.T vs Enew.T. Enew shape is (numxi, numE).
        # We plot xi on x-axis, E on y-axis? 
        # PlotFPHotSpot: ax3.set_xlabel("$\\xi$"), ax3.set_ylabel("$E/T_0$")
        # And does contourf(xinew.T, Enew.T, ...). 
        cs3 = ax3.contourf(xinew.T, Enew.T, fEVsxinew.T / fMaxnew.T, levels=50, cmap='jet')
        fig3.colorbar(cs3, ax=ax3)
        ax3.set_xlabel("$\\xi$")
        ax3.set_ylabel("$E/T_0$")
        ax3.set_title("$f_a/f^{Max}_a$")
        fig3.savefig("./figures/ratio_xi_E.png")

        # FIG 4: Residual(E, xi)
        fig4, ax4 = plt.subplots(num=4, clear=True)
        fig4.set_tight_layout(True)
        cs4 = ax4.contourf(xinew.T, Enew.T, resExinew.T, 50, cmap='jet')
        fig4.colorbar(cs4, ax=ax4)
        ax4.set_xlabel("$\\xi$")
        ax4.set_ylabel("$E/T_0$")
        ax4.set_title("Residual (RPF)")
        fig4.savefig("./figures/residual_xi_E.png")

        # ------------------------------------------------------------------
        # Plot 3: Average Energy Distribution <fa> vs E
        # ------------------------------------------------------------------
        print("Generating Average Energy Distribution...")
        # Integrate fEVsxinew over xi (axis 0 of Enew/fEVsxinew which corresponds to xigrid)
        # fEVsxinew shape is (numxi, numE)
        # We integrate along axis 0 (xi)
        fEnew = 0.5 * simps(fEVsxinew, xigrid, axis=0) # 0.5 factor from PlotFPHotSpot
        
        # Max dist at this location (independent of xi, so just take one slice)
        fMax_1d = fMaxnew[0, :] 

        # FIG 5: <fa> vs E
        fig5, ax5 = plt.subplots(num=5, clear=True)
        fig5.set_tight_layout(True)
        ax5.plot(Energygrid, fEnew, label='<$f_a$>', linestyle='-', color='blue', linewidth=2)
        ax5.plot(Energygrid, fMax_1d, label='$f^{Max}_a$', linestyle='--', color='blue', linewidth=2)
        ax5.set_xlabel("$E/T_0$")
        ax5.set_title("Average Energy Dist.")
        ax5.set_yscale("log")
        ax5.legend()
        fig5.savefig("./figures/avg_energy_dist.png")

        print("All plots saved to ./figures/")
        plt.show()






















    
if __name__ == "__main__":
    main()
