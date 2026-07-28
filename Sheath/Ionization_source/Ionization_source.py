import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_optimizer.optimizer.soap import SOAP
from torch.quasirandom import SobolEngine
from NN_modules import *

save_path = '/path/to/saved/model/and/to/save/figures/'

#Set the default data type and random seed for repeatability
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)

#Use CUDA enabled GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({'font.size': 16})


L = 50 # Size of domain in Debye lengths
Z = 1 # Ion charge
xMax = L
xMin = 0
S0Min =   1e28
S0Max =   1e29 #1e31
TeMin = 1
TeMax = 20
TiTeMin = 0 #normalized Ti/Te ratio
TiTeMax = 1
nnMin = 0.0 #neutral density
nnMax = 5.0
MiMe = 1836*40 #Mass ratio to train for (Argon in this case)
#niMax = 25

#network params
epochsADAM = 500
epochsBFGS = 40000
Num_SSB = 5
lr = 1.e-3
pts = 2000000
bdypts = 100000
sheath_pts = 200000
corner_points = 200000
test_every = 1000

#mass electron
me = 9.109e-31 #kg

#charge of an electron
q = 1.602e-19 #C

#permittivity of free space
Eps0 = 8.85e-12 #F/m

#ion mass
mi = MiMe*me#1.6735575e-27 #kg

#T and Ez in eV
def Sion(T,Ez): 
    #Fit to ionization rate from NRL Formulary
    #This version is for torch tensors
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * torch.exp(-Ez/T) #m^3/s

#T and Ez in eV
def Srec(T,Ez,Z):
    #Fit to recombination rate from NRL Formulary
    #This version is for torch tensors
    return 5.2e-20*Z*(Ez/T)**(1/2)*(0.43 + 0.5*torch.log(Ez/T) + 0.469*(Ez/T)**(-1/3)) #m^3/s

def Sion2(T,Ez): 
    #Same as above, but for numpy arrays or floats or integers
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * np.exp(-Ez/T) 

def Srec2(T,Ez,Z):
    #Same as above, but for numpy arrays or floats or integers
    return 5.2e-20*Z*(Ez/T)**(1/2)*(0.43 + 0.5*np.log(Ez/T) + 0.469*(Ez/T)**(-1/3))

def signn(E,xr,nr):
    #Fit to the ion -> neutral collision rate derived from the Phelps LXCat cross-section library
    E = E + 1e-4
    return (2e-19/((2*E)**0.5*(1+2*E)) + 3e-19*2*E/(1+2/3*E)**2.3) * xr*nr

def feature_transform(inputs):
    #transform x -> x^2
    #must return all inputs, even though they aren't modified
    xNorm,TeNorm,TiTeNorm,S0Norm,nnNorm = inputs.split(1,dim=1)
    xSq = xNorm * xNorm

    return torch.cat((xSq,TeNorm,TiTeNorm,S0Norm,nnNorm),dim=1)

def output_transform(inputs, outputs):
    #Output layer to enforce BCs
    xNorm,TeNorm,TiTeNorm = inputs[:,0:1],inputs[:,1:2],inputs[:,2:3]
    phi, ni, ue, nw = outputs.split(1,dim=1)
    
    x = xMin + ( xMax - xMin ) * xNorm
    Te = TeMin + (TeMax - TeMin ) * TeNorm
    TiTe = TiTeMin + (TiTeMax - TiTeMin ) *TiTeNorm

    #electron velocity at wall from random velocity
    uewall = np.sqrt(MiMe/(2*np.pi)) 

    #Phi is 0 at x=L and x=-L
    phiFinal = (L-x)*(L+x)/L**2 * phi 
    #ion density is always positive and can't go to exactly 0
    niFinal = F.softplus(ni) + 1e-8
    #Electron velocity is 0 at x=0 and uewall at x=L and -uewall at x=-L
    ueFinal = (L-x)*(L+x)/L**2*(x/L) * ue + ((1-torch.tanh(5-x/(0.2*L)))+(torch.tanh(5+x/(0.2*L))-1))*np.log(uewall+1)
    #electron wal density is always positive 
    nwFinal = F.softplus(nw) 

    return torch.cat((phiFinal,niFinal,ueFinal,nwFinal), dim=1)

#physics loss terms
def pde(model,inputs):
    inputs = inputs.requires_grad_(True)
    inputs_trans = feature_transform(inputs)
    xNorm, TeNorm, TiTeNorm, S0Norm, nnNorm = inputs.split(1,dim=1)

    #Right boundary x=L
    X = torch.cat((xNorm*0 +1.0, TeNorm, TiTeNorm, S0Norm, nnNorm),dim=1)
    X2 = feature_transform(X)

    #Scale inputs back to normalized values
    x = xMin + ( xMax - xMin ) * xNorm
    Te = TeMin + (TeMax - TeMin ) * TeNorm
    TiTe = TiTeMin + (TiTeMax - TiTeMin ) * TiTeNorm
    S0 = S0Min + (S0Max - S0Min ) * S0Norm
    nn = nnMin + (nnMax - nnMin) * nnNorm
    un = 0.0
    Ti = Te * TiTe

    outputs = model(inputs_trans)
    outputs_trans = output_transform(inputs, outputs)
    phi, ni, uet,_ = outputs_trans.split(1,dim=1)

    outputs2 = model(X2)
    outputs2_trans = output_transform(X,outputs2)
    _,_,_,nw = outputs2_trans.split(1,dim=1)

    #Compute electron density assuming Boltzmann electrons
    ne = nw*torch.exp(phi)
    #coordinate transforms
    ue = torch.expm1(uet)   # = exp(uet)-1, more stable for small uet

    ui = ne*ue/ni

    #normalizations
    uref = torch.sqrt((Te)*q/(mi))
    lref = (Eps0*Te*uref/(S0*L*q))**(1/3)
    nref = S0*L*lref/uref
    #H-13.6 | Ar-15.8
    #Compute ionization rate
    barSion = Sion(Te,15.8)*(lref*nref/uref)

    #Autograd for computing derivatives
    dphi_x = torch.autograd.grad(phi, inputs, create_graph=True, grad_outputs=torch.ones_like(phi))[0][:,0:1] / (xMax - xMin)
    dphi_xx = torch.autograd.grad(dphi_x, inputs, create_graph=True, grad_outputs=torch.ones_like(dphi_x))[0][:,0:1] / (xMax - xMin)
    dne_x = torch.autograd.grad(ne, inputs, create_graph=True, grad_outputs=torch.ones_like(ne))[0][:,0:1] / (xMax - xMin)
    due_x = torch.autograd.grad(ue, inputs, create_graph=True, grad_outputs=torch.ones_like(ue))[0][:,0:1] / (xMax - xMin)
    dni_x = torch.autograd.grad(ni, inputs, create_graph=True, grad_outputs=torch.ones_like(ni))[0][:,0:1] / (xMax - xMin)
    dui_x = torch.autograd.grad(ui, inputs, create_graph=True, grad_outputs=torch.ones_like(ui))[0][:,0:1] / (xMax - xMin)

    #PDEs
    lossb1 = dphi_xx - (ne - ni) # Poisson
    lossb2 = L*ne*due_x + L*ue*dne_x - 1 - L*nn*ne*barSion # electron continuity
    lossb5 = TiTe*dni_x + ni*dphi_x + ui*(1/L + nn*ne*barSion) + ni*ui*dui_x + nn*ni*1*ui*signn(0.5*(Te+TiTe*Te),lref,nref) #C_s/uref=1 #ion momentum
    #Cs appears in the ion -> neutral collision term since the cross section is evaluated at the sound speed, but when normalized, becomes 1

    return lossb1,lossb2,lossb5

#Build network architecture
model = FNN(in_dim=5, layers=[32]*5, out_dim=4).to(device)

def main():
    #Training points drawn from sobol distribution
    engine = SobolEngine(dimension=5, scramble=True, seed=1234)
    engine2 = SobolEngine(dimension=4, scramble=True, seed=5678)
    #Ensure training points on the spatial boundaries
    #This isn't actually needed, since none of the BCs need to be trained (they're all enforced through the output transform)
    #But, if you wanted to move a BC to the loss function, you would need points on the boundaries
    X_bound = engine2.draw(bdypts).to(device)
    X_rbound = torch.cat((X_bound[:,0:1]*0.0+1.0, X_bound),dim=1)
    X_lbound = torch.cat((X_bound[:,0:1]*0.0, X_bound),dim=1)
    #Generate a point grid for the entire domain
    X_train = engine.draw(pts).to(device)
    #Add the boundary points to the larger domain point distribution
    X_train = torch.cat((X_train, X_rbound, X_lbound),dim=0)

    #Generate a test distribution of points
    engine3 = SobolEngine(dimension=5, scramble=True, seed=4321)
    X_test = engine3.draw(100000).to(device)

    #Use SOAP for balancing and settling the loss function into the minima
    optimizer = SOAP(model.parameters(), lr=lr, betas=(.999, .999), weight_decay=0e-2, precondition_frequency=1)
    #Schedule the learning rate to reduce and make the optimizer take smaller steps over time
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99975)

    #Call the function to train the network and track the loss history
    losses_Soap, test_Soap = TrainSOAPorADAM(NumEpochs=epochsADAM,
                                            optimizer=optimizer,
                                            model=model,
                                            loss_fn=pde,
                                            lr=scheduler,
                                            X_train=X_train,
                                            X_test=X_test,
                                            test_every=test_every,
                                            save_every=1000,
                                            SavePath=save_path)

    loss_hist = losses_Soap
    test_hist = test_Soap

    #Use second order optimizer SSBroyden next to drill down and converge the loss
    for i in range(Num_SSB):
        losses_Scipy, test_loss = TrainScipy(model=model,
                                            loss_fn=pde,
                                            method="BFGS",
                                            X_train=X_train,
                                            X_test=X_test,
                                            test_every=test_every,
                                            epochs=len(loss_hist[0]),
                                            maxiter=epochsBFGS,
                                            save_every=500,
                                            SavePath=save_path)
        for j in range(len(loss_hist)):
            loss_hist[j] = np.append(loss_hist[j], losses_Scipy[j])
            test_hist[j] = np.append(test_hist[j], test_loss[j])
        np.savetxt(save_path + f"loss_history.txt", np.array(loss_hist).T)
        np.savetxt(save_path + f"test_history.txt", np.array(test_hist).T)

    #Save the model
    torch.save(model.state_dict(), save_path + f"model_final.pt")
if __name__ == "__main__":
    main()
