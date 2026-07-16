import deepxde as dde
import numpy as np
import torch
from scipy import special

import time
import sys

save_path = '/path/to/models/'

#Set default precision to float64, fix random seed for reproducibility, and set L-BFGS options
dde.config.set_default_float("float64")
dde.config.set_random_seed(1234)
dde.optimizers.set_LBFGS_options(maxiter=5000)

L =       50 # Size of domain in Debye lengths
Z =       1 # Ion charge
niMax =   10 #normalized density
xMax =    L
xMin =    0
MiMeMin = 1*1836 #hydrogen
MiMeMax = 40*1836 #argon
TiTeMin = 0 # eV
TiTeMax = 1 # eV
KnudMin = 0 # collisionality parameters for ion-neutral collisions
KnudMax = 3e-1


#network params
epochsADAM = 5000
epochsBFGS = 20000
NumBFGS = 10
lr = 1.e-5
pts = 200000
bdypts = 0

#mass electron
me = 9.109e-31 #kg

#charge of an electron
q = 1.602e-19 #C

#permittivity of free space
Eps0 = 8.85e-12 #F/m

#T and Ez in eV
def Sion(T,Ez): 
    #Fit to ionization rate from NRL formulary
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * np.exp(-Ez/T) #m^3/s

def Srecom(T,Ez,Z):
    #Fit to recombination rate from NRL formulary
    return 5.2e-20 * Z * (Ez/T)**(1/2) * ( 0.43 + 1/2*np.log(Ez/T) + 0.469*(Ez/T)**(-1.3) ) #m^3/s 

def save_solution(geom, model, filename):
    x = geom.uniform_points(40**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(save_path + 'data/' + filename + "_fine.dat", np.hstack((x, y_pred)))

    x = geom.uniform_points(20**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(save_path + 'data/' +  filename + "_coarse.dat", np.hstack((x, y_pred)))

#normalizations
Tref = 1 #eV

def feature_transform(inputs):
    xNorm, MiMeNorm,TiTeNorm,KnudNorm = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3],inputs[:,3:4]
    xSq = xNorm * xNorm
    return torch.cat((xSq,MiMeNorm,TiTeNorm,KnudNorm),dim=1)

def output_transform(inputs, outputs):
    xNorm = inputs[:, 0:1]

    x = xMin + ( xMax - xMin ) * xNorm

    #Phi is 0 at x=-L and x=L
    phiFinal = (L-x)*(L+x)/L**2 * outputs[:,0:1]
    #ni is positive and less than niMax
    niFinal = niMax*0.5*( 1 +torch.tanh(outputs[:, 1:2]) ) 
    
    return torch.cat((phiFinal,niFinal), dim=1)

#PDE Definition
def pde(inputs, outputs):
    phi, ni = outputs[:, 0:1], outputs[:, 1:2]

    #Unfolding the network inputs
    #They are currently scaled to [0,1] for the network
    xNorm, MiMeNorm, TiTeNorm,KnudNorm = inputs[:, 0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4]

    #Scaling the inputs back to their non-dimensional values, which the PDE expects
    x = xMin + ( xMax - xMin ) * xNorm
    MiMe = MiMeMin + (MiMeMax - MiMeMin) * MiMeNorm
    TiTe = TiTeMin + (TiTeMax - TiTeMin) * TiTeNorm
    Knud = KnudMin + (KnudMax - KnudMin) * KnudNorm

    #Defining constants in the PDE
    Source = 1/L # constant source
    Source_int = x/L #integral of continuity equation n*u = x/L
    FluxAtWall = 1 #when x=1, n*u = 1
    uewall = torch.sqrt(MiMe/(2*np.pi)) #Velocity at wall from random flux
    ne = FluxAtWall/uewall*torch.exp(phi) #electron density at wall
    ui = Source_int / ni #ambipolarity condition ui = ne*ue / ni = x/(L*ni)
    
    #Auto-grad for the derivatives | one 'x' is a first derivative, two 'x's is a second derivative
    #Derivatives need to be scaled since the network's inputs are scaled to [0,1]
    #d/dx = d/dxNorm * dxNorm/dx = d/dxNorm * 1/(xMax-xMin)
    dphi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (xMax-xMin)
    dphi_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0) / (xMax-xMin)**2
    dni_x = dde.grad.jacobian(outputs, inputs, i=1, j=0) / (xMax-xMin)
    dui_x = dde.grad.jacobian(ui, inputs, i=0, j=0) / (xMax-xMin)
    
    #Physics Equations
    lossb1 = dphi_xx - (ne-ni) # Poisson
    lossb2 = L*(TiTe*dni_x + Z*ni*dphi_x + ui*Source + Source_int*dui_x + Source_int*Knud)#ion momentum

    return lossb1,lossb2

def main():
    #set 4D hypercube for inputs scaled to [0,1]
    geom = dde.geometry.Hypercube([0,0,0,0], [1,1,1,1])

    #Set up the neural network architecture and apply input (feature) and output transforms
    net = dde.maps.FNN([4] + [32]*4 + [2], 'tanh', "Glorot normal")
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    losses = []

    #Pass the network structure, objective function, and point sampling to the data object
    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain=pts,
        num_boundary=bdypts,
        num_test=pts,
        train_distribution='Hammersley',
        test_distribution='Sobol',
    )

    #build model
    model = dde.Model(data, net)

    #Set up a callback to resample the PDE points every 500 iterations
    PDE_Resampler = dde.callbacks.PDEPointResampler(period=500,pde_points=True)

    #Manual weights for loss terms if desired
    loss_weights = [1,1]
    loss = ["MSE"] * 2

    #Compile and train model using ADAM
    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights)
    losshistory, train_state = model.train(epochs=0,callbacks=[PDE_Resampler], model_save_path = save_path + 'model.pt')

    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights)
    losshistory, train_state = model.train(epochs=epochsADAM,callbacks=[PDE_Resampler], model_save_path = save_path + 'model.pt')

    save_solution(geom, model, "solution0")
    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir= save_path + 'data')

    #Complie and train using SSBroyden
    for i in range(0,NumBFGS):
        model.compile("SSBroyden")
        losshistory, train_state = model.train(display_every=1000,callbacks=[PDE_Resampler],model_save_path = save_path + 'model.pt')

        save_solution(geom, model, "solution0")

        dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir= save_path + 'data')

if __name__ == "__main__":
    main()
