import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch

import time
import sys

save_path = '/path/to/model/'

#set default options
dde.config.set_default_float("float64")
dde.config.set_random_seed(1234)
dde.optimizers.set_LBFGS_options(maxiter=5000)

L = 100 # Size of domain in Debye lengths
Z = 1 # Ion charge

xMax = L
xMin = 0

#System Parameters
alpha = 0.71  #corresponds to the value when z=1 | Thermal force coefficient
MiMe = 1836
B0 = 5        # T
Ts = 3        #eV
S0 = 1e31     #particles m^-3 s^-1
Ti = 0.026/Ts

#network params
epochsADAM = 10000
epochsBFGS = 5000
NumBFGS = 15
lr = 1.e-4
pts = 150000
bdypts = 16

#mass electron
me = 9.109e-31 #kg

#charge of an electron
q = 1.602e-19 #C

#permittivity of free space
Eps0 = 8.85e-12 #F/m

#for hydrogen, 
#first ionization potential
Einf = 13.6 #eV

#ion mass
mi = MiMe*me#1.6735575e-27 #kg

#Normalizations
ur = np.sqrt(Ts*q/mi)               #m s^-1
xr = (Eps0*Ts*ur/(S0*L*q))**(1/3)   #m
nr = S0*L*xr/ur                     #m^-3 
                                    
#normalization of Epsilon_0 squared for collision rate calculation
Epsnormsq = Eps0**2*Ts**2/(q**2*xr*nr)

#Initialize trainable variable for electron temperature at the wall
Tewall = dde.Variable(-1.0, dtype=torch.float64)

def save_solution(geom, model, filename):
    x = geom.uniform_points(40**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(save_path + 'data/' + filename + "_fine.dat", np.hstack((x, y_pred)))

    x = geom.uniform_points(20**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(save_path + 'data/' +  filename + "_coarse.dat", np.hstack((x, y_pred)))

def egyrofreq(B):
    #B in Tesla
    #NRL formulary fit for electron gyrofrequency in rad/s
    return 1.76e3*B

def coulog(n,T):
    #n in m^-3 and T in eV
    #Coulomb logarithm from NRL Formulary
    n_dim = n*nr/1e6 #convert to cm^-3
    T_dim = T*Ts
    return 23.4 - 1.15*torch.log10(n_dim) + 3.45*torch.log10(T_dim)

def confine(n,T):
    #n in m^-3 and T in eV
    #From NRL Formulary
    return 3.5e4/(0.1*coulog(n,T)) * torch.sqrt(T**3)/n

def nuei(ni,ne,T):
    #inputs are normalized
    #From Collisional Transport in Magnetized Plasma by Per Helander
    ni_dim = ni*nr
    ne_dim = ne*nr
    T_dim = T*Ts
    return 0.468e-16*coulog(ne_dim,T_dim)*ni_dim*ne_dim/((T_dim/1000)*torch.sqrt(T_dim/1000)) #m^-3 s^-1

def nueq(ni,ne,T):
    #Inputs are normalized
    #From Collisional Transport in Magnetized Plasma by Per Helander
    ni_dim = ni*nr
    ne_dim = ne*nr
    T_dim = T*Ts
    return 2.9e-12 * ni_dim*ne_dim*coulog(ne_dim,T_dim)/((T_dim)*torch.sqrt(T_dim)) #m^-3 s^-1

def Taue(ni,T):
    #inputs are normalized
    #From Collisional Transport in Magnetized Plasma by Per Helander
    e = 4.8032e-10 #statC
    k = 1.3807e-16 #erg K^-1
    Z=1
    ni_dim = ni*nr 
    T_dim = T*Ts 
    return 12*np.pi**(1.5)/np.sqrt(2) * np.sqrt(1/MiMe)*Epsnormsq*T**(3/2) / (coulog(ni,T)*ni*Z**2) #s

def boundaryRight(x, on_boundary):
    #Evaluates on the domain "right" boundary or if point is close to boundary
    return on_boundary and dde.utils.isclose(x[0], 1)

def boundaryLeft(x, on_boundary):
    #Evaluates on the domain "left" boundary or if point is close to boundary
    return on_boundary and dde.utils.isclose(x[0], 0)


def feature_transform(inputs):
    #Transform x into x^2 
    xNorm = inputs[:,0:1]
    xSq = xNorm * xNorm
    return xSq

def output_transform(inputs, outputs):
    xNorm = inputs[:,0:1]

    x = xNorm*(xMax-xMin) + xMin
    #give Tewall a max value (0.6) and prevent it from going exactly to 0
    TewallF = 0.6*0.5*(1+torch.tanh(Tewall)) +0.00001
    uewall = torch.sqrt(MiMe*TewallF/(2*np.pi))
    newall = 1/uewall
    #constrain phi to be 0 at both walls
    phiFinal = (L-x)*(L+x)/L**2 * (outputs[:,0:1])
    #Constrain ni to be positive
    niFinal  = torch.log( 1 + torch.exp(outputs[:, 1:2]) )
    #constrain ne to be positive and equal newall at both walls
    neFinal = (L-x)*(L+x)/L**2 * torch.log(1+torch.exp(outputs[:,2:3])) + newall
    #constrain Te to be positive, below 0.6, and equal to Tewall at both walls
    TeFinal = (L-x)*(L+x)/L**2*0.6*0.5*(1+torch.tanh(outputs[:,3:4])) + (x/L)**2*TewallF

    return torch.cat((phiFinal,niFinal,neFinal,TeFinal), dim=1)


def pde(inputs, outputs):
    #Physics equations
    phi, ni, ne, Te = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:,3:4]
    xNorm = inputs[:, 0:1]
    
    x = xMin + ( xMax - xMin ) * xNorm

    #Use autograd for the gradients
    dphi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (xMax-xMin)
    dphi_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0) / (xMax-xMin)**2
    
    #Define constants
    source = x/L
    ui = source/ni
    ue = source/ne

    #Autograd for more gradients
    due_x = dde.grad.jacobian(ue,inputs, i=0, j=0) / (xMax-xMin)
    dui_x = dde.grad.jacobian(ui,inputs,i=0,j=0) / (xMax-xMin)
    dne_x = dde.grad.jacobian(ne,inputs, i=0, j=0) / (xMax-xMin)
    dni_x = dde.grad.jacobian(ni,inputs, i=0, j=0) / (xMax-xMin)
    dTe_x = dde.grad.jacobian(Te,inputs, i=0, j=0) / (xMax-xMin)
    dTe_xx = dde.grad.hessian(Te,inputs, i=0, j=0) / (xMax-xMin)**2

    #define convective and conductive losses
    qe = 0.71*ne*Te*(ue-ui) - 3.16*ne*Te*MiMe*dTe_x*Taue(ni,Te) 

    #gradient of losses using autograd
    dq_x = dde.grad.jacobian(qe, inputs, i=0,j=0) / (xMax-xMin)
    meoverme = 1

    #Friction term for electron-ion collisions
    Rue = -meoverme*(ue-ui)*ne/Taue(ni,Te) * 0.51
    #momentum losses due to temperature gradient
    RTe = -0.71*ne*MiMe*dTe_x                     

    #joule heating due to net drift of electrons against the dissipative force
    QRe = -(Rue + RTe)*(ue-ui)         
    #heat exchange due to thermal equilibrium ion-electron collisions
    Qeqe = -3/MiMe*ne/Taue(ni,Te)*(Te-Ti)  

    #momentum gain due to ion-electron collisions
    Rui = 1/MiMe*(ue-ui)*ne/Taue(ni,Te) *0.51
    #momentum loss due temperature gradient
    RTi = 0.71*ne*dTe_x

    lossb1 = dphi_xx - (ne - ni) # Poisson
    lossb3 = (ne*dTe_x + Te*dne_x - ne*dphi_x)*MiMe - Rue - RTe #electron momentum
    lossb5 = L*ni*dphi_x + ui + L*source*dui_x - Rui - RTi #ion momentum
    lossb6 = 3/2*x*dTe_x + L*ne*Te*due_x + L*dq_x - 1/(2)*(3-3*Te+1/MiMe*(ue**2)) - QRe - Qeqe #electron energy
                           #^convection^

    return lossb1,lossb3,lossb5,lossb6

def main():
    #Set up problem geometry
    geom = dde.geometry.Hypercube([0,0], [1,1])

    #Build network architecture
    #3 hidden layers with 32 neurons each using tanh activation function and glorot normal for the weight initialization
    n = 5
    activision = f"LAAF-{n} tanh"
    net = dde.maps.FNN([1]+[32]*3+[5], "tanh", "Glorot normal")
    #Apply both input (feature) and output transforms
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    losses = []

    def func(inputs,outputs):
        newall = outputs[:,2:3]
        niwall = outputs[:,1:2]
        uewall = 1/newall
        uiwall = 1/niwall
        #Define the robin boundary condition on temperature
        return (3/2 - 5/2*outputs[:,3:4] - 1/2*1/MiMe*uewall**2 - 1/2*(uiwall)**2-0.71*newall*outputs[:,3:4]*(uewall-uiwall))/( -3.16*MiMe*newall*outputs[:,3:4]*Taue(niwall,outputs[:,3:4]) )

    bc_wall = dde.icbc.RobinBC(geom, func, boundaryRight, component=3)

    #Pass the geometry and points, equations, and boundary conditions into data
    data = dde.data.PDE(
        geom,
        pde,
        [bc_wall],
        num_domain=pts,
        num_boundary=bdypts,
        num_test=pts,
        train_distribution='Hammersley'
    )
    #Build the model
    model = dde.Model(data, net)

    #Weight Poisson slightly higher than the others to equalize the initial residuals
    loss_weights = [10,1,1,1,1] 

    #resample points every 500 iterations
    PDE_Resampler = dde.callbacks.PDEPointResampler(period=500,pde_points=True)

    #tell the optimizer to use MSE for the loss terms
    loss = ["MSE"] * 5
    #compile and train the model with adam, allowing it to learn the trainable variable as well
    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights,external_trainable_variables=Tewall)
    variable = dde.callbacks.VariableValue(Tewall, period=1000)
    losshistory, train_state = model.train(epochs=0,callbacks=[variable,PDE_Resampler], model_save_path = save_path + 'model.pt')

    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights,external_trainable_variables=Tewall)
    losshistory, train_state = model.train(epochs=epochsADAM,callbacks=[variable,PDE_Resampler], model_save_path = save_path + 'model.pt')

    save_solution(geom, model, "solution0")
    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir= save_path + 'data')

    #Then polish the model with SSBroyden to drive residual down further
    #Do NOT resample points because it destroys SSB's training
    for i in range(0,NumBFGS):
        model.compile("SSBroyden",external_trainable_variables=Tewall)

        model.train_step.optimizer_kwargs = {'options': {'maxcor': 100,
                                                        'ftol': 1.0 * np.finfo(float).eps,
                                                        'gtol': 1.0 * np.finfo(float).eps,
                                                        'maxiter': epochsBFGS,
                                                        'maxfun':  epochsBFGS,
                                                        'method_bfgs':'SSBroyden2',
                                                        # 'method_bfgs':'l-bfgs-b',
                                                        'maxls': 200}}

        losshistory, train_state = model.train(display_every=1000,callbacks=[variable,PDE_Resampler],model_save_path = save_path + 'model.pt')

        
        save_solution(geom, model, "solution0")

        dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir= save_path + 'data')
        #Get the value of Tewall after training
        Tew = variable.get_value()
        print(Tew)
if __name__ == "__main__":
    main()
