import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize
#import scipy.differentiate
import scipy.integrate
import deepxde as dde
#import pytorch
from deepxde.backend import torch

sheathbool = False #Turns off/on sheath entrance calculations as well as graphs that use those calculations

import time
import sys

args = sys.argv
num_params = len(sys.argv)
print(args)
try:
    start_time = float(sys.argv[2])
except:
    start_time = time.time()
    print("Using: time.time()")

if num_params > 1:
    name = sys.argv[1]

save_path = 'path/to/save/figures'

Data_path = 'path/to/nn/checkpoint'
ckpt_save_path = Data_path + "model.pt-25033.pt"
#make sure to update the correct checkpoint number
trainpts = np.loadtxt(Data_path + 'data/train.dat')
loss = np.loadtxt(Data_path + 'data/loss.dat')

dde.config.set_default_float("float64")

numxpts = 1000 # only used when model is loaded

interiorpts = [2,2]

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
Ti = 0.026/3

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
nr = S0*L*xr/ur                     #m^-3 | For Ts=5eV and S0=1e25, nr~5.2e17 m^-3
                                   

Epsnormsq = Eps0**2*Ts**2/(q**2*xr*nr)

#Setup the exact same network as the training script
Tewall = dde.Variable(-0.219, dtype=torch.float64)

print(f"nr = {nr:.2e} m^-3")
print(f"ur = {ur:.2e} m s^-1")
print(f"xr = {xr:.2e} m")
print(f"Epsnormsq = {Epsnormsq:.2e} F^2 m^-2")
print(f"tr = {xr/ur:.2e} s")

def save_solution(geom, model, filename):
    x = geom.uniform_points(40**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    #np.savetxt(filename + "_fine.dat", np.hstack((x, y_pred, alpha(y_pred[:, -1:]))))
    np.savetxt(save_path + 'data/' + filename + "_fine.dat", np.hstack((x, y_pred)))

    #x = geom.uniform_points(4096)
    x = geom.uniform_points(20**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(save_path + 'data/' +  filename + "_coarse.dat", np.hstack((x, y_pred)))

def egyrofreq(B):
    #B in Tesla
    return 1.76e3*B

def coulog(n,T):
    #n in m^-3 and T in eV
    #Works for tensors
    n_dim = n*nr/1e6 #convert to cm^-3
    T_dim = T*Ts
    return 23.4 - 1.15*torch.log10(n_dim) + 3.45*torch.log10(T_dim)

def coulogd(n,T):
    #n in m^-3 and T in eV
    #works for np arrays
    n_dim = n*nr/1e6 #convert to cm^-3
    T_dim = T*Ts
    return 23.4 - 1.15*np.log10(n_dim) + 3.45*np.log10(T_dim)

def confine(n,T):
    #n in m^-3 and T in eV
    return 3.5e4/(0.1*coulog(n,T)) * torch.sqrt(T**3)/n

def nuei(ni,ne,T):
    ni_dim = ni*nr
    ne_dim = ne*nr
    T_dim = T*Ts
    return 0.468e-16*coulog(ne_dim,T_dim)*ni_dim*ne_dim/((T_dim/1000)*torch.sqrt(T_dim/1000)) #m^-3 s^-1

def nueq(ni,ne,T):
    ni_dim = ni*nr
    ne_dim = ne*nr
    T_dim = T*Ts
    return 2.9e-12 * ni_dim*ne_dim*coulog(ne_dim,T_dim)/((T_dim)*torch.sqrt(T_dim)) #m^-3 s^-1

def Taue(ni,T):
    #This works with tensors
    e = 4.8032e-10 #statC
    k = 1.3807e-16 #erg K^-1
    Z=1
    ni_dim = ni*nr
    T_dim = T*Ts #this can stay eV
    return 12*np.pi**(1.5)/np.sqrt(2) * np.sqrt(1/MiMe)*Epsnormsq*T**(3/2) / (coulog(ni,T)*ni*Z**2) #s

def Taued(ni,T):
    #This works with np arrays
    e = 4.8032e-10 #statC
    k = 1.3807e-16 #erg K^-1
    Z=1
    ni_dim = ni*nr
    T_dim = T*Ts #this can stay eV
    return 12*np.pi**(1.5)/np.sqrt(2) * np.sqrt(1/MiMe)*Epsnormsq*T**(3/2) / (coulogd(ni,T)*ni*Z**2) #s

def boundaryRight(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

def boundaryLeft(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def feature_transform(inputs):
    xNorm = inputs[:,0:1]
    xSq = xNorm * xNorm
    return xSq

def output_transform(inputs, outputs):
    xNorm = inputs[:,0:1]

    x = xNorm*(xMax-xMin) + xMin

    TewallF = 0.6*0.5*(1+torch.tanh(Tewall)) +0.00001
    uewall = torch.sqrt(MiMe*TewallF/(2*np.pi))
    newall = 1/uewall
    phiFinal = (L-x)*(L+x)/L**2 * (outputs[:,0:1])
    niFinal  = torch.log( 1 + torch.exp(outputs[:, 1:2]) )
    neFinal = (L-x)*(L+x)/L**2 * torch.log(1+torch.exp(outputs[:,2:3])) + newall
    TeFinal = (L-x)*(L+x)/L**2*0.6*0.5*(1+torch.tanh(outputs[:,3:4])) + (x/L)**2*TewallF

    return torch.cat((phiFinal,niFinal,neFinal,TeFinal), dim=1)


def pde(inputs, outputs):
    phi, ni, ne, Te = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:,3:4]#, outputs[:,4:5]

    xNorm = inputs[:, 0:1]

    x = xMin + ( xMax - xMin ) * xNorm

    dphi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (xMax-xMin)
    dphi_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0) / (xMax-xMin)**2
  
    source = x/L
    #source = n*u = x/L
    ui = source/ni
    ue = source/ne

    due_x = dde.grad.jacobian(ue,inputs, i=0, j=0) / (xMax-xMin)
    dui_x = dde.grad.jacobian(ui,inputs,i=0,j=0) / (xMax-xMin)
    dne_x = dde.grad.jacobian(ne,inputs, i=0, j=0) / (xMax-xMin)
    dni_x = dde.grad.jacobian(ni,inputs, i=0, j=0) / (xMax-xMin)
    dTe_x = dde.grad.jacobian(Te,inputs, i=0, j=0) / (xMax-xMin)
    dTe_xx = dde.grad.hessian(Te,inputs,component=0, i=0, j=0) / (xMax-xMin)**2

    qe = 0.71*ne*Te*(ue-ui) - 3.16*ne*Te*MiMe*dTe_x*Taue(ni,Te) #conduction

    dq_x = dde.grad.jacobian(qe, inputs, i=0,j=0) / (xMax-xMin)

    meoverme = 1

    Rue = -meoverme*(ue-ui)*ne/Taue(ni,Te) * 0.51                     
    RTe = -0.71*ne*MiMe*dTe_x                     

    QRe = -(Rue + RTe)*(ue-ui)           
    Qeqe = -3/MiMe*ne/Taue(ni,Te)*(Te-Ti)  

    Rui = 1/MiMe*(ue-ui)*ne/Taue(ni,Te) 
    RTi = 0.71*ne*dTe_x

    lossb1 = dphi_xx - (ne - ni) # Poisson
    lossb3 = (ne*dTe_x + Te*dne_x - ne*dphi_x)*MiMe - Rue - RTe # + ue/(L*MiMe))*MiMe #electron momentum
    lossb5 = L*ni*dphi_x + ui + L*source*dui_x - Rui - RTi #ion momentum
    lossb6 = 3/2*x*dTe_x + L*ne*Te*due_x + L*dq_x - 1/(2)*(3-3*Te+1/MiMe*(ue**2)) - QRe - Qeqe 
                           #^convection^

    return lossb1,lossb3,lossb5,lossb6

geom = dde.geometry.Hypercube([0,0], [1,1])

uniform_points = geom.random_points(interiorpts[0])

points = uniform_points

net = dde.maps.FNN([1]+[32]*3+[5], "tanh", "Glorot normal")
net.apply_feature_transform(feature_transform)
net.apply_output_transform(output_transform)

def func(inputs,outputs):
    newall = outputs[:,2:3]
    niwall = outputs[:,1:2]
    uewall = 1/newall
    uiwall = 1/niwall
    return (3/2 - 5/2*outputs[:,3:4] - 1/2*1/MiMe*uewall**2 - 1/2*(uiwall)**2-0.71*newall*outputs[:,3:4]*(uewall-uiwall))/( -3.16*MiMe*newall*outputs[:,3:4]*Taue(niwall,outputs[:,3:4]) )

bc_wall = dde.icbc.RobinBC(geom, func, boundaryRight, component=3)


data = dde.data.PDE(
    geom,
    pde,
    [],
    #num_domain=num_domain,
    #num_boundary=num_boundary,
    #num_test=2**13,
    #anchors = points
)

#loss_weights = [1,1,1,1]
#lossE = ["MSE"] * 4

model = dde.Model(data, net)
model.compile("SSBroyden")#,external_trainable_variables=Tewall)
#variable = dde.callbacks.VariableValue(Tewall, period=1000)
#Load model weights
model.restore(save_path = ckpt_save_path, verbose=1)

# Load training points
xtrainpts = trainpts[:]

# Load loss history

steps = loss[:,0]
trainloss = loss[:,1]
trainloss = trainloss + loss[:,2]
trainloss = trainloss + loss[:,3]
trainloss = trainloss + loss[:,4]
trainloss = trainloss + loss[:,5]

testloss = loss[:,6]
testloss = testloss + loss[:,7]
testloss = testloss + loss[:,8]
testloss = testloss + loss[:,9]
testloss = testloss + loss[:,10]

#create space array
xpts = np.linspace(0,1,numxpts)

#normalize dimensional space
def xnom(x):
    return (x - xMin) / (xMax-xMin)

#function for predicting profiles from a normalized xVal or array of normalized xVals
def pred(xVal):
    x = xMin + (xMax-xMin)*xVal
    X2 = np.zeros([len(xVal),1])
    X2[:,0] = xVal
    y2 = model.predict(X2)

    phi = y2[:,0]
    ni = y2[:,1]
    ne = y2[:,2]
    Te = y2[:,3]

    Ti = 0.026/3
    ui = x/L/ni
    ue = x/L/ne

    return phi,ni,ne,ui,ue,Te,Ti

#List form for evaluating at multiple specific spatial points if interested
phiList = []
neList = []
niList = []
ueList = []
uiList = []
TeList = []
TiList = []

#Evaluate profiles for all x
phi, ni, ne, ui, ue, Te, Ti = pred(xpts)
phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)
TiList.append(Ti)
TeList.append(Te)

#switch to dimensional space for plotting
xpts = np.linspace(0,L,numxpts)

#function that compute derivatives for numpy arrays
def derivative(y):
    dy = np.gradient(y, xpts)
    return dy

#Computes the robin boundary condition from predicted values
def func(ni,Te):
    ui = 1/ni
    uewall = np.sqrt(MiMe*Te/(2*np.pi))
    newall = 1/uewall
    return (3/2 - 5/2*Te - 1/2*1/MiMe*uewall**2 - 1/2*ui**2-0.71*newall*Te*(uewall-ui))/( -3.16*MiMe*newall*Te*Taued(ni,Te) )

#finds the location where the ion velocity reaches real sound speed
def Prhho(xVal):
    phi, ni, ne, ui, ue, Te, Ti = pred(xVal)

    return ui - np.sqrt(Te)

#normalized sheath entrance location
sol = optimize.root(Prhho, [(L-5)/L], args = ())
xSE1 = sol.x
print("sheath entrance Bohm = " + str(xSE1))

#predict values at the sheath entrance
phiSE1, niSE1, neSE1, uiSE1, ueSE1, TeSE1, TiSE1 = pred(xSE1)
pp = (niSE1-neSE1)/neSE1
print("Phro:", pp) #print for comparison

#this value is calculated from an earlier model that used the exact same assumptions as Bohm 
#to calculate the sheath entrance in order to get the fractional charge separation
#consistent with Langmuir's definition of the sheath entrance
Prho = 0.0652

#function to predict charge seperation at a given spatial point
def SheathEntrance(xVal):
    phi, ni, ne, ui, ue, Te, Ti = pred(xVal)

    return (ni-ne) / ne - Prho

#find the sheath entrance using Langmuir's definition
sol2 = optimize.root(SheathEntrance, [(L-5)/L], args = ())
xSE2 = sol2.x
print("Sheath Entrance Langmuir", xSE2)

################################################
#Plot sheath profiles
################################################

plt.rcParams.update({'font.size': 16})

fig1, ax1 = plt.subplots(num=1,nrows=1,ncols=1, clear=True)
fig1.set_tight_layout(True)

phi = phiList[0]
ax1.plot((xpts), phi, label='$\\phi$', linestyle='-',color='r',linewidth=3)

ax1.set_xlabel("$x/\\lambda_{de}$")
ax1.set_title("$e\\phi/T_s$")
fig1.savefig(save_path + "phi.png")

fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)

ne = neList[0]
ni = niList[0]
ax2.plot((xpts), ni, label='$n_i$', linestyle='-',color='b',linewidth=3)
ax2.plot((xpts), ne, label='$n_e$', linestyle='--',color='r',linewidth=3)

ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.legend()
ax2.set_title("Density")
fig2.savefig(save_path + "Densities.png")


fig3, ax3 = plt.subplots(num=3,nrows=1,ncols=1, clear=True)
fig3.set_tight_layout(True)

Te = TeList[0]
ax3.plot(xpts, np.sqrt(Te), label='$C_s$', linestyle=':',color='k',linewidth=3)

ue = ueList[0]
ui = uiList[0]
ax3.plot((xpts), ui, label='Ion', linestyle='-',color='b',linewidth=3)
ax3.plot((xpts), ue, label='electron',linestyle='--',color='r',linewidth=3)

ax3.set_xlabel("$x/\\lambda_{De}$")
ax3.set_title("Velocity [$C_s$]")          
ax3.set_ylim([0,5])
ax3.legend()
#ax3.set_yscale("log")
fig3.savefig(save_path + "Velocities.png")

fluchs = xpts/L

fig4, ax4 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig4.set_tight_layout(True)

ue = ueList[0]
ui = uiList[0]
ne = neList[0] 
ni = niList[0]

ax4.plot((xpts), ui*ni, label='Electron Flux', linestyle='-',color='b',linewidth=3)
ax4.plot((xpts), fluchs, label="Ana Flux",linestyle='--', color='k',linewidth=3)

ax4.set_xlabel("$x/\\lambda_{de}$")
ax4.set_title("Flux")
fig4.savefig(save_path + "Flux")

fig5, ax5 = plt.subplots(num=5,nrows=1,ncols=1, clear=True)
fig5.set_tight_layout(True)

Te = TeList[0]
ax5.plot((xpts), Te, label='$T_e$', linestyle='-',color='r',linewidth=3)

#ax5.legend()
ax5.set_xlabel("$x/\\lambda_{de}$")
ax5.set_title("Temperature ($T/T_s$)")
fig5.savefig(save_path + "Te")

print("drift velocity = ",ueList[0][200]-uiList[0][200] )

#compute the temperature derivative as a function of x
def func2(x,ne,ni,ue,ui,Te):
    return (3/2*x/L - 5/2*Te*ne*ue - 1/2*1/MiMe*ne*ue**3 - 1/2*ni*(ui)**3-0.71*ne*Te*(ue-ui))/( -3.16*MiMe*ne*Te*Taued(ni,Te) )

#total heat transfer
def qtote(x,ne,ni,ue,ui,T):
    qconv_th = 5/2*ne*T*ue #convection of thermal energy
    qconv_KE = 1/2/MiMe*ne*ue**3 #convection of kinetic energy
    qconv = qconv_th + qconv_KE
    dTe_x = func2(x,ne,ni,ue,ui,T) #derivative(T) 
    np.savetxt(save_path + "dTe_x.txt", dTe_x)
    qcond = - 3.16*ne*T*MiMe*dTe_x*Taued(ni,T) + 0.71*ne*T*(ue-ui) #conduction of energy through collisions
    qtot = qconv + qcond
    return qconv, qcond, qtot

#ion total heat transfer
def qtoti(n,ui,T):
    qconv_th = 5/2*n*T*ui #convection of thermal energy
    qconv_KE = 1/2*n*ui**3 #convection of kinetic energy
    qconv = qconv_th + qconv_KE
    dTi_x = 0 #derivative(T)
    qcond = 0 #assumed no ion conduction
    qtot = qconv + qcond
    return qconv, qcond, qtot

#calculate sheath heat transmission coefficient
def Heat_Transmission(x):
    phi, ni, ne, ui, ue, Te, Ti = pred(x)
    _,_,qe = qtote(x*L,ne,ni,ue,ui,Te)
    _,_,qi = qtoti(ni,ui,Ti)
    eHeatCoeff = qe/(ne*ue*Te)
    print("qe = ", qe)
    print("qi = ", qi)
    iHeatCoeff = qi/(ni*ui*Te)
    return eHeatCoeff, iHeatCoeff

eHeatCoeff, iHeatCoeff = Heat_Transmission(xSE1)
print("Electron Heat Transmission Coefficient = ", eHeatCoeff)
print("Ion Heat Transmission Coefficient = ", iHeatCoeff)
print("Total Heat Transmission Coefficient = ", eHeatCoeff+iHeatCoeff)

fig6, ax6 = plt.subplots(num=6,nrows=1,ncols=1, clear=True)
fig6.set_tight_layout(True)

Te = TeList[0]
ne = neList[0]
ni = niList[0]
ue = ueList[0]
ui = uiList[0]
qeconv,qecond,qetot = qtote(xpts,ne,ni,ue,ui,Te)
qiconv,qicond,qitot = qtoti(ni,ui,Ti)
qtot = qetot + qitot
#qtot[-1] = 1.5

ax6.plot((xpts), qeconv, label='Convective', linestyle='-',color='b',linewidth=3)
ax6.plot((xpts), qecond, label='Conductive', linestyle='-',color='r',linewidth=3)
ax6.plot((xpts), qtot, label='Total', linestyle='-',color='k',linewidth=3)

ax6.plot((xpts), qitot, label='Ion total', linestyle='--',color='k',linewidth=3)

ax6.set_xlabel("$x/\\lambda_{de}$")
#ax6.set_yscale('log')
ax6.legend()
ax6.set_title("Heat Flux")
fig6.savefig(save_path + "q")

#compute collision rate for all ne and Te
def Tau(ne,Te):
    T = np.zeros((len(ne),len(Te)))
    for i in range(len(ne)):
        for j in range(len(Te)):
            T[i,j] = Taued(ne[i],Te[j])
    return T

fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
fig7.set_tight_layout(True)

Te = TeList[0]
ne = neList[0]
ue = ueList[0]
cs = ax7.contourf(ne, Te, Tau(ne,Te).T, levels=50, cmap='jet')

ax7.set_xlabel("$n_e$")
ax7.set_ylabel("$T_e$")
fig7.colorbar(cs, ax=ax7)
ax7.set_title("$\\tau$")

fig7.savefig(save_path + "Tau")
plt.close()

def fixloss(l):
    #number of steps don't align with number of tests due to test_every > 1
    #only pull the test loss for the steps where it is actually calculated
    teststeps = [0]
    loss2 = [l[0]]
    for i in range(len(l)-1):
        if l[i+1] != l[i]:
            loss2.append(l[i+1])
            teststeps.append(steps[i+1])
    return np.array(teststeps),loss2

fig16, ax16 = plt.subplots(num=16,nrows=1,ncols=1, clear=True)
fig16.set_tight_layout(True)

ax16.plot(steps/1000, loss[:,1], label='Poisson', linestyle='-',color='red',linewidth=3)
ax16.plot(steps/1000, loss[:,2], label='Elec Momentum', linestyle='-',color='blue',linewidth=3)
ax16.plot(steps/1000, loss[:,3], label='Ion Momentum', linestyle='-',color='k',linewidth=3)
ax16.plot(steps/1000, loss[:,4], label='Elec Energy', linestyle='-',color='g',linewidth=3)
ax16.plot(steps/1000, loss[:,5], label='Boundary Loss', linestyle='-',color='c',linewidth=3)
x,y = fixloss(loss[:,6])
ax16.scatter(x/1000, y, color='r',linewidth=2) #label='P Test'
x,y = fixloss(loss[:,7])
ax16.scatter(x/1000, y, color='b',linewidth=2) #label='EM Test'
x,y = fixloss(loss[:,8])
ax16.scatter(x/1000, y, color='k',linewidth=2) #label='IM Test'
x,y = fixloss(loss[:,9])
ax16.scatter(x/1000, y, color='g',linewidth=2) #label='EE Test' 
x,y = fixloss(loss[:,10])
ax16.scatter(x/1000, y, color='c',linewidth=2) #label='BC Test'

#ax16.set_ylabel("$\\phi$")
ax16.set_xlabel("Epochs (x1000)")
ax16.set_title("Losses")
#ax16.set_ylim([1e-7,5e-4])
ax16.set_yscale("log")
ax16.legend()
fig16.savefig(save_path + "Losses.png")

xpts = np.linspace(0,1,100)
X2 = xpts
y2 = model.predict(X2,operator=pde)

resphi = y2[0]
resem = y2[1]
resim = y2[2]
resee = y2[3]

fig17,ax17 = plt.subplots(num=17,nrows=1,ncols=1, clear=True)
fig17.set_tight_layout(True)
ax17.plot(xpts,resphi,c='k')

fig17.savefig(save_path + 'Residual_phi.png') 

fig18,ax18 = plt.subplots(num=18,nrows=1,ncols=1, clear=True)
fig18.set_tight_layout(True)
ax18.plot(xpts,resem,c='k')

fig18.savefig(save_path + 'Residual_em.png') 

fig19,ax19 = plt.subplots(num=19,nrows=1,ncols=1, clear=True)
fig19.set_tight_layout(True)
ax19.plot(xpts,resim,c='k')

fig19.savefig(save_path + 'Residual_im.png') 

fig20,ax20 = plt.subplots(num=20,nrows=1,ncols=1, clear=True)
fig20.set_tight_layout(True)
ax20.plot(xpts,resee,c='k')

fig20.savefig(save_path + 'Residual_ee.png') 
