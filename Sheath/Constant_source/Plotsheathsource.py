import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize
from scipy import special
from scipy.stats import qmc
import scipy.integrate
import math
import os
import deepxde as dde
#from deepxde.backend import tf
import torch


sheathbool = True #Turns off/on sheath entrance calculations as well as graphs that use those calculations

import time
import sys

#save_path = '/blue/cmcdevitt/ewebb2/Sheath/models/'+name+'/'

#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2024-10-24-12:42:46/'
#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2024-01-31-18:12:29/'
#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2024-02-14-18:21:07/'
#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2024-02-15-15:50:19/'
#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2024-02-16-12:50:30/'
#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2025-02-03-11:40:36/'
Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/3001/'
save_path = '/blue/cmcdevitt/ewebb2/Sheath/models/3001/Figures/'

#Best L-BFGS: /blue/cmcdevitt/ewebb2/Sheath/models/2025-02-03-11:40:36/

ckpt_save_path = Data_path + "model.pt-45000.pt"

#soln = np.loadtxt('./data/solutionAnalytic0_fine.dat')
soln = np.loadtxt(Data_path + 'data/solution0_fine.dat')
trainpts = np.loadtxt(Data_path + 'data/train.dat')
loss = np.loadtxt(Data_path + 'data/loss.dat')
#soln = np.loadtxt('./ArchiveData/40DebyeHydrogenJune13_2022/solution0_fine.dat')
#trainpts = np.loadtxt('./ArchiveData/40DebyeHydrogenJune13_2022/train.dat')
#loss = np.loadtxt('./ArchiveData/40DebyeHydrogenJune13_2022/loss.dat')

dde.config.set_default_float("float64")

numxpts = 1000 # only used when model is loaded

L = 50 # Size of domain in Debye lengths
Z = 1 # Ion charge
AtomicMass = 1 # atomic mass to plot
#IonT = 1 #Ti to plot

#mi = 6.646e-27 #Mass helium in kg
mi = 1.67355e-27 #mass hydrogen in kg
me = 9.109e-31 #Mass electron Kg
q = 1.602e-19 #Charge of electron C
E0 = 8.8542e-12 #permittivity of free space F/m
Einf = 13.6

#KnudNeutralMin = 0 # collisionality parameters for ion-neutral collisions
#KnudNeutralMax = 3e-1
MiMeMin = 1*1836
MiMeMax = 40*1836
TiTeMin = 0
TiTeMax = 1 #Ti = Te 
xMin = 0
xMax = L
KnudMin = 0 # collisionality parameters for ion-neutral collisions
KnudMax = 3e-1

# Choose values to evaluate
#KnudNeutralVal1 = 3e-1 #Pressure = 10^4
MiMeVal1 = 1836
TiTeVal1 = TiTeMin
KnudVal1 = 0

#KnudNeutralVal2 = 7e-2 #Pressure = 10^3
MiMeVal2 = 40*1836
TiTeVal2 = TiTeMin
KnudVal2 = 0

#KnudNeutralVal3 = 1e-2 #Pressure = 10^2
MiMeVal3 = 1836
TiTeVal3 = TiTeMin
KnudVal3 = 3e-1

#KnudNeutralVal4 = 1.5e-3 #Pressure = 10^1
MiMeVal4 = 40*1836
TiTeVal4 = TiTeMin
KnudVal4 = 3e-1

#KnudNeutralVal5 = 1.5e-4 #Pressure = 10^0
"""MiMeVal5 = AtomicMass*1836
TiTeVal5 = TiTeMin
KnudVal5 = 1.5e-4

MiMeVal6 = AtomicMass*1836
TiTeVal6 = TiTeMax
KnudVal6 = 0.3"""

KnudNumVec = np.array([KnudVal1,KnudVal2,KnudVal3,KnudVal4])
MiMeVec = np.array([MiMeVal1,MiMeVal2,MiMeVal3,MiMeVal4])
TiTeVec = np.array([TiTeVal1,TiTeVal2,TiTeVal3,TiTeVal4])

def Sion(T,Ez): 
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * np.exp(-Ez/T) 

def Srecom(T,Ez,Z):
    return 5.2e-20 * Z * (Ez/T)**(1/2) * ( 0.43 + 1/2*np.log(Ez/T) + 0.469*(Ez/T)**(-1.3) ) #m^3/s

#Parameters
Tref = 1 #eV

#normalizations
Tref = 1 #eV

niMax =	10 # maximum density allowed

interiorpts = [2,2]

# Load training points
xtrainpts = trainpts[:]

# Load loss history

steps = loss[:,0]
trainloss = loss[:,1]
trainloss = trainloss + loss[:,2]
#trainloss += loss[:,3]
#trainloss += loss[:,4]
#trainloss += loss[:,5]
#trainloss += loss[:,6]
#trainloss += loss[:,7]

testloss = loss[:,3]
testloss = testloss + loss[:,4]
#testloss += loss[:,6]
#testloss += loss[:,7]
#testloss += loss[:,10]
#testloss += loss[:,13]
#testloss += loss[:,14]
#testloss += loss[:,15]

#C = dde.Variable(-1.0,dtype='float64')
#nT = tf.Variable(C, dtype='float64', trainable=True)

def feature_transform(inputs):
    xNorm, MiMeNorm,TiTeNorm,KnudNorm = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3],inputs[:,3:4]
    #xNorm, MiMeNorm,TiTeNorm= inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
    xSq = xNorm * xNorm
    return torch.cat((xSq,MiMeNorm,TiTeNorm,KnudNorm),dim=1)


def output_transform(inputs, outputs):
    xNorm = inputs[:, 0:1]

    x = xMin + ( xMax - xMin ) * xNorm

    phiFinal = (L-x)*(L+x)/L**2 * outputs[:,0:1]
    niFinal = niMax*0.5*( 1 + torch.tanh(outputs[:, 1:2]) ) 

    return torch.cat((phiFinal,niFinal), dim=1)

def pde(inputs, outputs):
    phi, ni = outputs[:, 0:1], outputs[:, 1:2]

    xNorm, MiMeNorm, TiTeNorm,KnudNorm = inputs[:, 0:1], inputs[:,1:2], inputs[:,2:3],inputs[:,3:4]
    
    x = xMin + ( xMax - xMin ) * xNorm
    MiMe = MiMeMin + (MiMeMax - MiMeMin) * MiMeNorm
    TiTe = TiTeMin + (TiTeMax - TiTeMin) * TiTeNorm
    Knud = KnudMin + (KnudMax - KnudMin) * KnudNorm

    Source = 1/L
    Source_int = x/L
    FluxAtWall = 1
    uewall = torch.sqrt(MiMe/(2*np.pi))
    ne = FluxAtWall/uewall*torch.exp(phi)
    #ue = Source_int / ne
    ui = Source_int / ni
    
    dphi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (xMax-xMin)
    dphi_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0) / (xMax-xMin)**2
    dni_x = dde.grad.jacobian(outputs, inputs, i=1, j=0) / (xMax-xMin)
    
    #dne_x = dde.grad.jacobian(ne, inputs, i=0, j=0) / (xMax-xMin)
    #due_x = dde.grad.jacobian(ue, inputs, i=0, j=0) / (xMax-xMin)
    dui_x = dde.grad.jacobian(ui, inputs, i=0, j=0) / (xMax-xMin)
    
    lossb1 = dphi_xx - (ne - ni) # Poisson
    lossb2 = L*(TiTe*dni_x + Z*ni*dphi_x + ui*Source + Source_int*dui_x + Source_int*Knud) #ion momentum

    return lossb1,lossb2

geom = dde.geometry.Hypercube([0,0,0,0], [1,1,1,1])

uniform_points = geom.random_points(interiorpts[0])

points = uniform_points

net = dde.maps.FNN([4]+[32]*4+[2], "tanh", "Glorot normal")
net.apply_feature_transform(feature_transform)
net.apply_output_transform(output_transform)

data = dde.data.PDE(
    geom,
    pde,
    [],
    #num_domain=num_domain,
    #num_boundary=num_boundary,
    #num_test=2**13,
    #anchors = points
)

loss_weights = [1,1]
lossE = ["MSE"] * 2

model = dde.Model(data, net)
model.compile("SSBroyden")#,external_trainable_variables=Tewall)
#variable = dde.callbacks.VariableValue(Tewall, period=1000)
model.restore(save_path = ckpt_save_path, verbose=1)
#nw = variable.get_value()
#nwall = niMax*0.5*(1+np.tanh(nw))

xpts = np.linspace(0,1,numxpts)

def xnom(x):
    return (x - xMin) / (xMax-xMin)

def pred(xVal,MiMe,TiTe,Knud):
    MiMeNorm = (MiMe - MiMeMin ) / ( MiMeMax - MiMeMin )
    TiTeNorm = (TiTe-TiTeMin) / (TiTeMax-TiTeMin) 
    KnudNorm = (Knud - KnudMin) / (KnudMax - KnudMin)
    x = xMin + (xMax-xMin)*xVal
    X2 = np.zeros([len(xVal),4])
    X2[:,0] = xVal
    X2[:,1] = MiMeNorm
    X2[:,2] = TiTeNorm
    X2[:,3] = KnudNorm
    y2 = model.predict(X2)

    uewall = np.sqrt(MiMe/(2*np.pi))

    phi = y2[:,0]
    ni = y2[:,1]
    ne = 1/uewall*np.exp(phi)
    ui = x/L/ni
    ue = x/L/ne

    return phi,ni,ne,ui,ue

def pred1(xVal,MiMe,TiTe,Knud):
    MiMeNorm = (MiMe - MiMeMin ) / ( MiMeMax - MiMeMin )
    TiTeNorm = (TiTe-TiTeMin) / (TiTeMax-TiTeMin) 
    KnudNorm = (Knud - KnudMin) / (KnudMax - KnudMin)
    x = (xVal-xMin)/(xMax-xMin)
    X2 = np.zeros([1,4])
    X2[:,0] = x
    X2[:,1] = MiMeNorm
    X2[:,2] = TiTeNorm
    X2[:,3] = KnudNorm
    y2 = model.predict(X2)

    uewall = np.sqrt(MiMe/(2*np.pi))

    #x = xMin + (xMax-xMin)*xpts
    phi = y2[:,0]
    ni = y2[:,1]
    ne = 1/uewall*np.exp(phi)
    ui = xVal/L/ni
    ue = xVal/L/ne

    return phi,ni,ne,ui,ue

"""sampler = qmc.Sobol(d=4, scramble=True)
points = sampler.random(2e5)
y2 = model.predict(points,operator=pde)
poisson_test = y2[:,0]
ionmom_test = y2[:,1]"""

def Prhho(xVal,MiMeVal,TiTeVal,KnudVal):
    phi, ni, ne, ui, ue = pred1(xVal,MiMeVal,TiTeVal,KnudVal)

    return ui - 1


sol = optimize.root(Prhho, [L-5], args = (MiMeVal1,TiTeVal1,KnudVal1))
xSE1 = sol.x
print("sheath entrance = " + str(xSE1))

phiSE5, niSE5, neSE5, uiSE5, ueSE5 = pred1(xSE1,MiMeVal1,TiTeVal1,KnudVal1)
pp = (niSE5-neSE5)/neSE5
print("Phro:", pp)

Prho = pp

def SheathEntrance(xVal,MiMeVal,TiTeVal,KnudVal):
    phi, ni, ne, ui, ue = pred1(xVal,MiMeVal,TiTeVal,KnudVal)

    return (ni-ne) / ne - Prho

numKnudpts = 50
numMiMepts = 40
numTiTepts = 10
KnudScan = np.logspace(-4,-1,25)
KnudScan = np.append(KnudScan, np.linspace(0.1, KnudMax, 25))
MiMeScan = np.linspace(MiMeMin,MiMeMax,numMiMepts)
TiTeScan = np.linspace(TiTeMin, TiTeMax, numTiTepts)
xSEvec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
phiSEvec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
uiSEvec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
neSEvec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
MiMevec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
TiTevec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
niSEvec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
KnSEvec = np.zeros([numMiMepts, numTiTepts,numKnudpts])
#print(np.shape(phiSEvec))


phiList = []
neList = []
niList = []
ueList = []
uiList = []
    
phi, ni, ne, ui, ue = pred(xpts,MiMeVal1,TiTeVal1,KnudVal1)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

np.savetxt(save_path + "phi1.txt", phi)
np.savetxt(save_path + "ne1.txt", ne)
np.savetxt(save_path + "ni1.txt", ni)
np.savetxt(save_path + "ui1.txt", ui)
np.savetxt(save_path + "ue1.txt", ue)

phi, ni, ne, ui, ue = pred(xpts,MiMeVal2,TiTeVal2,KnudVal2)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

np.savetxt(save_path + "phi2.txt", phi)
np.savetxt(save_path + "ne2.txt", ne)
np.savetxt(save_path + "ni2.txt", ni)
np.savetxt(save_path + "ui2.txt", ui)
np.savetxt(save_path + "ue2.txt", ue)
    
phi, ni, ne, ui, ue = pred(xpts,MiMeVal3,TiTeVal3,KnudVal3)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

np.savetxt(save_path + "phi3.txt", phi)
np.savetxt(save_path + "ne3.txt", ne)
np.savetxt(save_path + "ni3.txt", ni)
np.savetxt(save_path + "ui3.txt", ui)
np.savetxt(save_path + "ue3.txt", ue)

phi, ni, ne, ui, ue = pred(xpts,MiMeVal4,TiTeVal4,KnudVal4)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

np.savetxt(save_path + "phi4.txt", phi)
np.savetxt(save_path + "ne4.txt", ne)
np.savetxt(save_path + "ni4.txt", ni)
np.savetxt(save_path + "ui4.txt", ui)
np.savetxt(save_path + "ue4.txt", ue)

"""phi, ni, ne, ui, ue = pred(xpts,MiMeVal5,TiTeVal5,KnudVal5)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue = pred(xpts,MiMeVal6,TiTeVal6,KnudVal6)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)"""



if sheathbool == True:
    # Find sheath entrance
    sol = optimize.root(SheathEntrance, [L-5], args = (MiMeVal1,TiTeVal1,KnudVal1))
    xSE1 = sol.x
    #print("sheath entrance = " + str(xSE1))
    sol = optimize.root(SheathEntrance, [L-5], args = (MiMeVal2,TiTeVal2,KnudVal2))
    xSE2 = sol.x
    #print("sheath entrance = " + str(xSE2))
    sol = optimize.root(SheathEntrance, [L-5], args = (MiMeVal3,TiTeVal3,KnudVal3))
    xSE3 = sol.x
    #print("sheath entrance = " + str(xSE3))
    sol = optimize.root(SheathEntrance, [L-5], args = (MiMeVal4,TiTeVal4,KnudVal4))
    xSE4 = sol.x
    #print("sheath entrance = " + str(xSE4))
    """sol = optimize.root(SheathEntrance, [L-5], args = (MiMeVal5,TiTeVal5,KnudVal5))
    xSE5 = sol.x

    sol = optimize.root(SheathEntrance, [L-5], args = (MiMeVal6,TiTeVal6,KnudVal6))
    xSE6 = sol.x"""
    #print("sheath entrance = " + str(xSE5))
    print("xSE1, xSE2, xSE3, xSE4:", xSE1, xSE2, xSE3, xSE4)
           
    # Compute values at sheath entrance
    phiSE1, niSE1, neSE1, uiSE1, ueSE1 = pred1(xSE1,MiMeVal1,TiTeVal1,KnudVal1)
    phiSE2, niSE2, neSE2, uiSE2, ueSE2 = pred1(xSE2,MiMeVal2,TiTeVal2,KnudVal2)
    phiSE3, niSE3, neSE3, uiSE3, ueSE3 = pred1(xSE3,MiMeVal3,TiTeVal3,KnudVal3)
    phiSE4, niSE4, neSE4, uiSE4, ueSE4 = pred1(xSE4,MiMeVal4,TiTeVal4,KnudVal4)
    #phiSE5, niSE5, neSE5, uiSE5, ueSE5 = pred1(xSE5,MiMeVal5,TiTeVal5,KnudVal5)
    #phiSE6, niSE6, neSE6, uiSE6, ueSE6 = pred1(xSE6,MiMeVal6,TiTeVal6,KnudVal6)

    X = np.zeros([1,4])
    for i in range(0,numMiMepts):
        for j in range(0,numTiTepts):  
            for k in range(0,numKnudpts):    
            
                sol = optimize.root( SheathEntrance, [L-5], args = (MiMeScan[i],TiTeScan[j],KnudScan[k]))
                xSE = sol.x
                #print("sheath:",xSE)
                phiSE, niSE, neSE, uiSE, ueSE = pred1(xSE,MiMeScan[i],TiTeScan[j],KnudScan[k])    
                phiEdge, niEdge, neEdge, uiEdge, ueEdge = pred1(L,MiMeScan[i],TiTeScan[j],KnudScan[k])  
                phiCenter, niCenter, neCenter, uiCenter, ueCenter = pred1(0,MiMeScan[i],TiTeScan[j],KnudScan[k]) 

                phiSEvec[i,j,k] = phiSE
                uiSEvec[i,j,k] = neSE*ueSE / niSE
                neSEvec[i,j,k] = neSE / niCenter
                niSEvec[i,j,k] = niSE / niCenter
                xSEvec[i,j,k] = xSE
                MiMevec[i,j,k] = MiMeScan[i]
                TiTevec[i,j,k] = TiTeScan[j]
                KnSEvec[i,j,k] = KnudScan[k] #np.sqrt(neCenter)

np.savetxt(save_path + "phiSE.txt", phiSEvec[:,:,0])

"""uiList = []
for i in range(5):
    uiList.append( neList[i] * ueList[i] / niList[i])"""

"""MiMeNorm = (MiMeVal5 - MiMeMin ) / ( MiMeMax - MiMeMin )
TiTeNorm = (TiTeVal5-TiTeMin) / (TiTeMax-TiTeMin) 
#x = (xVal-xMin)/(xMax-xMin)
X2 = np.zeros([numxpts,3])
X2[:,0] = xpts
X2[:,1] = MiMeNorm
X2[:,2] = TiTeNorm
y2 = model.predict(X2)

uewall = np.sqrt(MiMeVal5/(2*np.pi))

phit = y2[:,0]
nit = y2[:,1]
net = special.erf(L/sigma)/uewall*np.exp(phi)
uit = special.erf(xpts/sigma)/ni
uet = special.erf(xpts/sigma)/ne"""

#0 - Argon at 0 Ti/Te | xSE1 | 0 Knud
#1 - Hydrogen at 0 Ti/Te | xSE2 | 1e-2 Knud
#2 - Hydrogen at 0 Ti/Te | xSE3 | 0.3 Knud
#3 - Hydrogen at 0 Ti/Te | xSE4 | 0.07 Knud
#4 - Hydrogen at 0 Ti/Te | xSE5 #Corresponds to Bohm criterion | 0 Knud
#5 - Hydrogen at 1 Ti/Te | xSE6 | 0.3 Knud

ptsize= 100

plt.rcParams.update({'font.size': 16})
xpts = np.linspace(0,L,numxpts)
fig1, ax1 = plt.subplots(num=1,nrows=1,ncols=1, clear=True)
fig1.set_tight_layout(True)

phi = phiList[0]
ne = neList[0]
#ax1.plot((xpts-L)/np.sqrt(ne[0]), phi-phi[-1], label='$\\phi$', linestyle='-',color='black',linewidth=2)
#ax1.plot((xSE1-L)/np.sqrt(ne[0]), phiSE1-phi[-1],'ok',linewidth=2)
ax1.plot((xpts), phi, label='Hydrogen', linestyle='-',color='r',linewidth=3)
if sheathbool == True:
    ax1.scatter((xSE1), phiSE1,c='r',s=ptsize,zorder=10)

phi = phiList[1]
ne = neList[1]
#ax1.plot((xpts-L)/np.sqrt(ne[0]), phi-phi[-1], label='$\\phi$', linestyle='-',color='red',linewidth=2)
#ax1.plot((xSE2-L)/np.sqrt(ne[0]), phiSE2-phi[-1],'or',linewidth=2)
ax1.plot((xpts), phi, label='Argon', linestyle='-',color='b',linewidth=3)
if sheathbool == True:
    ax1.scatter((xSE2), phiSE2,c='b',s=ptsize,zorder=10)

"""phi = phiList[2]
ne = neList[2]
#ax1.plot((xpts-L)/np.sqrt(ne[0]), phi-phi[-1], label='$\\phi$', linestyle='-',color='blue',linewidth=2)
#ax1.plot((xSE3-L)/np.sqrt(ne[0]), phiSE3-phi[-1],'ob',linewidth=2)
ax1.plot((xpts), phi, label='Helium', linestyle='-',color='g',linewidth=2)
if sheathbool == True:
    ax1.plot((xSE3), phiSE3,'og',linewidth=2)

phi = phiList[3]
ne = neList[3]
#ax1.plot((xpts-L)/np.sqrt(ne[0]), phi-phi[-1], label='$\\phi$', linestyle='-',color='blue',linewidth=2)
#ax1.plot((xSE3-L)/np.sqrt(ne[0]), phiSE3-phi[-1],'ob',linewidth=2)
ax1.plot((xpts), phi, label='Hydrogen', linestyle='-',color='cyan',linewidth=2)
if sheathbool == True:
    ax1.plot((xSE4), phiSE4,'oc',linewidth=2)

phi = phiList[4]
ne = neList[4]
#ax1.plot((xpts-L)/np.sqrt(ne[0]), phi-phi[-1], label='$\\phi$', linestyle='-',color='blue',linewidth=2)
#ax1.plot((xSE3-L)/np.sqrt(ne[0]), phiSE3-phi[-1],'ob',linewidth=2)
ax1.plot((xpts), phi, label='H,Min Knud,cold', linestyle='-',color='k',linewidth=2)
if sheathbool == True:
    ax1.plot((xSE5), phiSE5,'ok',linewidth=2)

phi = phiList[5]
ne = neList[5]
#ax1.plot((xpts-L)/np.sqrt(ne[0]), phi-phi[-1], label='$\\phi$', linestyle='-',color='blue',linewidth=2)
#ax1.plot((xSE3-L)/np.sqrt(ne[0]), phiSE3-phi[-1],'ob',linewidth=2)
ax1.plot((xpts), phi, label='H,0.3 Knud,$T_i=T_e$', linestyle='-',color='m',linewidth=2)
if sheathbool == True:
    ax1.plot((xSE6), phiSE6,'om',linewidth=2)"""

#ax1.set_ylabel("$\\phi$")
ax1.set_xlabel("$x/\\lambda_{de}$")
ax1.set_title("$e\\phi/T$")
#ax1.set_yscale("log")
#ax1.legend()
fig1.savefig(save_path + "phi.png")

fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)

ne = neList[0]
ni = niList[0]
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ne, label='$n_e$', linestyle='--',color='black',linewidth=2)
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ni, label='$n_i$', linestyle='-',color='black',linewidth=2)
ax2.plot((xpts), ne, label='Electron', linestyle='--',color='r',linewidth=3)
ax2.plot((xpts), ni, label='Ion', linestyle='-',color='r',linewidth=3)
if sheathbool == True:
    ax2.scatter((xSE1), niSE1,c='r',s=ptsize,zorder=10)
ne = neList[1]
ni = niList[1]
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ne, label='$n_e$', linestyle='--',color='red',linewidth=2)
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ni, label='$n_i$', linestyle='-',color='red',linewidth=2)
ax2.plot((xpts), ne, linestyle='--',color='b',linewidth=3)
ax2.plot((xpts), ni, linestyle='-',color='b',linewidth=3)
if sheathbool == True:
    ax2.scatter((xSE2), niSE2,c='b',s=ptsize,zorder=10)
"""ne = neList[2]
ni = niList[2]
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ne, label='$n_e$', linestyle='--',color='blue',linewidth=2)
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ni, label='$n_i$', linestyle='-',color='blue',linewidth=2)
ax2.plot((xpts), ne, linestyle='--',color='g',linewidth=2)
ax2.plot((xpts), ni, label='He', linestyle='-',color='g',linewidth=2)
if sheathbool == True:
    ax2.plot((xSE3), niSE3,'og',linewidth=2)

ne = neList[3]
ni = niList[3]
ax2.plot((xpts), ne, linestyle='--',color='cyan',linewidth=2)
ax2.plot((xpts), ni, label='H', linestyle='-',color='cyan',linewidth=2)
if sheathbool == True:
    ax2.plot((xSE4), niSE4,'oc',linewidth=2)

ne = neList[4]
ni = niList[4]
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ne, label='$n_e$', linestyle='--',color='blue',linewidth=2)
#ax2.plot((xpts-L)/np.sqrt(ne[0]), ni, label='$n_i$', linestyle='-',color='blue',linewidth=2)
ax2.plot((xpts), ne, linestyle='--',color='k',linewidth=2)
ax2.plot((xpts), ni, label='H,Min Knud,cold', linestyle='-',color='k',linewidth=2)
if sheathbool == True:
    ax2.plot((xSE5), niSE5,'ok',linewidth=2)

ne = neList[5]
ni = niList[5]
ax2.plot((xpts), ne, linestyle='--',color='m',linewidth=2)
ax2.plot((xpts), ni, label='H,0.3 Knud,$T_i=T_e$', linestyle='-',color='m',linewidth=2)
if sheathbool == True:
    ax2.plot((xSE6), niSE6,'om',linewidth=2)"""

"""ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.set_title("Density Argon")
ax2.legend()
ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.set_title("Density Helium")
ax2.legend()"""
ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.set_title("Density")
ax2.legend()
fig2.savefig(save_path + "Densities.png")


fig3, ax3 = plt.subplots(num=3,nrows=1,ncols=1, clear=True)
fig3.set_tight_layout(True)

ax3.plot((xpts), np.ones(len(xpts)), label='$C_s$', linestyle=':',color='black',linewidth=3)

ue = ueList[0]
ui = uiList[0]
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammae/ne, label='$u_e$', linestyle='--',color='black',linewidth=2)
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammai/ni, label='$u_i$', linestyle='-',color='black',linewidth=2)
#ax3.plot((xSE1-L)/np.sqrt(ne[0]), GammaiSE1/niSE1,'ok',linewidth=2)
ax3.plot((xpts), ue, linestyle='--',color='r',linewidth=3)
ax3.plot((xpts), ui, label='Hydrogen', linestyle='-',color='r',linewidth=3)
if sheathbool == True:
    ax3.scatter((xSE1), uiSE1,c='r',s=ptsize,zorder=10)

ue = ueList[1]
ui = uiList[1]
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammae/ne, label='$u_e$', linestyle='--',color='red',linewidth=2)
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammai/ni, label='$u_i$', linestyle='-',color='red',linewidth=2)
#ax3.plot((xSE2-L)/np.sqrt(ne[0]), GammaiSE2/niSE2,'or',linewidth=2)
ax3.plot((xpts), ue, linestyle='--',color='b',linewidth=3)
ax3.plot((xpts), ui, label='Argon', linestyle='-',color='b',linewidth=3)
if sheathbool == True:
    ax3.scatter((xSE2), uiSE2,c='b',s=ptsize,zorder=10)

"""ue = ueList[2]
ui = uiList[2]
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammae/ne, label='$u_e$', linestyle='--',color='blue',linewidth=2)
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammai/ni, label='$u_i$', linestyle='-',color='blue',linewidth=2)
#ax3.plot((xSE3-L)/np.sqrt(ne[0]), GammaiSE3/niSE3,'ob',linewidth=2)
ax3.plot((xpts), ue, linestyle='--',color='g',linewidth=2)
ax3.plot((xpts), ui, label='Helium', linestyle='-',color='g',linewidth=2)
if sheathbool == True:
    ax3.plot((xSE3), uiSE3,'og',linewidth=2)

ue = ueList[3]
ui = uiList[3]
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammae/ne, label='$u_e$', linestyle='--',color='blue',linewidth=2)
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammai/ni, label='$u_i$', linestyle='-',color='blue',linewidth=2)
#ax3.plot((xSE3-L)/np.sqrt(ne[0]), GammaiSE3/niSE3,'ob',linewidth=2)
ax3.plot((xpts), ue, linestyle='--',color='cyan',linewidth=2)
ax3.plot((xpts), ui, label='Hydrogen', linestyle='-',color='cyan',linewidth=2)
if sheathbool == True:
    ax3.plot((xSE4), uiSE4,'oc',linewidth=2)

ue = ueList[4]
ui = uiList[4]
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammae/ne, label='$u_e$', linestyle='--',color='blue',linewidth=2)
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammai/ni, label='$u_i$', linestyle='-',color='blue',linewidth=2)
#ax3.plot((xSE3-L)/np.sqrt(ne[0]), GammaiSE3/niSE3,'ob',linewidth=2)
ax3.plot((xpts), ue, linestyle='--',color='k',linewidth=2)
ax3.plot((xpts), ui , label='H,Min Knud,cold', linestyle='-',color='k',linewidth=2)
if sheathbool == True:
    ax3.plot((xSE5), uiSE5,'ok',linewidth=2)

ue = ueList[5]
ui = uiList[5]
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammae/ne, label='$u_e$', linestyle='--',color='blue',linewidth=2)
#ax3.plot((xpts-L)/np.sqrt(ne[0]), Gammai/ni, label='$u_i$', linestyle='-',color='blue',linewidth=2)
#ax3.plot((xSE3-L)/np.sqrt(ne[0]), GammaiSE3/niSE3,'ob',linewidth=2)
ax3.plot((xpts), ue, linestyle='--',color='m',linewidth=2)
ax3.plot((xpts), ui, label='H,0.3 Knud,$T_i=T_e$', linestyle='-',color='m',linewidth=2)
if sheathbool == True:
    ax3.plot((xSE6), uiSE6,'om',linewidth=2)"""

#ax3.set_ylabel("$\\phi$")
ax3.set_xlabel("$x/\\lambda_{de}$")
#ax3[0,1].set_xlabel("$x/\\lambda_{de}$")
#ax3[1,0].set_xlabel("$x/\\lambda_{de}$")
ax3.set_title("Velocity [$C_s$]")
ax3.set_ylim([0,5])
#ax3[0,1].set_title("Velocity [$C_s$] Helium")
#ax3[1,0].set_title("Velocity [$C_s$] Hydrogen")
ax3.legend()
#ax3[0,1].legend()
#ax3[1,0].legend()
#ax3.set_yscale("log")
fig3.savefig(save_path + "Velocities.png")

fluchs = xpts/L

fig4, ax4 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig4.set_tight_layout(True)

ue = ueList[3]
ui = uiList[3]
ne = neList[3] 
ni = niList[3]
ax4.plot((xpts), ue*ne, label='Elec. Flux', linestyle='-',color='red',linewidth=3)
ax4.plot((xpts), ui*ni, label='Ion Flux', linestyle='-',color='blue',linewidth=3)
ax4.plot((xpts), fluchs, label="Ana Flux", color='g',linewidth=3)

#ax4.set_ylabel("$\\phi$")
ax4.set_xlabel("$x/\\lambda_{de}$")
ax4.set_title("Flux")
#ax4.set_yscale("log")
ax4.legend()
fig4.savefig(save_path + "Fluxes.png")

#Sc = np.exp(-xpts**2/sigma**2)

"""fig5, ax5 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig5.set_tight_layout(True)

#Gammae = GammaeList[0]
#Gammai = GammaiList[0]
#ne = neList[0] 
#ax5.plot((xpts), neList*Source*nref, label='Ionization Source', linestyle='-',color='red',linewidth=2)
ax5.plot((xpts), xpts/xpts/L, label='Center Source', linestyle='-',color='blue',linewidth=2)

#ax4.set_ylabel("$\\phi$")
ax5.set_xlabel("$x/\\lambda_{de}$")
ax5.set_title("Sources")
#ax4.set_yscale("log")
ax5.legend()
fig5.savefig(save_path + "Sources")"""

def fixloss(l):
    teststeps = [0]
    loss2 = [l[0]]
    for i in range(len(l)-1):
        if l[i+1] != l[i]:
            loss2.append(l[i+1])
            teststeps.append(steps[i+1])
    return np.array(teststeps),loss2

fig16, ax16 = plt.subplots(num=16,nrows=1,ncols=1, clear=True)
fig16.set_tight_layout(True)
#np.savetxt(save_path + 'Ploss.txt',loss[:,1])
ax16.plot(steps/1000, loss[:,1], label='Poisson Train Loss ', linestyle='-',color='red',linewidth=3)
ax16.plot(steps/1000, loss[:,2], label='Ion Momentum Train Loss', linestyle='-',color='blue',linewidth=3)
x,y = fixloss(loss[:,3])
ax16.scatter(steps[::5]/1000, loss[::5,3], label='Poisson Test Loss ', linestyle='-',color='r',lw=2)
x,y = fixloss(loss[:,4])
ax16.scatter(steps[::5]/1000, loss[::5,4], label='Ion Momentum Test Loss', linestyle='-',color='b',lw=2)

#ax16.set_ylabel("$\\phi$")
ax16.set_xlabel("Epochs (x1000)")
ax16.set_title("Losses")
#ax16.set_ylim([1e-12,1e4])
ax16.set_yscale("log")
ax16.legend()
fig16.savefig(save_path + "Losses.png")

xVal = 1e-2
MiMeNorm = (1836 - MiMeMin ) / ( MiMeMax - MiMeMin )
TiTeNorm = (0-TiTeMin) / (TiTeMax-TiTeMin) 
KnudNorm = (0 - KnudMin) / (KnudMax - KnudMin)
x = (xVal-xMin)/(xMax-xMin)
X2 = np.zeros([1,4])
X2[:,0] = x
X2[:,1] = MiMeNorm
X2[:,2] = TiTeNorm
X2[:,3] = KnudNorm
y2 = model.predict(X2)

uewall = np.sqrt(1836/(2*np.pi))

#x = xMin + (xMax-xMin)*xpts
phi = y2[:,0]
ni = y2[:,1]
ne = 1/uewall*np.exp(phi)
ui = xVal/L/ni
ue = xVal/L/ne

print("Phi:",phi, "ui:", ui, "ni:",ne)

phi2list = []
ni2list = []
ne2list = []
ui2list = []
ue2list = []

xpts = np.linspace(0,1,numxpts)

phi,ni,ne,ui,ue = pred(xpts,1836,0,0)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)

"""phi,ni,ne,ui,ue = pred(xpts,20,0,1e31,0)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)

phi,ni,ne,ui,ue = pred(xpts,1,0,1e31,0.3)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)

phi,ni,ne,ui,ue = pred(xpts,20,0,2e31,0)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)

phi,ni,ne,ui,ue = pred(xpts,20,0,1e31,0.3)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)

phi,ni,ne,ui,ue = pred(xpts,20,1,1e31,0)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)"""


x0 = 0.01

Source = 1/L
#Source_int = x/L
FluxAtWall = 1
#ue = Source_int / ne
#ui = Source_int / ni

def runge(MiMe, Knud):
    #MiMe = 1836
    #Te = 1
    TiTe = 0
    uewall = np.sqrt(MiMe/(2*np.pi))
    nn = 1
    #Knud = 0.3
    E0 = 0
    phi,ni,ne,ui,ue = pred1(x0,MiMe,TiTe,Knud)
    phi0 = phi[0]
    ui0 = ui[0]

    np.savetxt(save_path + "initial.txt", [phi0,ui0])
    np.savetxt(save_path + "inputs.txt", [MiMe,Knud])

    def system(x,y): 
        phi = y[0] 
        E = y[1]
        ui = y[2]
        dphi_dx = -E 
        dE_dx = Source*x/ui - 1/uewall*np.exp(phi) 
        dui_dx = E/ui - ui/x - Knud 
        return dphi_dx, dE_dx, dui_dx

    xlist = [x0] 
    philist = [phi0]
    Elist = [E0]
    uilist = [ui0]

    qwerk = scipy.integrate.RK45(system,x0,[phi0,E0,ui0],50.1,rtol = 1e-12, atol = 1e-30)
    while qwerk.status == 'running' and qwerk.y[0] > 0:
        qwerk.step()
        xlist.append(qwerk.t)
        philist.append(qwerk.y[0])
        Elist.append(qwerk.y[1])
        uilist.append(qwerk.y[2])
        #print(qwerk.status)
    ne = 1/uewall * np.exp(np.array(philist))
    ue = np.array(xlist)/L/ne
    ni = np.array(xlist)/L/np.array(uilist)
        
    return xlist, philist, uilist, Elist,ue, ne, ni

phiRKlist = []
xRKlist = []
neRKlist = []
uiRKlist = []
ueRKlist = []
niRKlist = []

xl, phi, ui, E, ue, ne, ni = runge(1836,0)
phiRKlist.append(phi)
neRKlist.append(ne)
xRKlist.append(xl)
ueRKlist.append(ue)
uiRKlist.append(ui)
niRKlist.append(ni)

"""xl, phi, ui, E, ue, ne, ni = runge(20,0,1e31,0)
phiRKlist.append(phi)
neRKlist.append(ne)
xRKlist.append(xl)
ueRKlist.append(ue)
uiRKlist.append(ui)

xl, phi, ui, E, ue, ne, ni = runge(1,0,1e31,0.3)
phiRKlist.append(phi)
neRKlist.append(ne)
xRKlist.append(xl)
ueRKlist.append(ue)
uiRKlist.append(ui)

xl, phi, ui, E, ue, ne, ni = runge(20,0,2e31,0)
phiRKlist.append(phi)
neRKlist.append(ne)
xRKlist.append(xl)
ueRKlist.append(ue)
uiRKlist.append(ui)

xl, phi, ui, E, ue, ne, ni = runge(20,0,1e31,0.3)
phiRKlist.append(phi)
neRKlist.append(ne)
xRKlist.append(xl)
ueRKlist.append(ue)
uiRKlist.append(ui)"""

#ni = ne*np.array(uelist)/(uilist)

print(len(xRKlist[0]))
xpts = np.linspace(0,L,numxpts)
fig20, ax20 = plt.subplots(num=20,nrows=1,ncols=1, clear=True)
fig20.set_tight_layout(True)
x = xRKlist[0]
phi = phi2list[0]
phiRK = phiRKlist[0]
ax20.scatter(x[-5001:][::500],phiRK[-5001:][::500],color='k',label="RK45",marker='x',s=60)
ax20.scatter([x[-250], x[-100]], [phiRK[-250], phiRK[-100]],color='k',s=60,marker='x')
ax20.plot(xpts,phi,color='k',linestyle='-',label='PINN',linewidth=2)
ax20.legend()
ax20.set_title("$e\\phi /T$")
ax20.set_xlabel("$\\lambda_{de}$")
fig20.savefig(save_path + "phiRK.png")
plt.close()

#print(x)
fig21, ax21 = plt.subplots(num=21,nrows=1,ncols=1, clear=True)
fig21.set_tight_layout(True)
ui = ui2list[0]
uiRK = uiRKlist[0]
ax21.scatter(x[-5001:][::500],uiRK[-5001:][::500],color='k',label="RK45",marker='x',s=60)
ax21.scatter([x[-250], x[-100]], [uiRK[-250], uiRK[-100]],color='k',s=60,marker='x')
ax21.plot(xpts,ui,color='k',linestyle='-',label='PINN',linewidth=2)
ax21.legend()
ax21.set_title("Velocity")
ax21.set_xlabel('$\\lambda_{De}$')
fig21.savefig(save_path + "uiRK.png")
plt.close()

#print(x)
fig22, ax22 = plt.subplots(num=22,nrows=1,ncols=1, clear=True)
fig22.set_tight_layout(True)
ue = ue2list[0]
ueRK = ueRKlist[0]
ax22.scatter(x[-5001:][::500],ueRK[-5001:][::500],color='k',label="RK45",marker='x',s=60)
ax22.scatter([x[-250], x[-100]], [ueRK[-250], ueRK[-100]],color='k',s=60,marker='x')
ax22.plot(xpts,ue,color='k',linestyle='-',label='PINN',linewidth=2)
ax22.legend()
ax22.set_title("Velocity")
ax22.set_xlabel('$\\lambda_{De}$')
fig22.savefig(save_path + "ueRK.png")
plt.close()

fig23, ax23 = plt.subplots(num=23,nrows=1,ncols=1, clear=True)
fig23.set_tight_layout(True)
ne = ne2list[0]
ni = ni2list[0]
neRK = neRKlist[0]
niRK = niRKlist[0]
ax23.scatter(x[-5001:][::500],neRK[-5001:][::500],color='b',label="ne-RK45",marker='x',s=60)
ax23.scatter([x[-250], x[-100]], [neRK[-250], neRK[-100]],color='b',s=60,marker='x')
ax23.scatter(x[-5001:][::500],niRK[-5001:][::500],color='r',label="ni-RK45",marker='x',s=60)
ax23.scatter([x[-250], x[-100]], [niRK[-250], niRK[-100]],color='r',s=60,marker='x')
ax23.plot(xpts,ne,color='b',linestyle='-',label='ne-PINN',linewidth=2)
ax23.plot(xpts,ni,color='r',linestyle='-',label='ni-PINN',linewidth=2)
ax23.legend()
ax23.set_title("Densities")
ax23.set_xlabel('$\\lambda_{De}$')
fig23.savefig(save_path + "DensitiesRK.png")
plt.close()

"""fig23, ax23 = plt.subplots(num=23,nrows=1,ncols=1, clear=True)
fig23.set_tight_layout(True)
x = xRKlist[1]
phi = phi2list[1]
phiRK = phiRKlist[1]
ax23.plot(x,phiRK,color='b',label="RK45")
ax23.plot(xpts,phi,color='g',linestyle='--',label='PINN')
ax23.legend()
ax23.set_title("Phi")
ax23.set_xlabel('$\\lambda_{De}$')
fig23.savefig(save_path + "phiRK1.png")
plt.close()

#print(x)
fig24, ax24 = plt.subplots(num=24,nrows=1,ncols=1, clear=True)
fig24.set_tight_layout(True)
ui = ui2list[1]
uiRK = uiRKlist[1]
ax24.plot(x,uiRK,color='b',label="RK45")
ax24.plot(xpts,ui,color='g',linestyle='--',label='PINN')
ax24.legend()
ax24.set_title("ui")
ax24.set_xlabel('$\\lambda_{De}$')
fig24.savefig(save_path + "uiRK1.png")
plt.close()

#print(x)
fig25, ax25 = plt.subplots(num=25,nrows=1,ncols=1, clear=True)
fig25.set_tight_layout(True)
ue = ue2list[1]
ueRK = ueRKlist[1]
ax25.plot(x,ueRK,color='b',label="RK45")
ax25.plot(xpts,ue,color='g',linestyle='--',label='PINN')
ax25.legend()
ax25.set_title("ue")
ax25.set_xlabel('$\\lambda_{De}$')
fig25.savefig(save_path + "ueRK1.png")
plt.close()

fig26, ax26 = plt.subplots(num=26,nrows=1,ncols=1, clear=True)
fig26.set_tight_layout(True)
x = xRKlist[2]
phi = phi2list[2]
phiRK = phiRKlist[2]
ax26.plot(x,phiRK,color='b',label="RK45")
ax26.plot(xpts,phi,color='g',linestyle='--',label='PINN')
ax26.legend()
ax26.set_title("Phi")
ax26.set_xlabel('$\\lambda_{De}$')
fig26.savefig(save_path + "phiRK2.png")
plt.close()

#print(x)
fig27, ax27 = plt.subplots(num=27,nrows=1,ncols=1, clear=True)
fig27.set_tight_layout(True)
ui = ui2list[2]
uiRK = uiRKlist[2]
ax27.plot(x,uiRK,color='b',label="RK45")
ax27.plot(xpts,ui,color='g',linestyle='--',label='PINN')
ax27.legend()
ax27.set_title("ui")
ax27.set_xlabel('$\\lambda_{De}$')
fig27.savefig(save_path + "uiRK2.png")
plt.close()

#print(x)
fig28, ax28 = plt.subplots(num=28,nrows=1,ncols=1, clear=True)
fig28.set_tight_layout(True)
ue = ue2list[2]
ueRK = ueRKlist[2]
ax28.plot(x,ueRK,color='b',label="RK45")
ax28.plot(xpts,ue,color='g',linestyle='--',label='PINN')
ax28.legend()
ax28.set_title("ue")
ax28.set_xlabel('$\\lambda_{De}$')
fig28.savefig(save_path + "ueRK2.png")
plt.close()

fig29, ax29 = plt.subplots(num=29,nrows=1,ncols=1, clear=True)
fig29.set_tight_layout(True)
x = xRKlist[3]
phi = phi2list[3]
phiRK = phiRKlist[3]
ax29.plot(x,phiRK,color='b',label="RK45")
ax29.plot(xpts,phi,color='g',linestyle='--',label='PINN')
ax29.legend()
ax29.set_title("Phi")
ax29.set_xlabel('$\\lambda_{De}$')
fig29.savefig(save_path + "phiRK3.png")
plt.close()

#print(x)
fig30, ax30 = plt.subplots(num=30,nrows=1,ncols=1, clear=True)
fig30.set_tight_layout(True)
ui = ui2list[3]
uiRK = uiRKlist[3]
ax30.plot(x,uiRK,color='b',label="RK45")
ax30.plot(xpts,ui,color='g',linestyle='--',label='PINN')
ax30.legend()
ax30.set_title("ui")
ax30.set_xlabel('$\\lambda_{De}$')
fig30.savefig(save_path + "uiRK3.png")
plt.close()

#print(x)
fig31, ax31 = plt.subplots(num=31,nrows=1,ncols=1, clear=True)
fig31.set_tight_layout(True)
ue = ue2list[3]
ueRK = ueRKlist[3]
ax31.plot(x,ueRK,color='b',label="RK45")
ax31.plot(xpts,ue,color='g',linestyle='--',label='PINN')
ax31.legend()
ax31.set_title("ue")
ax31.set_xlabel('$\\lambda_{De}$')
fig31.savefig(save_path + "ueRK3.png")
plt.close()

fig32, ax32 = plt.subplots(num=32,nrows=1,ncols=1, clear=True)
fig32.set_tight_layout(True)
x = xRKlist[4]
phi = phi2list[4]
phiRK = phiRKlist[4]
ax32.plot(x,phiRK,color='b',label="RK45")
ax32.plot(xpts,phi,color='g',linestyle='--',label='PINN')
ax32.legend()
ax32.set_title("Phi")
ax32.set_xlabel('$\\lambda_{De}$')
fig32.savefig(save_path + "phiRK4.png")
plt.close()

#print(x)
fig33, ax33 = plt.subplots(num=33,nrows=1,ncols=1, clear=True)
fig33.set_tight_layout(True)
ui = ui2list[4]
uiRK = uiRKlist[4]
ax33.plot(x,uiRK,color='b',label="RK45")
ax33.plot(xpts,ui,color='g',linestyle='--',label='PINN')
ax33.legend()
ax33.set_title("ui")
ax33.set_xlabel('$\\lambda_{De}$')
fig33.savefig(save_path + "uiRK4.png")
plt.close()

#print(x)
fig34, ax34 = plt.subplots(num=34,nrows=1,ncols=1, clear=True)
fig34.set_tight_layout(True)
ue = ue2list[4]
ueRK = ueRKlist[4]
ax34.plot(x,ueRK,color='b',label="RK45")
ax34.plot(xpts,ue,color='g',linestyle='--',label='PINN')
ax34.legend()
ax34.set_title("ue")
ax34.set_xlabel('$\\lambda_{De}$')
fig34.savefig(save_path + "ueRK4.png")
plt.close()

fig35, ax35 = plt.subplots(num=35,nrows=1,ncols=1, clear=True)
fig35.set_tight_layout(True)
x = xRKlist[5]
phi = phi2list[5]
phiRK = phiRKlist[5]
ax35.plot(x,phiRK,color='b',label="RK45")
ax35.plot(xpts,phi,color='g',linestyle='--',label='PINN')
ax35.legend()
ax35.set_title("Phi")
ax35.set_xlabel('$\\lambda_{De}$')
fig35.savefig(save_path + "phiRK5.png")
plt.close()

#print(x)
fig36, ax36 = plt.subplots(num=36,nrows=1,ncols=1, clear=True)
fig36.set_tight_layout(True)
ui = ui2list[5]
uiRK = uiRKlist[5]
ax36.plot(x,uiRK,color='b',label="RK45")
ax36.plot(xpts,ui,color='g',linestyle='--',label='PINN')
ax36.legend()
ax36.set_title("ui")
ax36.set_xlabel('$\\lambda_{De}$')
fig36.savefig(save_path + "uiRK5.png")
plt.close()

#print(x)
fig37, ax37 = plt.subplots(num=37,nrows=1,ncols=1, clear=True)
fig37.set_tight_layout(True)
ue = ue2list[5]
ueRK = ueRKlist[5]
ax37.plot(x,ueRK,color='b',label="RK45")
ax37.plot(xpts,ue,color='g',linestyle='--',label='PINN')
ax37.legend()
ax37.set_title("ue")
ax37.set_xlabel('$\\lambda_{De}$')
fig37.savefig(save_path + "ueRK5.png")
plt.close()"""

numMiMepts = 20
numKnudpts = 40
MiMeScan = np.linspace(MiMeMin,MiMeMax,numMiMepts)
KnudScan = np.logspace(-4, -1, 20)
KnudScan = np.append(KnudScan, np.linspace(0.1,KnudMax, 20))
print(KnudScan)
phivec = np.zeros([numMiMepts,numKnudpts])
uivec = np.zeros([numMiMepts,numKnudpts])
uevec = np.zeros([numMiMepts,numKnudpts])
nevec = np.zeros([numMiMepts,numKnudpts])
nivec = np.zeros([numMiMepts,numKnudpts])
MiMevecRK = np.zeros([numMiMepts,numKnudpts])
TiTevecRK = np.zeros([numMiMepts,numKnudpts])
KnudvecRK = np.zeros([numMiMepts,numKnudpts])
xSEvecRK = np.zeros([numMiMepts,numKnudpts])

#sheathxx = []

def sheathedge(MiMe,Knud,x):
    xl, phi, ui, E, ue, ne, ni = runge(MiMe,Knud)
    #print("x=",x[0])
    initguess = np.array([x[0] for i in range(len(xl))])
    sheathx = np.where(np.isclose(xl,initguess,rtol=5e-4,atol=1e-6))[0][0]
    #print("heat",sheathx)
    #print(xl[sheathx])
    #sheathxx.append(sheathx)
    dif = []
    dif.append((ni[sheathx]-ne[sheathx]) / ne[sheathx])
    
    #dif = np.abs((ni[sheathx:]-ne[sheathx:]) / ne[sheathx:] - Prho)
    np.savetxt(save_path + "dif.txt", dif)
    #sheathenter = np.where(min(dif)==dif)[0][0]
    return xl, phi, ui, E, ue, ne, ni,sheathx

counts = 0

for i in range(numMiMepts):
        for k in range(numKnudpts):
            sol = optimize.root(SheathEntrance, [L-5], args = (MiMeScan[i],0,KnudScan[k]))
            xSE = sol.x
            #print("sheath:",xSE)
            #phiSE, niSE, neSE, uiSE, ueSE = pred1(xSE,MiMeScan[i],TiTeScan[j],KnudScan[k])
            xl, phi, ui, E, ue, ne, ni, edgepnt = sheathedge(MiMeScan[i],KnudScan[k],xSE)
            xlc, phic, uic, Ec, uec, nec, nic = runge(MiMeScan[i],KnudScan[k])

            phivec[i,k] = phi[edgepnt]
            uivec[i,k] = ui[edgepnt]
            uevec[i,k] = ue[edgepnt]
            nivec[i,k] = ni[edgepnt] / nic[0]
            nevec[i,k] = ne[edgepnt] / nec[0]

            MiMevecRK[i,k] = MiMeScan[i]
            KnudvecRK[i,k] = KnudScan[k] 
            xSEvecRK[i,k] = xl[edgepnt]

            counts = counts +1
            if counts%100 == 0:
                print("Percent done:", counts*100/(numMiMepts*(numKnudpts)),'%')

                
fig40, ax40 = plt.subplots(num=40,nrows=1,ncols=1, clear=True)
fig40.set_tight_layout(True)


ax40.plot(KnSEvec[39,0,:], phiSEvec[39,0,:], 'b', label='Argon',linewidth=2)
ax40.plot(KnSEvec[0,0,:], phiSEvec[0,0,:], 'r', label='Hydrogen',linewidth=2)
ax40.scatter(KnudvecRK[0,::3], phivec[0,::3], color='r', label='H-RK45',marker='x',s=60)
ax40.scatter(KnudvecRK[0,::3], phivec[-1,::3], color='b', label='Ar-RK45',marker='x',s=60)
ax40.plot(KnSEvec[39,9,:], phiSEvec[39,9,:], 'b', linestyle='--',linewidth=2)
ax40.plot(KnSEvec[0,9,:], phiSEvec[0,9,:], 'r', linestyle='--',linewidth=2)
ax40.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax40.set_title("Potential Drop [$e\\phi/T$]")
ax40.set_xscale("log")
#ax8.legend()
fig40.savefig(save_path + "Potential_DropRK.png")

fig41, ax41 = plt.subplots(num=41,nrows=1,ncols=1, clear=True)
fig41.set_tight_layout(True)


ax41.plot(KnSEvec[39,0,:], uiSEvec[39,0,:], 'b', label='Ar,$T_i=0$',linewidth=2)
ax41.plot(KnSEvec[0,0,:], uiSEvec[0,0,:], 'r', label='H,$T_i=0$',linewidth=2)
ax41.scatter(KnudvecRK[0,::3], uivec[0,::3], color='r', label='H-RK45',marker='x',s=60)
ax41.scatter(KnudvecRK[0,::3], uivec[-1,::3], color='b', label='Ar-RK45',marker='x',s=60)
ax41.plot(KnSEvec[39,9,:], uiSEvec[39,9,:]*np.sqrt(1/2), 'b', label='Ar,$T_i=T_e$', linestyle='--',linewidth=2)
ax41.plot(KnSEvec[0,9,:], uiSEvec[0,9,:]*np.sqrt(1/2), 'r', label='H,$T_i=T_e$', linestyle='--',linewidth=2)

ax41.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax41.set_title("Ion Speed at Sheath Entrance [$C_s$]")
ax41.set_xscale("log")
#ax9.legend()
fig41.savefig(save_path + "Bohm_CriterionRK.png")

fig42, ax42 = plt.subplots(num=42,nrows=1,ncols=1, clear=True)
fig42.set_tight_layout(True)


ax42.plot(KnSEvec[39,0,:], L - xSEvec[39,0,:], 'b', label='Ar,cold',linewidth=2)
ax42.plot(KnSEvec[0,0,:], L - xSEvec[0,0,:], 'r', label='H,cold',linewidth=2)
ax42.scatter(KnudvecRK[0,::3], L-xSEvecRK[0,::3], c='r', label='H-RK45',marker='x',s=60)
ax42.scatter(KnudvecRK[0,::3], L-xSEvecRK[-1,::3], c='b', label='Ar-RK45',marker='x',s=60)
ax42.plot(KnSEvec[39,9,:], L - xSEvec[39,9,:], 'b', linestyle='--',label='Ar,$T_i=T_e$',linewidth=2)
ax42.plot(KnSEvec[0,9,:], L - xSEvec[0,9,:], 'r', linestyle='--',label='H,$T_i=T_e$',linewidth=2)

#ax9.set_ylabel("$\\phi$")
ax42.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
#ax42.set_ylabel(" $\\lambda_{De}$")
ax42.set_title("Sheath Width [$\\lambda_{De}$]")
ax42.set_xscale('log')
#ax11.legend()
fig42.savefig(save_path + "Sheath_WidthRK")

fig44, ax44 = plt.subplots(num=44,nrows=1,ncols=1, clear=True)
fig44.set_tight_layout(True)


ax44.plot(KnSEvec[39,0,:], niSEvec[39,0,:], 'b', label='Ar, $T_i=0$',linewidth=2)
ax44.plot(KnSEvec[0,0,:], niSEvec[0,0,:], 'r', label='H, $T_i=0$',linewidth=2)
ax44.scatter(KnudvecRK[0,::3], nivec[0,::3], c='r', label='H-RK45', marker='x', s=60)
ax44.scatter(KnudvecRK[0,::3], nivec[-1,::3], c='b', label='Ar-RK45' , marker='x', s=60)
ax44.plot(KnSEvec[39,9,:], niSEvec[39,9,:], 'b', linestyle='--',label='Ar,$T_i=T_e$',linewidth=2)
ax44.plot(KnSEvec[0,9,:], niSEvec[0,9,:], 'r',linestyle='--',label='H,$T_i=T_e$',linewidth=2)

ax44.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax44.set_title("Edge-to-Center Density")
ax44.set_xscale("log")
ax44.legend()
fig44.savefig(save_path + "Density_RatioRK.png")

fig43, ax43 = plt.subplots(num=43,nrows=1,ncols=1, clear=True)
fig43.set_tight_layout(True)

ax43.plot(TiTevec[0,:,0], phiSEvec[0,:,0], 'k', label='Te=1eV, Ti=0',linewidth=2)
ax43.scatter(0, phivec[0,0], c='k',marker='x', label='Te=1eV, Ti=0',s=60)
ax43.set_ylim([2.25,2.75])
#ax7.set_ylabel("$\\phi$")
ax43.set_xlabel("$T_{i}/T_{e}$")
ax43.set_title("Potential Drop [$e\\phi/T$]")
#ax43.set_xscale("log")
#ax8.legend()
fig43.savefig(save_path + "Potential_DropTiRK.png")

def realCs(ui, Ti):
    real = []
    for i in range(len(ui)):
        real.append(ui[i] / np.sqrt(1 + Ti[i]))
    return np.array(real)

fig45, ax45 = plt.subplots(num=45,nrows=1,ncols=1, clear=True)
fig45.set_tight_layout(True)

ax45.plot(TiTevec[0,:,0], realCs(uiSEvec[0,:,0], TiTevec[0,:,0]), 'k', label='Te=1eV, Ti=0',linewidth=2)
ax45.scatter(0, uivec[0,0],c='k', marker ='x', label='Te=1eV, Ti=0',s=60)
#ax45.set_ylim([0.5,1.1])
#ax7.set_ylabel("$\\phi$")
ax45.set_xlabel("$T_{i}/T_{e}$")
ax45.set_title("Ion Speed at Sheath Entrance [$C_s$]")
#ax7.set_xscale("log")
#ax8.legend()
fig45.savefig(save_path + "Bohm_CriteriaTiRK.png")

fig46, ax46 = plt.subplots(num=46,nrows=1,ncols=1, clear=True)
fig46.set_tight_layout(True)

ax46.plot(TiTevec[0,:,0], L-xSEvec[0,:,0], 'k', label='Te=1eV, Ti=0',linewidth=2)
ax46.scatter(0, L-xSEvecRK[0,0], c='k', marker='x', label='Te=1eV, Ti=0',s=60)
ax46.set_ylim([5,5.4])
#ax7.set_ylabel("$\\phi$")
ax46.set_xlabel("$T_{i}/T_{e}$")
ax46.set_title("Sheath Width [$\\lambda_{De}$]")
#ax7.set_xscale("log")
#ax8.legend()
fig46.savefig(save_path + "Sheath_WidthTiRK.png")

fig47, ax47 = plt.subplots(num=47,nrows=1,ncols=1, clear=True)
fig47.set_tight_layout(True)

ax47.plot(TiTevec[0,:,0], niSEvec[0,:,0], 'k', label='PINN',linewidth=2)
ax47.scatter(0, nivec[0,0], c='k', marker='x', label='RK45', s=60)
ax47.set_ylim([0.5,0.6])
#ax7.set_ylabel("$\\phi$")
ax47.set_xlabel("$T_{i}/T_{e}$")
ax47.set_title("Edge-to-Center Density")
#ax43.set_xscale("log")
ax47.legend()
fig47.savefig(save_path + "Density_ratioTiRK.png")

if sheathbool == True:

    fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
    fig7.set_tight_layout(True)

    ax7.plot(KnSEvec[39,0,:], niSEvec[39,0,:], 'b', label='Ar, $T_i=0$',linewidth=3)
    ax7.plot(KnSEvec[0,0,:], niSEvec[0,0,:], 'r', label='H, $T_i=0$',linewidth=3)
    #ax7.plot(KnSEvec[4,0,:], niSEvec[4,0,:], 'g', label='Helium')
    #ax7.plot(KnSEvec[19,0,:], niSEvec[19,0,:], 'm', label='Neon')
    #ax7.plot(KnSEvec[4,9,:], niSEvec[4,9,:], 'g',linestyle='--')
    ax7.plot(KnSEvec[39,9,:], niSEvec[39,9,:], 'b', linestyle='--',label='Ar,$T_i=T_e$',linewidth=3)
    ax7.plot(KnSEvec[0,9,:], niSEvec[0,9,:], 'r',linestyle='--',label='H,$T_i=T_e$',linewidth=3)
    #ax7.plot(KnSEvec[19,9,:], niSEvec[19,9,:], 'm', linestyle='--')
    #ax7.set_ylabel("$\\phi$")
    ax7.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax7.set_title("Density Ratio")
    ax7.set_xscale("log")
    ax7.legend()
    fig7.savefig(save_path + "Density_Ratio.png")

    fig8, ax8 = plt.subplots(num=8,nrows=1,ncols=1, clear=True)
    fig8.set_tight_layout(True)

    ax8.plot(KnSEvec[39,0,:], phiSEvec[39,0,:], 'b', label='Argon',linewidth=3)
    ax8.plot(KnSEvec[0,0,:], phiSEvec[0,0,:], 'r', label='Hydrogen',linewidth=3)
    #ax8.plot(KnSEvec[4,0,:], phiSEvec[4,0,:], 'g', label='Helium')
    #ax8.plot(KnSEvec[19,0,:], phiSEvec[19,0,:], 'm', label='Neon')
    #ax8.plot(KnSEvec[4,9,:], phiSEvec[4,9,:], 'g', linestyle='--')
    ax8.plot(KnSEvec[39,9,:], phiSEvec[39,9,:], 'b', linestyle='--',linewidth=3)
    ax8.plot(KnSEvec[0,9,:], phiSEvec[0,9,:], 'r', linestyle='--',linewidth=3)
    #ax8.plot(KnSEvec[19,9,:], phiSEvec[19,9,:], 'm', linestyle='--')
    #ax8.set_ylabel("$e\\phi/T_0$")
    ax8.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax8.set_title("Potential Drop [$e\\phi/T_0$]")
    ax8.set_xscale("log")
    #ax8.legend()
    fig8.savefig(save_path + "Potential_Drop.png")

    print(phiSEvec[0,0,0])

    fig9, ax9 = plt.subplots(num=9,nrows=1,ncols=1, clear=True)
    fig9.set_tight_layout(True)

    ax9.plot(KnSEvec[39,0,:], uiSEvec[39,0,:], 'b', label='Ar,cold',linewidth=3)
    ax9.plot(KnSEvec[0,0,:], uiSEvec[0,0,:], 'r', label='H,cold',linewidth=3)
    #ax9.plot(KnSEvec[4,0,:], uiSEvec[4,0,:], 'g', label='He,cold')
    #ax9.plot(KnSEvec[19,0,:], uiSEvec[19,0,:], 'm', label='He,cold')
    #ax9.plot(KnSEvec[4,9,:], uiSEvec[4,9,:], 'g', linestyle='--')
    ax9.plot(KnSEvec[39,9,:], uiSEvec[39,9,:], 'b', linestyle='--',linewidth=3)
    ax9.plot(KnSEvec[0,9,:], uiSEvec[0,9,:], 'r', linestyle='--',linewidth=3)
    #ax9.plot(KnSEvec[19,9,:], uiSEvec[19,9,:], 'm', linestyle='--')
    
    ax9.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax9.set_title("Ion Speed at Sheath Entrance [$C_s$]")
    ax9.set_xscale("log")
    #ax9.legend()
    fig9.savefig(save_path + "Bohm_Criterion.png")

    #Sheath_width= np.array([L-xSE1,L-xSE2,L-xSE3,L-xSE4,L-xSE5])
    fig11, ax11 = plt.subplots(num=11,nrows=1,ncols=1, clear=True)
    fig11.set_tight_layout(True)

    ax11.plot(KnSEvec[39,0,:], L - xSEvec[39,0,:], 'b', label='Ar,cold',linewidth=3)
    ax11.plot(KnSEvec[0,0,:], L - xSEvec[0,0,:], 'r', label='H,cold',linewidth=3)
    #ax11.plot(KnSEvec[4,0,:], L - xSEvec[4,0,:], 'g', label='He,cold')
    #ax11.plot(KnSEvec[19,0,:], L - xSEvec[19,0,:], 'm', label='He,cold')
    #ax11.plot(KnSEvec[4,9,:], L - xSEvec[4,9,:], 'g', linestyle='--')
    ax11.plot(KnSEvec[39,9,:], L - xSEvec[39,9,:], 'b', linestyle='--',label='Ar,$T_i=T_e$',linewidth=3)
    ax11.plot(KnSEvec[0,9,:], L - xSEvec[0,9,:], 'r', linestyle='--',label='H,$T_i=T_e$',linewidth=3)
    #ax11.plot(KnSEvec[19,9,:], L - xSEvec[19,9,:], 'm', linestyle='--')
    
    ax11.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax11.set_title("Sheath Width")
    ax11.set_xscale("log")
    #ax11.legend()
    fig11.savefig(save_path + "Sheath_Width")

