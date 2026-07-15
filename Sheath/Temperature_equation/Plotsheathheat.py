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

save_path = '/blue/cmcdevitt/ewebb2/Sheath/models/3000_latest/Figures/'

#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2025-06-04-11:37:27/'
#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2025-05-19-15:49:09/'
#Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/2025-06-05-10:52:14/'
Data_path = '/blue/cmcdevitt/ewebb2/Sheath/models/3000_latest/'
#34888
ckpt_save_path = Data_path + "model.pt-25033.pt"
#soln = np.loadtxt('./data/solutionAnalytic0_fine.dat')
#soln = np.loadtxt(Data_path + 'data/solution0_fine.dat')
trainpts = np.loadtxt(Data_path + 'data/train.dat')
loss = np.loadtxt(Data_path + 'data/loss.dat')
#nw = np.loadtxt(Data_path + 'data/nt.txt')

dde.config.set_default_float("float64")

numxpts = 1000 # only used when model is loaded

interiorpts = [2,2]

KnudVal1 = 0.0
KnudVal2 = 0.2
KnudVal3 = 0.3

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
KnudMin = 0.1
KnudMax = 0.3
Knud = 0

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
                                    #nr is significantly more sensitive to S0, than temperature
                                    #1 order of magnitude of S0 change, results in roughly half an order
                                    #of magnitude change in nr in the same direction. 
                                    #(i.e. more source means higher density)

Epsnormsq = Eps0**2*Ts**2/(q**2*xr*nr)

Tewall = dde.Variable(-0.219, dtype=torch.float64)
#Tewall = torch.tensor(C, dtype=torch.float64, requires_grad=True)


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

def log10(y):
    #top = torch.log(y)
    #bot = torch.log(torch.constant(10,dtype=top.dtype))
    return torch.log10(y)

def egyrofreq(B):
    #B in Tesla
    return 1.76e3*B

def coulog(n,T):
    #n in m^-3 and T in eV
    n_dim = n*nr/1e6 #convert to cm^-3
    T_dim = T*Ts
    """if T <= 50:
        return 23.4 - 1.15*tf.log(n) + 3.45*tf.log(T)
    else:
        return 25.3 - 1.15*torch.log10(n_dim) + 2.3*torch.log10(T_dim)"""
    return 23.4 - 1.15*torch.log10(n_dim) + 3.45*torch.log10(T_dim)

def coulogd(n,T):
    #n in m^-3 and T in eV
    n_dim = n*nr/1e6 #convert to cm^-3
    T_dim = T*Ts
    """if T <= 50:
        return 23.4 - 1.15*tf.log(n) + 3.45*tf.log(T)
    else:
        return 25.3 - 1.15*np.log10(n_dim) + 2.3*np.log10(T_dim)"""
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
    #me = 9.109e-28 #g
    e = 4.8032e-10 #statC
    k = 1.3807e-16 #erg K^-1
    #Ts = 11600 #K
    Z=1
    ni_dim = ni*nr#/1e6 #convert to cm^-3
    T_dim = T*Ts #this can stay eV
    #return 0.75*torch.sqrt(me/(2*torch.pi)) / (coulog(ni/1e6,Te)*e**4)*(k*Ts)**(3/2) * (Te_dim**(3/2)/(Z**2*ni_dim)) #s
    #return 3.5e5/coulog(ni,Te) * torch.sqrt(Te_dim**3)/(ni_dim*Z**2) #s
    #return 12*np.pi**(1.5)/np.sqrt(2) * np.sqrt(me)*Eps0**2*T_dim**(3/2) / (q**(2.5)*coulog(ni,T)*ni_dim*Z**2)*ur/xr #s
    return 12*np.pi**(1.5)/np.sqrt(2) * np.sqrt(1/MiMe)*Epsnormsq*T**(3/2) / (coulog(ni,T)*ni*Z**2) #s

def Taued(ni,T):
    #me = 9.109e-28 #g
    e = 4.8032e-10 #statC
    k = 1.3807e-16 #erg K^-1
    #Ts = 11600 #K
    Z=1
    ni_dim = ni*nr#/1e6 #convert to cm^-3
    T_dim = T*Ts #this can stay eV
    #return 0.75*torch.sqrt(me/(2*torch.pi)) / (coulog(ni/1e6,Te)*e**4)*(k*Ts)**(3/2) * (Te_dim**(3/2)/(Z**2*ni_dim)) #s
    #return 3.5e5/coulog(ni,Te) * torch.sqrt(Te_dim**3)/(ni_dim*Z**2) #s
    #return 12*np.pi**(1.5)/np.sqrt(2) * np.sqrt(me)*Eps0**2*T_dim**(3/2) / (q**(2.5)*coulogd(ni,T)*ni_dim*Z**2) * ur/xr #s
    return 12*np.pi**(1.5)/np.sqrt(2) * np.sqrt(1/MiMe)*Epsnormsq*T**(3/2) / (coulogd(ni,T)*ni*Z**2) #s

def boundaryRight(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

def boundaryLeft(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


print(f"Taued = {Taued(0.5,0.5):.2e} s")
x = torch.tensor([[1, 2, 0.5]], dtype=torch.float64)
print("Taue = ",Taue(x,x), "s")

def feature_transform(inputs):
    xNorm = inputs[:,0:1]
    xSq = xNorm * xNorm
    #return torch.cat((xSq),dim=1)
    return xSq

def output_transform(inputs, outputs):
    xNorm = inputs[:,0:1]

    x = xNorm*(xMax-xMin) + xMin

    TewallF = 0.6*0.5*(1+torch.tanh(Tewall)) +0.00001
    uewall = torch.sqrt(MiMe*TewallF/(2*np.pi))
    newall = 1/uewall
    phiFinal = (L-x)*(L+x)/L**2 * (outputs[:,0:1])
    #phiFinal = (L-x)*(L+x)/L**2 * 5*(1+torch.tanh(outputs[:,0:1]))
    #phiFinal = (0.5*(1+torch.tanh(2.5-x/10))+0.5*(torch.tanh(2.5+x/10)-1)) * 5*(1+torch.tanh(outputs[:,0:1]))
    niFinal  = torch.log( 1 + torch.exp(outputs[:, 1:2]) )
    #niFinal = torch.clamp(outputs[:, 1:2], min=0.0, max=None) 
    neFinal = (L-x)*(L+x)/L**2 * torch.log(1+torch.exp(outputs[:,2:3])) + newall
    TeFinal = (L-x)*(L+x)/L**2*0.6*0.5*(1+torch.tanh(outputs[:,3:4])) + (x/L)**2*TewallF
    #TeFinal = torch.log(1 + torch.exp(outputs[:,3:4]))
    #TeFinal = (L-x)*(L+x)/L**2 *1/2*(1+torch.tanh(outputs[:,3:4]))
    #TeFinal = 0.6 + torch.log(0.548811636095+torch.exp(outputs[:,3:4]))
    #TiFinal = 0.6 + torch.log(1+torch.exp(outputs[:,4:5])-(1-np.exp(0.548811636094)))


    return torch.cat((phiFinal,niFinal,neFinal,TeFinal), dim=1)


def pde(inputs, outputs):
    phi, ni, ne, Te = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:,3:4]#, outputs[:,4:5]
    #phi, ni, ne = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

    xNorm = inputs[:, 0:1]
    #xNorm,KnudNorm = inputs[:,0:1], inputs[:,1:2]
    
    x = xMin + ( xMax - xMin ) * xNorm
    #Knud = KnudMin + (KnudMax - KnudMin) * KnudNorm

    dphi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (xMax-xMin)
    dphi_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0) / (xMax-xMin)**2

    #dne_x = dde.grad.jacobian(outputs, inputs, i=2, j=0) / (xMax-xMin)
    #dni_x = dde.grad.jacobian(outputs, inputs, i=1, j=0) / (xMax-xMin)

    #uewall = np.sqrt(MiMe/(2*np.pi))
    #coordinate transforms
    #ne = 1/uewall*tf.exp(phi)
    #ue = tf.exp(uet)-1
    #nwt = ln(nw)
    #nw = tf.exp(nwt)
    #net = tf.exp(ne)
    #ne = tf.log(net)
    
    source = x/L
    #source = n*u = x/L
    ui = source/ni
    ue = source/ne

    due_x = dde.grad.jacobian(ue,inputs, i=0, j=0) / (xMax-xMin)
    dui_x = dde.grad.jacobian(ui,inputs,i=0,j=0) / (xMax-xMin)
    dne_x = dde.grad.jacobian(ne,inputs, i=0, j=0) / (xMax-xMin)
    dni_x = dde.grad.jacobian(ni,inputs, i=0, j=0) / (xMax-xMin)
    dTe_x = dde.grad.jacobian(Te,inputs, i=0, j=0) / (xMax-xMin)
    #dTi_x = dde.grad.jacobian(Ti,inputs, i=0, j=0) / (xMax-xMin)
    dTe_xx = dde.grad.hessian(Te,inputs,component=0, i=0, j=0) / (xMax-xMin)**2
    #due_x = (ne - x*dne_x)/(L*ne**2)

    #qe = -30692/coulog(ne,Te)*Te**2*tf.sqrt(Te)*dTe_x*np.sqrt(MiMe)*Ts**2/(xr*nr) #conduction
    #q = 7*Te*ne*ue
    qe = 0.71*ne*Te*(ue-ui) - 3.16*ne*Te*MiMe*dTe_x*Taue(ni,Te) #conduction
    #qe=0
    #dq_x = 0
    #q1 = 0.71*(source*dTe_x + Te/L - ne*ui*dTe_x - ne*Te*dui_x-Te*ui*dne_x)
    #q2 = 3.16*MiMe*(ne*Te*Taue(ni,Te)*dTe_xx*ur/xr + ne*Taue(ni,Te)*dTe_x**2 *ur/xr + Te*Taue(ni,Te)*dne_x*dTe_x*ur/xr + 
    #    12*torch.pi**(1.5)*np.sqrt(1/MiMe)*Epsnormsq/(np.sqrt(2)*coulog(ne,Te)*Z**2) * (3/2*ne*Te**(3/2)*dTe_x**2/ni - ne*Te**(5/2)*dni_x*dTe_x/ni**2))
    #K0 = -30692/coulog(ne,Te)
    #dq_x = q1 - q2 

    #qi = -3.9*ni*Ti*2**(0.5)*Taue(ni,Ti)*ur/xr*dTi_x

    dq_x = dde.grad.jacobian(qe, inputs, i=0,j=0) / (xMax-xMin)
    #dq_x = (5/2*K0*Te**(3/2)*dTe_x**2 + K0*Te**(5/2)*dTe_xx)*MiMe*ur/(nr*xr) #conduction
    #dqi_x = dde.grad.jacobian(qi, inputs, i=0, j=0) / (xMax-xMin)

    meoverme = 1

    #Rue = -meoverme*(ue-ui)*nuei(ni,ne,Te) * xr/(ur*nr) #momentum loss due to collisions with ions 
    Rue = -meoverme*(ue-ui)*ne/Taue(ni,Te) * 0.51
    #RTe = -0.71*ne*MiMe*dTe_x                       #momentum loss due to hot electrons pushing ions towards increasing Te
    RTe = -0.71*ne*MiMe*dTe_x                     

    #QRe = (Rue + RTe)/MiMe*(ue-ui)
    QRe = -(Rue + RTe)*(ue-ui)                  #joule heating due to net drift of electrons against the dissipative force
    #Qeqe = -3/MiMe*nueq(ni,ne,Te)*(Te-Ti)* xr/(ur*nr)    #heat exchange due to thermal equilibrium ion-electron collisions
    Qeqe = -3/MiMe*ne/Taue(ni,Te)*(Te-Ti)  

    #Rui = 1/MiMe*(ue-ui)*nuei(ni,ne,Te) * xr/(ur*nr)    #momentum gain due to ion-electron collisions
    Rui = 1/MiMe*(ue-ui)*ne/Taue(ni,Te) 
    #RTi = 0.71*ne*dTe_x
    RTi = 0.71*ne*dTe_x

    #QRi = (Rui + RTi)*(ue-ui)
    #Qeqi = 3/MiMe*nueq(ni,ne,Te)*(Te-Ti)* xr/(ur*nr)

    lossb1 = dphi_xx - (ne - ni) # Poisson
    #lossb2 = (L*ne*due_x + L*ue*dne_x - 1 ) # electron continuity
    #lossb3 = source*due_x + MiMe*ne*dTe_x + MiMe*Te*dne_x - MiMe*ne*dphi_x + ue/(L) #electron momentum
    lossb3 = (ne*dTe_x + Te*dne_x - ne*dphi_x)*MiMe - Rue - RTe # + ue/(L*MiMe))*MiMe #electron momentum
    lossb5 = L*ni*dphi_x + ui + L*source*dui_x + L*source*Knud - Rui - RTi #ion momentum
    #lossb5 = L*ni*dTi_x + L*Ti*dni_x + L*ni*dphi_x + ui + L*source*dui_x + L*source*Knud - L*Rui - L*RTi #ion momentum
    #lossb6 = 3/2*x*dTe_x + L*ne*Te*due_x + L*dq_x - 1/(2)*(3-3*Te+1/MiMe*(ue**2)) #- 1/L 
    lossb6 = 3/2*x*dTe_x + L*ne*Te*due_x + L*dq_x - 1/(2)*(3-3*Te+1/MiMe*(ue**2)) - QRe - Qeqe #- ue*(Rue+RTe)  #- 1/L #- L*1/(2)*source*ue*Knud
                           #^convection^
    #lossb6 = 3/2*x*dTe_x + L*ne*Te*due_x + dq_x- 1/(2*L)*(3-3*Te+1/MiMe*(ue**2))
    #lossb6 = -3.16*MiMe*(ne*Te*Taue(ni,Te)*dTe_x*ur/xr) 
    #loss7  = 3/2*x*dTi_x/L + ni*Ti*dui_x + dqi_x - 1/(2*L)*(3-3*Ti+(ui**2)) + Qeqe + ui*source*Knud
    #could the source term be incorrect?

    return lossb1,lossb3,lossb5,lossb6#, loss7/1e1

geom = dde.geometry.Hypercube([0,0], [1,1])

uniform_points = geom.random_points(interiorpts[0])

points = uniform_points

net = dde.maps.FNN([1]+[32]*3+[5], "tanh", "Glorot normal")
net.apply_feature_transform(feature_transform)
net.apply_output_transform(output_transform)

def func(inputs,outputs):
    #uewall = np.sqrt(MiMe*outputs[:,3:4]/(2*np.pi))
    newall = outputs[:,2:3]
    niwall = outputs[:,1:2]
    uewall = 1/newall
    uiwall = 1/niwall
    return (3/2 - 5/2*outputs[:,3:4] - 1/2*1/MiMe*uewall**2 - 1/2*(uiwall)**2-0.71*newall*outputs[:,3:4]*(uewall-uiwall))/( -3.16*MiMe*newall*outputs[:,3:4]*Taue(niwall,outputs[:,3:4]) )
    #return -1
    #return (3/2 - 5/2*outputs[:,3:4])/( -3.16*MiMe*newall*outputs[:,3:4]*Taue(niwall,outputs[:,3:4]) )
    #try: (3/2 - 5/2*outputs[:,3:4])/( -3.16*MiMe*newall*outputs[:,3:4]*Taue(niwall,outputs[:,3:4]) )

"""bc_wall = dde.icbc.RobinBC(
    geom,
    func,#lambda x,y: y[:,0:1],
    boundaryRight,
    component=0,
)"""

#bc_c = dde.icbc.NeumannBC(geom, lambda x: 0, boundaryLeft)
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
#trainloss += loss[:,6]
#trainloss += loss[:,7]

testloss = loss[:,6]
testloss = testloss + loss[:,7]
testloss = testloss + loss[:,8]
testloss = testloss + loss[:,9]
testloss = testloss + loss[:,10]
#testloss = testloss + loss[:,10]
#testloss += loss[:,13]
#testloss += loss[:,14]
#testloss += loss[:,15]
#nT=dde.Variable(1.0)

xpts = np.linspace(0,1,numxpts)

def xnom(x):
    return (x - xMin) / (xMax-xMin)

def pred(xVal,Knud):
    x = xMin + (xMax-xMin)*xVal
    KnudVal = (Knud - KnudMin)/(KnudMax-KnudMin)

    X2 = np.zeros([len(xVal),1])
    X2[:,0] = xVal
    #X2[:,1] = KnudVal
    y2 = model.predict(X2)

    phi = y2[:,0]
    ni = y2[:,1]
    ne = y2[:,2]
    Te = y2[:,3]
    #Te = [1 for i in range(len(xVal))]
    #Ti = y2[:,4]
    Ti = 0.026/3
    ui = x/L/ni
    ue = x/L/ne

    return phi,ni,ne,ui,ue,Te,Ti

def pred1(xVal,Knud):
    x = xMin + (xMax-xMin)*xVal
    KnudVal = (Knud - KnudMin)/(KnudMax-KnudMin)

    X2 = np.zeros([1,1])
    X2[:,0] = xVal
    #X2[:,1] = KnudVal
    y2 = model.predict(X2)

    phi = y2[:,0]
    ni = y2[:,1]
    ne = y2[:,2]
    Te = y2[:,3]
    #Te = [1 for i in range(len(xVal))]
    #Ti = y2[:,4]
    Ti = 0.026/3
    ui = x/L/ni
    ue = x/L/ne

    return phi,ni,ne,ui,ue,Te,Ti

phiList = []
neList = []
niList = []
ueList = []
uiList = []
TeList = []
TiList = []

phi, ni, ne, ui, ue, Te, Ti = pred(xpts,KnudVal1)
phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)
TiList.append(Ti)
TeList.append(Te)

phi, ni, ne, ui, ue, Te, Ti = pred(xpts,KnudVal2)
phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)
TiList.append(Ti)
TeList.append(Te)

phi, ni, ne, ui, ue, Te, Ti = pred(xpts,KnudVal3)
phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)
TiList.append(Ti)
TeList.append(Te)

xpts = np.linspace(0,L,numxpts)

def derivative(y):
    dy = np.gradient(y, xpts)
    return dy

def func(ni,Te):
    ui = 1/ni
    uewall = np.sqrt(MiMe*Te/(2*np.pi))
    newall = 1/uewall
    return (3/2 - 5/2*Te - 1/2*1/MiMe*uewall**2 - 1/2*ui**2-0.71*newall*Te*(uewall-ui))/( -3.16*MiMe*newall*Te*Taued(ni,Te) )

def Prhho(xVal, KnudVal):
    phi, ni, ne, ui, ue, Te, Ti = pred1(xVal,KnudVal)

    return ui - np.sqrt(Te)

sol = optimize.root(Prhho, [(L-5)/L], args = (KnudVal1))
xSE1 = sol.x
print("sheath entrance Bohm = " + str(xSE1))

phiSE1, niSE1, neSE1, uiSE1, ueSE1, TeSE1, TiSE1 = pred1(xSE1,KnudVal1)
pp = (niSE1-neSE1)/neSE1
print("Phro:", pp)

Prho = 0.0652

def SheathEntrance(xVal,KnudVal):
    phi, ni, ne, ui, ue, Te, Ti = pred1(xVal,KnudVal)

    return (ni-ne) / ne - Prho


sol2 = optimize.root(SheathEntrance, [(L-5)/L], args = (KnudVal1))
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

"""phi = phiList[1]
ax1.plot((xpts), phi, label='$\\phi$', linestyle='-',color='b',linewidth=3)

phi = phiList[2]
ax1.plot((xpts), phi, label='$\\phi$', linestyle='-',color='g',linewidth=3)"""

ax1.set_xlabel("$x/\\lambda_{de}$")
ax1.set_title("$e\\phi/T_s$")
fig1.savefig(save_path + "phi.png")

fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)

ne = neList[0]
ni = niList[0]
ax2.plot((xpts), ni, label='$n_i$', linestyle='-',color='b',linewidth=3)
ax2.plot((xpts), ne, label='$n_e$', linestyle='--',color='r',linewidth=3)


"""ne = neList[1]
ni = niList[1]
ax2.plot((xpts), ne, label='$n_e$', linestyle='--',color='b',linewidth=3)
ax2.plot((xpts), ni, label='$n_i$', linestyle='-',color='b',linewidth=3)

ne = neList[2]
ni = niList[2]
ax2.plot((xpts), ne, label='$n_e$', linestyle='--',color='g',linewidth=3)
ax2.plot((xpts), ni, label='$n_i$', linestyle='-',color='g',linewidth=3)"""

ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.legend()
ax2.set_title("Density")
fig2.savefig(save_path + "Densities.png")


fig3, ax3 = plt.subplots(num=3,nrows=1,ncols=1, clear=True)
fig3.set_tight_layout(True)

#ax3.plot((xpts), np.ones(len(xpts)), label='$C_s$', linestyle=':',color='black',linewidth=3)
Te = TeList[0]
ax3.plot(xpts, np.sqrt(Te), label='$C_s$', linestyle=':',color='k',linewidth=3)

"""Te = TeList[1]
ax3.plot(xpts, np.sqrt(Te), label='$C_s$', linestyle=':',color='b',linewidth=3)

Te = TeList[2]
ax3.plot(xpts, np.sqrt(Te), label='$C_s$', linestyle=':',color='g',linewidth=3)"""

ue = ueList[0]
ui = uiList[0]
ax3.plot((xpts), ui, label='Ion', linestyle='-',color='b',linewidth=3)
ax3.plot((xpts), ue, label='electron',linestyle='--',color='r',linewidth=3)
"""
ue = ueList[1]
ui = uiList[1]
ax3.plot((xpts), ui, label='Ion', linestyle='-',color='b',linewidth=3)
ax3.plot((xpts), ue, label='electron',linestyle='--',color='b',linewidth=3)

ue = ueList[2]
ui = uiList[2]
ax3.plot((xpts), ui, label='Ion', linestyle='-',color='g',linewidth=3)
ax3.plot((xpts), ue, label='electron',linestyle='--',color='g',linewidth=3)"""

ax3.set_xlabel("$x/\\lambda_{De}$")
ax3.set_title("Velocity [$C_s$]")          
ax3.set_ylim([0,5])
ax3.legend()
#ax3.set_yscale("log")
fig3.savefig(save_path + "Velocities.png")

#xxpt = xpts/sigma
fluchs = xpts/L

fig4, ax4 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig4.set_tight_layout(True)

ue = ueList[0]
ui = uiList[0]
ne = neList[0] 
ni = niList[0]

#ax4.plot((xpts), ue*ne, label='Ion Flux', linestyle='-',color='r',linewidth=3)
ax4.plot((xpts), ui*ni, label='Electron Flux', linestyle='-',color='b',linewidth=3)
ax4.plot((xpts), fluchs, label="Ana Flux",linestyle='--', color='k',linewidth=3)


ax4.set_xlabel("$x/\\lambda_{de}$")
ax4.set_title("Flux")
fig4.savefig(save_path + "Flux")

fig5, ax5 = plt.subplots(num=5,nrows=1,ncols=1, clear=True)
fig5.set_tight_layout(True)

Te = TeList[0]
#Ti = TiList[0]
ax5.plot((xpts), Te, label='$T_e$', linestyle='-',color='r',linewidth=3)
#ax5.plot((xpts), Ti, label='$T_i$', linestyle='--',color='r',linewidth=3)

"""Te = TeList[1]
#Ti = TiList[1]
ax5.plot((xpts), Te, label='$T_e$', linestyle='-',color='b',linewidth=3)
#ax5.plot((xpts), Ti, label='$T_i$', linestyle='--',color='b',linewidth=3)

Te = TeList[2]
#Ti = TiList[2]
ax5.plot((xpts), Te, label='$T_e$', linestyle='-',color='g',linewidth=3)
#ax5.plot((xpts), Ti, label='$T_i$', linestyle='--',color='g',linewidth=3)"""

#ax5.legend()
ax5.set_xlabel("$x/\\lambda_{de}$")
ax5.set_title("Temperature ($T/T_s$)")
fig5.savefig(save_path + "Te")

print("drift velocity = ",ueList[0][200]-uiList[0][200] )

"""fig6, ax6 = plt.subplots(num=6,nrows=1,ncols=1, clear=True)
fig6.set_tight_layout(True)

ax6.plot(xpts[:500], ueList[0][:500]-uiList[0][:500], label='Drift Velocity', linestyle='-',color='r',linewidth=2)

ax6.set_xlabel("$x/\\lambda_{de}$")
ax6.set_title("Drift Velocity")
ax6.legend()
fig6.savefig(save_path + "DriftVelocity")
plt.close()"""

def func2(x,ne,ni,ue,ui,Te):
    return (3/2*x/L - 5/2*Te*ne*ue - 1/2*1/MiMe*ne*ue**3 - 1/2*ni*(ui)**3-0.71*ne*Te*(ue-ui))/( -3.16*MiMe*ne*Te*Taued(ni,Te) )

"""check_val = -2

print(f"Check = {func2(xpts[check_val],neList[0][check_val],niList[0][check_val],ueList[0][check_val],uiList[0][check_val],TeList[0][check_val]):.2e}")
print(xpts[check_val])
print(neList[0][check_val])
print(niList[0][check_val])
print(ueList[0][check_val])
print(uiList[0][check_val])
#print(np.where(TeList[0]==min(TeList[0]))[0][0]) 
print(TeList[0][check_val])    
print(Taued(niList[0][check_val],TeList[0][check_val]))"""

def qtote(x,ne,ni,ue,ui,T):
    qconv_th = 5/2*ne*T*ue
    qconv_KE = 1/2/MiMe*ne*ue**3
    qconv = qconv_th + qconv_KE
    dTe_x = func2(x,ne,ni,ue,ui,T) #derivative(T) 
    #dTe_x[-1] = dTe_x[-1]/2
    np.savetxt(save_path + "dTe_x.txt", dTe_x)
    qcond = - 3.16*ne*T*MiMe*dTe_x*Taued(ni,T) + 0.71*ne*T*(ue-ui)
    qtot = qconv + qcond
    return qconv, qcond, qtot

def qtoti(n,ui,T):
    qconv_th = 5/2*n*T*ui
    qconv_KE = 1/2*n*ui**3
    qconv = qconv_th + qconv_KE
    dTe_x = 0 #derivative(T)
    qcond = 0#-3.16*n*T*MiMe*dTe_x*Taued(n,T)
    qtot = qconv + qcond
    return qconv, qcond, qtot

def Heat_Transmission(x):
    phi, ni, ne, ui, ue, Te, Ti = pred1(x,KnudVal1)
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

"""Te = TeList[1]
ne = neList[1]
ue = ueList[1]

ax6.plot((xpts), qconv(ne,ue,Te), label='Convective', linestyle='-',color='b',linewidth=3)
ax6.plot((xpts), qcond(ne,Te), label='Conductive', linestyle='--',color='b',linewidth=3)

Te = TeList[2]
ne = neList[2]
ue = ueList[2]

ax6.plot((xpts), qconv(ne,ue,Te), label='Convective', linestyle='-',color='g',linewidth=3)
ax6.plot((xpts), qcond(ne,Te), label='Conductive', linestyle='--',color='g',linewidth=3)"""

ax6.set_xlabel("$x/\\lambda_{de}$")
#ax6.set_yscale('log')
ax6.legend()
ax6.set_title("Heat Flux")
fig6.savefig(save_path + "q")

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

"""Te = TeList[1]
ne = neList[1]
ue = ueList[1]
cs1 = ax7[0,1].contourf(ne, Te, Tau(ne,Te).T, levels=50, cmap='jet')

Te = TeList[2]
ne = neList[2]
ue = ueList[2]
cs2 = ax7[1,0].contourf(ne, Te, Tau(ne,Te).T, levels=50, cmap='jet')"""

ax7.set_xlabel("$n_e$")
ax7.set_ylabel("$T_e$")
fig7.colorbar(cs, ax=ax7)
ax7.set_title("$\\tau$")# Kn=" + str(KnudVal1))

"""ax7[0,1].set_xlabel("$n_e$")
ax7[0,1].set_ylabel("$T_e$")
fig7.colorbar(cs1, ax=ax7[0,1])
ax7[0,1].set_title("$\\tau$ Kn=" + str(KnudVal2))

ax7[1,0].set_xlabel("$n_e$")
ax7[1,0].set_ylabel("$T_e$")
fig7.colorbar(cs2, ax=ax7[1,0])
ax7[1,0].set_title("$\\tau$ Kn=" + str(KnudVal3))"""
fig7.savefig(save_path + "Tau")
plt.close()

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

ax16.plot(steps/1000, loss[:,1], label='Poisson', linestyle='-',color='red',linewidth=3)
ax16.plot(steps/1000, loss[:,2], label='Elec Momentum', linestyle='-',color='blue',linewidth=3)
ax16.plot(steps/1000, loss[:,3], label='Ion Momentum', linestyle='-',color='k',linewidth=3)
ax16.plot(steps/1000, loss[:,4], label='Elec Energy', linestyle='-',color='g',linewidth=3)
ax16.plot(steps/1000, loss[:,5], label='Boundary Loss', linestyle='-',color='c',linewidth=3)
#ax16.plot(steps/1000, loss[:,5], label='EE Train Loss', linestyle='-',color='c',linewidth=3)
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
Knudpts = np.linspace(0,1,10)

xnew, Knudnew = np.meshgrid(xpts,Knudpts)
X2 = np.vstack((np.ravel(xnew),np.ravel(Knudnew))).T
y2 = model.predict(X2,operator=pde)
#print(y2.shape)
resphi = y2[0]
resem = y2[1]
resim = y2[2]
resee = y2[3]
#resBC = y2[4]

resphi = resphi.reshape(xnew.shape)
resem = resem.reshape(xnew.shape)
resim = resim.reshape(xnew.shape)
resee = resee.reshape(xnew.shape)
#resBC = resBC.reshape(xnew.shape)
#resphinew = resphi.T
#resecnew = resec.T
#resimnew = resim.T
#print(resphinew.shape)
#print(resphi.shape)
fig17,ax17 = plt.subplots(num=17,nrows=1,ncols=1, clear=True)
fig17.set_tight_layout(True)
ax17.plot(xpts,resphi[9,:],c='k')

fig17.savefig(save_path + 'Residual_phi.png') 

fig18,ax18 = plt.subplots(num=18,nrows=1,ncols=1, clear=True)
fig18.set_tight_layout(True)
ax18.plot(xpts,resem[9,:],c='k')

fig18.savefig(save_path + 'Residual_em.png') 

fig19,ax19 = plt.subplots(num=19,nrows=1,ncols=1, clear=True)
fig19.set_tight_layout(True)
ax19.plot(xpts,resim[9,:],c='k')

fig19.savefig(save_path + 'Residual_im.png') 

fig20,ax20 = plt.subplots(num=20,nrows=1,ncols=1, clear=True)
fig20.set_tight_layout(True)
ax20.plot(xpts,resee[9,:],c='k')

fig20.savefig(save_path + 'Residual_ee.png') 


"""fig26,ax26 = plt.subplots(num=26,nrows=1,ncols=1, clear=True)
fig26.set_tight_layout(True)
ax26.plot(xpts,resie[9,:],c='k')
fig26.savefig(save_path + 'Residual_ie.png')"""

"""def nueid(ni,ne,T):
    ni_dim = ni*nr
    ne_dim = ne*nr
    T_dim = T*Ts
    return 0.468e-16*coulogd(ne_dim,T_dim)*ni_dim*ne_dim/((T_dim/1000)*np.sqrt(T_dim/1000)) #m^-3 s^-1

def nueqd(ni,ne,T):
    ni_dim = ni*nr
    ne_dim = ne*nr
    T_dim = T*Ts
    return 2.9e-12 * ni_dim*ne_dim*coulogd(ne_dim,T_dim)/((T_dim)*np.sqrt(T_dim)) #m^-3 s^-1

xxpts = np.linspace(0,1,numxpts)
x0 = 0.01
#x0 = 1
S = 1/L

def runge(Knud):
    #MiMe = 1836
    #Te = 1
    TiTe=0
    nn = 1
    #Knud = 0.3
    #E0 = 0
    phi1,ni1,ne1,ui1,ue1,Te1 = pred1(x0/L,Knud)
    phi0 = phi1[0]
    ui0 = ui1[0]
    ue0 = ue1[0]
    ne0 = ne1[0]
    ni0 = ni1[0]
    Te0 = Te1[0]
    compue = 0.01/L/ne0
    phi1,ni,ne,ui,ue1,Te1 = pred(xxpts,Knud)
    Tgrad0 = derivative(Te1)[0]
    E0 = -1*derivative(phi1)[0]

    print(Tgrad0)
    np.savetxt(save_path + "initial.txt",[phi0,E0,ui0,ue0,Tgrad0,Te0,ne0,ni0])#[['phi','E','ui','ue','Te','Tgrad','ne','ni'], 

    '''def system(x,y): 
        phi = y[0] 
        E = y[1]
        ui = y[2]
        Te = y[3]
        ne = y[4]

        #ni = ne*ue/ui x/(L*ui)
        
        source = x/L
        #source = n*u = x/L
        meoverme = 1
        Ti=0
        nom = xr/(nr*ur)

        dphi_dx = -E 
        dE_dx = (source/ui-ne)
        dTe_dx = (-Te/L - source*E - source/MiMe*nueid(source/ui,ne,Te)*(source/ne-ui)/ne*nom - 
        nom/MiMe*nueid(source/ui,ne,Te)*(source/ne-ui)**2 - 3*nom/MiMe*nueqd(source/ui,ne,Te)*(Te-Ti))/(3.92*source - 0.71*ne*ui)
        dui_dx = E/ui - ui/x - Knud + nom/MiMe/source*nueid(source/ui,ne,Te)*(source/ne-ui) + ne/source*0.71*dTe_dx
        dne_dx = (-ne*E - nom/MiMe*nueid(source/ui,ne,Te)*(source/ne-ui) - 1.71*ne*dTe_dx)/Te
        return dphi_dx, dE_dx, dui_dx, dTe_dx, dne_dx'''
    
    def system(x,y): 
        phi = y[0] 
        E = y[1]
        ui = y[2]
        ne = y[3]
        Tgrad = y[4]
        Te = y[5]
        #ni = ne*ue/ui x/(L*ui)
        
        source = x/L

        meoverme = 1
        Ti=0

        Rue = -meoverme*(source/ne-ui)*nueid(source/ui,ne,Te) * xr/(ur*nr) #momentum loss due to collisions with ions 
        RTe = -0.71*ne*MiMe*Tgrad                            #momentum loss due to hot electrons pushing ions towards increasing Te
        #RTe = 0

        Rui = 1/MiMe*(source/ne-ui)*nueid(source/ui,ne,Te) * xr/(ur*nr)    #momentum gain due to ion-electron collisions
        RTi = 0.71*ne*Tgrad
        #RTi = 0RTe = -0.71*ne*MiMe*Tgrad
        #RTi = 0.71*ne*Tgrad
        K0e = -30692/coulogd(ne,Te)

        QRe = (Rue + RTe)/MiMe*(source/ne-ui)                     #joule heating due to net drift of electrons against the dissipative force
        Qeqe = -3/MiMe*nueqd(source/ui,ne,Te)*(Te-Ti)* xr/(ur*nr)   #heat exchange due to thermal equilibrium ion-electron collisions


        #dTe_dx = (-E + L*ue*Rue/(MiMe*x) - Te/x + 1/2*(3-3*Te + 1/MiMe*ue**2) + L/x*(Rue/MiMe*(ue-ui) + Qeqe))/(5/2 + 0.71/ue*(ue-ui) + 0.71)
        #dTe_dx = (-E - Te/x + 1/(2*x)*(3-3*Te + 1/MiMe*ue**2) +2*L/x*Rue/MiMe*ue- L/x*Rue/MiMe*ui + L/x*Qeqe)/(3.92 - 0.71*ui/ue)
        
        dphi_dx = -E 
        dTe_dx = Tgrad
        dE_dx = (source/ui-ne)
        dui_dx = E/ui - ui/x - Knud + L/x * (Rui+RTi)
        dne_dx = (-ne*E - 1.71*ne*Tgrad + Rue/MiMe)/Te
        dTgrad_dx = (1/(2*L)*(3-3*Te + 1/MiMe*((source/ne)**2)) - 3*x*Tgrad/(2*L) - 
            Te*(ne-x*dne_dx)/(L*ne) - 
            5/2*K0e*Te**(3/2)*Tgrad**2*MiMe*ur/(nr*xr) + QRe + 
            Qeqe)/(K0e*Te**(5/2)*MiMe*ur/(nr*xr))

        return dphi_dx, dE_dx, dui_dx, dne_dx, dTgrad_dx, dTe_dx

    xlist = [x0] 
    philist = [phi0]
    Elist = [E0]
    uilist = [ui0]
    uelist = [ue0]
    Telist = [Te0]
    Tgradlist = [Tgrad0]
    nelist = [ne0]

    qwerk = scipy.integrate.RK45(system,x0,[phi0,E0,ui0,ne0,Tgrad0,Te0],105,rtol = 1e-12, atol = 1e-30)
    while qwerk.status == 'running' and qwerk.y[0] > 0:
        qwerk.step()
        xlist.append(qwerk.t)
        philist.append(qwerk.y[0])
        Elist.append(qwerk.y[1])
        uilist.append(qwerk.y[2])
        nelist.append(qwerk.y[3])
        Tgradlist.append(qwerk.y[4])
        Telist.append(qwerk.y[5])
        
        #print(qwerk.status)
    uelist = np.array(xlist)/(L*np.array(nelist))
    nilist = np.array(xlist)/(L*np.array(uilist))

    '''np.savetxt(Data_path + "x.txt", xlist)
    np.savetxt(Data_path + "ni.txt", ni)
    np.savetxt(Data_path + "ne.txt", ne)
    np.savetxt(Data_path + "E.txt", Elist)
    np.savetxt(Data_path + "ui.txt", uilist)
    np.savetxt(Data_path + "ue.txt", uelist)
    np.savetxt(Data_path + "Te.txt", Telist)
    np.savetxt(Data_path + "phi.txt", philist)'''
        
    return xlist, philist, Elist, uilist, uelist,Tgradlist,Telist, nilist, nelist

xRKlist = []
phiRKlist = []
ERKlist = []
niRKlist = []
neRKlist = []
uiRKlist = []
ueRKlist = []
TeRKlist = []
Tgradlist = []

xl, phi, E, ui, ue,Tgrad, Te, ni, ne = runge(0)
xRKlist.append(xl)
phiRKlist.append(phi)
ERKlist.append(E)
niRKlist.append(ni)
neRKlist.append(ne)
uiRKlist.append(ui)
ueRKlist.append(ue)
TeRKlist.append(Te)
Tgradlist.append(Tgrad)

xpts = np.linspace(0,L,numxpts)
fig20, ax20 = plt.subplots(num=20,nrows=1,ncols=1, clear=True)
fig20.set_tight_layout(True)
x = xRKlist[0]
phi = phiList[0]
phiRK = phiRKlist[0]
ax20.plot(x,phiRK,color='b',label="RK45",linewidth=3)
ax20.plot(xpts,phi,color='r',linestyle='--',label='PINN',linewidth=3)
#ax20.legend()
ax20.set_title("$e\\phi /T_s$")
ax20.set_xlabel("$\\lambda_{de}$")
fig20.savefig(save_path + "phiRK.png")
plt.close()

fig21, ax21 = plt.subplots(num=21,nrows=1,ncols=1, clear=True)
fig21.set_tight_layout(True)
ui = uiList[0]
uiRK = uiRKlist[0]
ax21.plot(x,uiRK,color='b',label="RK45",linewidth=3)
ax21.plot(xpts,ui,color='r',linestyle='--',label='PINN',linewidth=3)
ax21.legend()
ax21.set_title("Velocity")
ax21.set_xlabel('$\\lambda_{De}$')
fig21.savefig(save_path + "uiRK.png")
plt.close()

#print(x)
fig22, ax22 = plt.subplots(num=22,nrows=1,ncols=1, clear=True)
fig22.set_tight_layout(True)
ue = ueList[0]
ueRK = ueRKlist[0]
ax22.plot(x,ueRK,color='b',label="RK45",linewidth=3)
ax22.plot(xpts,ue,color='r',linestyle='--',label='PINN',linewidth=3)
ax22.legend()
ax22.set_title("Velocity")
ax22.set_xlabel('$\\lambda_{De}$')
fig22.savefig(save_path + "ueRK.png")
plt.close()

fig23, ax23 = plt.subplots(num=23,nrows=1,ncols=1, clear=True)
fig23.set_tight_layout(True)
ne = neList[0]
ni = niList[0]
neRK = neRKlist[0]
niRK = niRKlist[0]
ax23.plot(x,neRK,color='b',label="ne-RK45",linewidth=3)
ax23.plot(x,niRK,color='c',label="ni-RK45",linewidth=3)
ax23.plot(xpts,ne,color='r',linestyle='--',label='ne-PINN',linewidth=3)
ax23.plot(xpts,ni,color='m',linestyle='--',label='ni-PINN',linewidth=3)
ax23.legend()
ax23.set_title("Density")
ax23.set_xlabel('$\\lambda_{De}$')
fig23.savefig(save_path + "DensitiesRK.png")
plt.close()

fig24, ax24 = plt.subplots(num=24,nrows=1,ncols=1, clear=True)
fig24.set_tight_layout(True)

Te = TeList[0]
TeRK = TeRKlist[0]
ax24.plot(x,TeRK,color='b',label="ne-RK45",linewidth=3)
#ax24.plot(xpts,Te,color='r',label="PINN",linewidth=3)
ax24.legend()
ax24.set_title("Temperature")
ax24.set_xlabel('$\\lambda_{De}$')
fig24.savefig(save_path + "TemperatureRK.png")
plt.close()

fig25, ax25 = plt.subplots(num=25,nrows=1,ncols=1, clear=True)
fig25.set_tight_layout(True)

ERK = ERKlist[0]
ax25.plot(x,ERK,color='b',label="ne-RK45",linewidth=3)
ax25.set_title("Electric Field")
ax25.set_xlabel('$\\lambda_{De}$')
fig25.savefig(save_path + "ERK.png")
plt.close()"""
