import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize
import scipy.integrate
import math

from MagicNeutrals import *

sheathbool = False #Turns off/on sheath entrance calculations as well as graphs that use those calculations

#save_path = '/blue/cmcdevitt/ewebb2/PyTorch/models/test2/figures'
Data_path = '/blue/cmcdevitt/ewebb2/PyTorch/models/test2/'
#model44548
ckpt_save_path = Data_path + "model_final.pt"
#ckpt_save_path = Data_path + "secondmodel18001.pt"

loss = np.loadtxt(Data_path + 'loss_history.txt')
test_loss = np.loadtxt(Data_path + 'test_history.txt')
steps = np.arange(0,len(loss[:,0]),1)
test_steps = np.arange(0,len(test_loss[:,0]),1)
Data_path = Data_path + 'figures/'

numxpts = 1000 # only used when model is loaded


#KnudNeutralVal1 = 3e-1 #Pressure = 10^4
#KnudNeutralVal2 = 7e-2 #Pressure = 10^3
#KnudNeutralVal3 = 1e-2 #Pressure = 10^2
#KnudNeutralVal4 = 1.5e-3 #Pressure = 10^1
#KnudNeutralVal5 = 1.5e-4 #Pressure = 10^0


tempKnudMax = 0.03
# least ionization and recombination no collisionality
TeVal1 = TeMax
TiTeVal1 = TiTeMin
S0Val1 = S0Min
#KnudVal1 = KnudMin
nnVal1 = nnMin

TeVal2 = TeMax
TiTeVal2 = TiTeMin
S0Val2 = S0Min
#KnudVal2 = 0.015
nnVal2 = nnMax

"""
TeVal2 = 20
TiTeVal2 = 0
S0Val2 = 9e31
KnudVal2 = 0.15
nnVal2 = 4

TeVal2 = 10
TiTeVal2 = 1
S0Val2 = 1e31
KnudVal2 = 0.3
nnVal2 = 2
"""

TeVal3 = TeMax
TiTeVal3 = TiTeMin
S0Val3 = S0Max
#KnudVal3 = tempKnudMax
nnVal3 = nnMax

TeVal4 = TeMax
TiTeVal4 = TiTeMin
S0Val4 = S0Max
#KnudVal4 = KnudMin
nnVal4 = nnMax

TeVal5 = TeMax
TiTeVal5 = TiTeMax
S0Val5 = S0Max
#KnudVal5 = KnudMax
nnVal5 = 2

TeVec = np.array([TeVal1,TeVal2,TeVal3,TeVal4,TeVal5])
TiTeVec = np.array([TiTeVal1,TiTeVal2,TiTeVal3,TiTeVal4,TiTeVal5])
S0Vec = np.array([S0Val1,S0Val2,S0Val3,S0Val4,S0Val5])
#KnudVec = np.array([KnudVal1,KnudVal2,KnudVal3,KnudVal4,KnudVal5])
nnVec = np.array([nnVal1,nnVal2,nnVal3,nnVal4,nnVal5])

def Sion2(T,Ez): 
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * np.exp(-Ez/T) 

def Srec2(T,Ez,Z):
    return 5.2e-20 * Z * (Ez/T)**(1/2) * ( 0.43 + 1/2*np.log(Ez/T) + 0.469*(Ez/T)**(-1.3) ) #m^3/s

#dummy vars
#nn=1

interiorpts = [2,2]

Prho = 0.0685 #0.0652

model.load_state_dict(torch.load(ckpt_save_path, map_location=device))

xpts = np.linspace(0,L,numxpts)

def pred(xVal,Te,TiTe,S0,nn):
    model.eval()
    TeNorm = (Te - TeMin ) / ( TeMax - TeMin )
    TiTeNorm = (TiTe-TiTeMin) / (TiTeMax-TiTeMin) 
    S0Norm = (S0 - S0Min ) / ( S0Max - S0Min )
    #KnudNorm = (Knud - KnudMin) / (KnudMax - KnudMin)
    nnNorm = (nn - nnMin) / (nnMax - nnMin)
    xNorm = (xVal-xMin)/(xMax-xMin)

    x = torch.tensor(xNorm,device=device).reshape(-1,1)
    x2 = torch.tensor(1,device=device).reshape(-1,1)
    #X2 = torch.ones_like(X)
    #X = torch.cat((x,TeNorm +x*0, TiTeNorm + x*0, S0Norm + x*0, nnNorm + x*0),dim=1)
    #X2 = torch.cat((x2,TeNorm +x2*0, TiTeNorm +x2*0, S0Norm +x2*0, nnNorm +x2*0),dim=1)
    X = torch.cat((x,TeNorm + x*0, TiTeNorm + x*0, S0Norm + x*0, nnNorm + x*0),dim=1)
    X2 = torch.cat((x2,TeNorm +x2*0, TiTeNorm +x2*0, S0Norm +x2*0, nnNorm +x2*0),dim=1)

    with torch.no_grad():
        X_trans = feature_transform(X)
        outputs = model(X_trans)
        outputs_trans = output_transform(X, outputs).cpu().numpy()

        outputs2 = model(feature_transform(X2))
        outputs2_trans = output_transform(X2, outputs2).cpu().numpy()

    phi, ni, uet = outputs_trans[:,0], outputs_trans[:,1], outputs_trans[:,2]

    #newall = np.exp(outputs2_trans[:,3])
    newall = outputs2_trans[:,3]
    #print("newall at wall:", newall)
    #newall = np.exp(newallt)
    #print(newall)
    #exit()

    ne = newall*np.exp(phi)
    #ue = np.exp(uet)-1
    ue = np.expm1(uet)
    #ue = uet
    ui = ne*ue/ni

    return phi,ni,ne,ui,ue,newall

def res(test_pts):
    model.zero_grad() 
    residuals = pde(model, test_pts)
   
    if not isinstance(residuals, (tuple, list)):
        residuals = [residuals]
    losses = [torch.mean(res**2).detach().cpu().numpy() for res in residuals]
    #space_res = [resi.detach().cpu().numpy() for resi in residuals]
    loss = sum(losses)

    #total_loss = [hist for hist in losses]

    return losses, loss

def SheathEntrance(xVal,TeVal,TiTeVal,S0Val,nnVal):
    _, ni, ne, _, _,_ = pred(xVal,TeVal,TiTeVal,S0Val,nnVal)

    return (ni-ne) / ne - Prho

numTepts = 10
numTiTepts = 10
numS0pts = 10
#numKnudpts = 30
#numKnudlogpts = 15
numnnpts = 10
TeScan = np.linspace(TeMin,TeMax,numTepts)
TiTeScan = np.linspace(TiTeMin, TiTeMax, numTiTepts)
S0Scan = np.linspace(S0Min, S0Max, numS0pts)
#KnudScan = np.logspace(-4,-1,numKnudlogpts)
#KnudScan = np.append(KnudScan, np.linspace(0.1, KnudMax, numKnudpts))
nnScan = np.linspace(nnMin, nnMax, numnnpts)
#numKnud = len(KnudScan)
xSEvec = np.zeros([numTiTepts,numnnpts])
phiSEvec = np.zeros([numTiTepts,numnnpts])
uiSEvec = np.zeros([numTiTepts,numnnpts])
neSEvec = np.zeros([numTiTepts,numnnpts])
#Tevec = np.zeros([numTiTepts,numnnpts])
TiTevec = np.zeros([numTiTepts,numnnpts])
#S0vec = np.zeros([numTiTepts,numnnpts])
#Knudvec = np.zeros([numTepts, numTiTepts,numS0pts,numKnud,numnnpts])
nnvec = np.zeros([numTiTepts,numnnpts])
niSEvec = np.zeros([numTiTepts,numnnpts])


phiList = []
neList = []
niList = []
ueList = []
uiList = []
    
phi, ni, ne, ui, ue,_ = pred(xpts,TeVal1,TiTeVal1,S0Val1,nnVal1)

print("ni", ni[0])

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_= pred(xpts,TeVal2,TiTeVal2,S0Val2,nnVal2)

print("ni", ni[0])

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_ = pred(xpts,TeVal3,TiTeVal3,S0Val3,nnVal3)

print("ni", ni[0])

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_ = pred(xpts,TeVal4,TiTeVal4,S0Val4,nnVal4)

print("ni", ni[0])

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_ = pred(xpts,TeVal5,TiTeVal5,S0Val5,nnVal5)

print("ni", ni[0])

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

#X_test = torch.rand(pts,6).to(device).requires_grad_(True)
#test_loss, tot_test = res(X_test)



def residual_pts(x,Te,TiTe,S0,nn):
    xNorm = (x - xMin) / (xMax - xMin)
    TeNorm = (Te - TeMin ) / ( TeMax - TeMin )
    TiTeNorm = (TiTe - TiTeMin) / (TiTeMax - TiTeMin)
    S0Norm = (S0 - S0Min ) / ( S0Max - S0Min )
    #KnudNorm = (Knud - KnudMin) / ( KnudMax - KnudMin )
    nnNorm = (nn - nnMin) / ( nnMax - nnMin )

    x_t = torch.tensor(xNorm,device=device).reshape(-1,1)
    Te_t = torch.tensor(TeNorm,device=device).reshape(-1,1)
    TiTe_t = torch.tensor(TiTeNorm,device=device).reshape(-1,1)
    S0_t = torch.tensor(S0Norm,device=device).reshape(-1,1)
    #Knud_t = torch.tensor(KnudNorm,device=device).reshape(-1,1)
    nn_t = torch.tensor(nnNorm,device=device).reshape(-1,1)

    XX = torch.cat((x_t,Te_t, TiTe_t, S0_t, nn_t),dim=1)

    Y = pde(model, XX)
    if not isinstance(Y, (tuple, list)):
        Y = [Y]
    Y = [res.detach().cpu().numpy() for res in Y]

    return Y

testFlux1 = []
testFlux2 = []
testFlux3 = []
testFlux4 = []

def mfp(Te,TiTe,S0,nn):
    uref = np.sqrt((Te+Te*TiTe)*q/(mi))
    #lref = torch.pow(Eps0*Te*uref/(S0*L*q), 1/3)
    lref = (Eps0*Te*uref/(S0*L*q))**(1/3)
    nref = S0*L*lref/uref
    ui = 1
    un = 0
    #print("lref:", lref)
    #print("nref:", nref)
    #print(signn(TiTe*Te,lref,nref))

    return nn*(ui-un)*signn(TiTe*Te,lref,nref) #ni*ui*

meanfp = []

for i in TeScan:
    x = mfp(i,TiTeMax,S0Max,nnMax)
    meanfp.append(x)

fig, ax = plt.subplots(nrows=1,ncols=1, clear=True)
fig.set_tight_layout(True)
ax.plot(TeScan, meanfp, color='b')
ax.set_yscale("log")
ax.set_xlabel("Ion Temperature (eV)")
ax.set_ylabel("Mean Free Path ($\\lambda_{De}$)")
ax.set_title("Mean Free Path vs Ion Temperature")
fig.savefig(Data_path + "MeanFreePath.png")
plt.close()

for i in nnScan:
    sol = optimize.root(SheathEntrance, [L-5], args = (TeVal1,TiTeVal1,S0Val1,i))
    xSE = sol.x
    #phiSE, niSE, neSE, uiSE, ueSE = pred(xSE,TeVal1,TiTeVal1,S0Val1,KnudVal1,i)
    #phic, nic, nec, uic, uec = pred(0,TeVal1,TiTeVal1,S0Val1,KnudVal1,i)
    phiedge1, niedge1, needge1, uiedge1, ueedge1,_ = pred(L,TeVal1,TiTeVal1,S0Val4,i)
    testFlux1.append(uiedge1*niedge1)
    phiedge2, niedge2, needge2, uiedge2, ueedge2,_ = pred(L,TeVal2,TiTeVal1,S0Val4,i)
    testFlux2.append(uiedge2*niedge2)
    phiedge3, niedge3, needge3, uiedge3, ueedge3,_ = pred(L,TeVal1,TiTeVal1,S0Val1,i)
    testFlux3.append(uiedge3*niedge3)
    phiedge4, niedge4, needge4, uiedge4, ueedge4,_ = pred(L,TeVal2,TiTeVal1,S0Val1,i)
    testFlux4.append(uiedge4*niedge4)

fig, ax = plt.subplots(nrows=1,ncols=1, clear=True)
fig.set_tight_layout(True)

ax.plot(nnScan, testFlux1, linestyle='-',label='Te=1 eV', color='r', linewidth=2)
ax.plot(nnScan, testFlux4, linestyle='-', label='Low $S_0$', color='b', linewidth=2)
ax.plot(nnScan, testFlux2, linestyle='-', label='High $S_0$', color='g', linewidth=2)


ax.set_xlabel("Neutral Density $n_n$")
ax.set_ylabel("Flux ($\\Gamma$)")
ax.set_title("Flux with High Source")
ax.legend()
fig.savefig(Data_path + "SheathParams1.png")
plt.close()

fig, ax = plt.subplots(nrows=1,ncols=1, clear=True)
fig.set_tight_layout(True)

ax.plot(nnScan, testFlux3, linestyle='-',label='Te=1 eV', color='b', linewidth=2)
ax.plot(nnScan, testFlux4, linestyle='-', label='Te=20 eV', color='r', linewidth=2)

ax.set_xlabel("Neutral Density $n_n$")
ax.set_ylabel("Flux ($\\Gamma$)")
ax.set_title("Flux with low Source")
fig.savefig(Data_path + "SheathParams2.png")
plt.close()

# Find sheath entrance
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal1,TiTeVal1,S0Val1,nnVal1))
xSE1 = sol.x
#print("sheath entrance = " + str(xSE1))
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal2,TiTeVal2,S0Val2,nnVal2))
xSE2 = sol.x
#print("sheath entrance = " + str(xSE2))
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal3,TiTeVal3,S0Val3,nnVal3))
xSE3 = sol.x
#print("sheath entrance = " + str(xSE3))
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal4,TiTeVal4,S0Val4,nnVal4))
xSE4 = sol.x
#print("sheath entrance = " + str(xSE4))
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal5,TiTeVal5,S0Val5,nnVal5))
xSE5 = sol.x
#print("sheath entrance = " + str(xSE5))
print("xSE1, xSE2, xSE3, xSE4:", xSE1, xSE2, xSE3, xSE4)
    
# Compute values at sheath entrance
phiSE1, niSE1, neSE1, uiSE1, ueSE1,_ = pred(xSE1,TeVal1,TiTeVal1,S0Val1,nnVal1)
phiSE2, niSE2, neSE2, uiSE2, ueSE2,_ = pred(xSE2,TeVal2,TiTeVal2,S0Val2,nnVal2)
phiSE3, niSE3, neSE3, uiSE3, ueSE3,_ = pred(xSE3,TeVal3,TiTeVal3,S0Val3,nnVal3)
phiSE4, niSE4, neSE4, uiSE4, ueSE4,_ = pred(xSE4,TeVal4,TiTeVal4,S0Val4,nnVal4)
phiSE5, niSE5, neSE5, uiSE5, ueSE5,_ = pred(xSE5,TeVal5,TiTeVal5,S0Val5,nnVal5)
if sheathbool == True:
    counts = 0

    #for i in range(0,numTepts):
    for j in range(0,numTiTepts):   
            #for k in range(0,numS0pts):  
                #for g in range(0,numKnud):
        for h in range(0,numnnpts):

            sol = optimize.root( SheathEntrance, [L-5], args = (TeMax,TiTeScan[j],S0Max,nnScan[h]))
            xSE = sol.x
            #print("sheath:",xSE)
            phiSE, niSE, neSE, uiSE, ueSE,_ = pred(xSE,TeMax,TiTeScan[j],S0Max,nnScan[h])    
            #phiEdge, niEdge, neEdge, uiEdge, ueEdge = pred(L,TeScan[i],TiTeScan[j],S0Scan[k],KnudScan[g],nnScan[h])  
            phiCenter, niCenter, neCenter, uiCenter, ueCenter,_ = pred(0,TeMax,TiTeScan[j],S0Max,nnScan[h]) 

            phiSEvec[j,h] = phiSE
            uiSEvec[j,h] = uiSE
            neSEvec[j,h] = neSE / niCenter
            niSEvec[j,h] = niSE / niCenter
            xSEvec[j,h] = xSE
            Tevec[j,h] = TeMax
            TiTevec[j,h] = TiTeScan[j]
            S0vec[j,h] = S0Max
            #Knudvec[i,j,k,g,h] = KnudScan[g] #np.sqrt(neCenter)
            nnvec[j,h] = nnScan[h]

            counts = counts +1
            if counts%10000 == 0:
                print("Percent done:", counts*100/(numTiTepts*numnnpts),'%')


#0 - Hydrogen at TeMin TiTeMin S0Min | xSE1
#1 - Helium at TeMax TiTeMax S0Max | xSE2
#2 - Argon at TeMax/2 TiTeMax S0Min | xSE3

ptsize = 100

#plt.rcParams.update({'font.size': 14})
xpts = np.linspace(0,L,numxpts)
fig1, ax1 = plt.subplots(num=1,nrows=1,ncols=1, clear=True)
fig1.set_tight_layout(True)

phi = phiList[0]
ne = neList[0]
ax1.plot((xpts), phi, label='$S_0=10^{28}$, $n_n=0$', linestyle='-',color='r',linewidth=3,zorder=1)
ax1.scatter(xSE1, phiSE1,color='red',s=ptsize,zorder=10)

phi = phiList[1]
ne = neList[1]
ax1.plot((xpts), phi, label='$S_0=10^{28}$, $n_n=5$', linestyle='-',color='b',linewidth=3,zorder=1)
ax1.scatter(xSE2, phiSE2,color='blue',s=ptsize,zorder=10)

phi = phiList[2]
ne = neList[2]
ax1.plot((xpts), phi, label='$S_0=10^{29}$, $n_n=5$', linestyle='-',color='g',linewidth=3,zorder=1)
ax1.scatter(xSE3, phiSE3,color='green',s=ptsize,zorder=10)
'''
phi = phiList[3]
ne = neList[3]
ax1.plot((xpts), phi, label='$\\phi$', linestyle='-',color='orange',linewidth=3)
ax1.scatter((xSE4), phiSE4,color='orange',s=ptsize,zorder=10)

phi = phiList[4]
ne = neList[4]
ax1.plot((xpts), phi, label='$\\phi$', linestyle='-',color='orange',linewidth=3)
if sheathbool == True:
    ax1.scatter((xSE5), phiSE5,color='orange', lw=2)
'''
#ax1.set_ylabel("$\\phi$")
ax1.set_xlabel("$x/\\lambda_{de}$")
ax1.set_title("$e\\phi/T_0$")
#ax1.set_yscale("log")
ax1.legend()
fig1.savefig(Data_path + "phi.png")

fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)

ne = neList[0]
ni = niList[0]
ax2.plot((xpts), ni, label='Ions', linestyle='-',color='r',linewidth=3,zorder=1)
ax2.plot((xpts), ne, label='Electrons', linestyle='--',color='r',linewidth=3,zorder=1)
ax2.scatter((xSE1), niSE1,color='red',s=ptsize,zorder=10)

ne = neList[1]
ni = niList[1]
ax2.plot((xpts), ni, linestyle='-',color='b',linewidth=3,zorder=1)
ax2.plot((xpts), ne, linestyle='--',color='b',linewidth=3,zorder=1)
ax2.scatter((xSE2), niSE2,color='blue',s=ptsize,zorder=10)

ne = neList[2]
ni = niList[2]
ax2.plot((xpts), ni,  linestyle='-',color='g',linewidth=3,zorder=1)
ax2.plot((xpts), ne, linestyle='--',color='g',linewidth=3,zorder=1)
ax2.scatter((xSE3), niSE3,color='green',s=ptsize,zorder=10)
'''
ne = neList[3]
ni = niList[3]
ax2.plot((xpts), ni, linestyle='-',color='orange',linewidth=3)
ax2.plot((xpts), ne, linestyle='--',color='orange',linewidth=3)
ax2.scatter((xSE4), niSE4,color='orange',s=ptsize,zorder=10)

ne = neList[4]
ni = niList[4]
ax2.plot((xpts), ne, label='$n_e$', linestyle='--',color='orange',linewidth=3)
ax2.plot((xpts), ni, label='$Min Te | Max Knud$', linestyle='-',color='orange',linewidth=3)
if sheathbool == True:
    ax2.scatter((xSE5), niSE5,color='orange',linewidth=2)
'''
"""ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.set_title("Density Argon")
ax2.legend()
ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.set_title("Density Helium")
ax2.legend()"""
ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.set_title("Density")
ax2.legend()
fig2.savefig(Data_path + "Densities.png")

fig3, ax3 = plt.subplots(num=3,nrows=1,ncols=1, clear=True)
fig3.set_tight_layout(True)

ax3.plot((xpts), np.ones(len(xpts)), label='$C_s$', linestyle=':',color='black',linewidth=3)
#ax3.plot((xpts), ((1-np.tanh(100-xpts/(0.01*L)))+(np.tanh(100+xpts/(0.01*L))-1))*17.0999193011, label='$C_s$', linestyle=':',color='m',linewidth=2)
#ax3.plot((xpts), np.ones(len(xpts)), label='$C_s$', linestyle=':',color='black',linewidth=2)

ue = ueList[0]
ui = uiList[0]
ax3.plot((xpts), ui, label='Ions', linestyle='-',color='r',linewidth=3,zorder=1)
ax3.plot((xpts), ue, label='Electrons',linestyle='--',color='r',linewidth=3,zorder=1)
ax3.scatter((xSE1), uiSE1,color='red',s=ptsize,zorder=10)

ue = ueList[1]
ui = uiList[1]
ax3.plot((xpts), ui, linestyle='-',color='b',linewidth=3,zorder=1)
ax3.plot((xpts), ue, linestyle='--',color='b',linewidth=3,zorder=1)
ax3.scatter((xSE2), uiSE2,color='blue',s=ptsize,zorder=10)

ue = ueList[2]
ui = uiList[2]
ax3.plot((xpts), ui, linestyle='-',color='g',linewidth=3,zorder=1)
ax3.plot((xpts), ue, linestyle='--',color='g',linewidth=3,zorder=1)
ax3.scatter((xSE3),uiSE3,color='green',s=ptsize,zorder=10)
'''
ue = ueList[3]
ui = uiList[3]
ax3.plot((xpts), ui, linestyle='-',color='orange',linewidth=3)
ax3.plot((xpts), ue, linestyle='--',color='orange',linewidth=3)
ax3.scatter((xSE4),uiSE4,color='orange',s=ptsize,zorder=10)

ue = ueList[4]
ui = uiList[4]
ax3.plot((xpts), ui, label='Max', linestyle='-',color='orange',linewidth=3)
ax3.plot((xpts), ue, linestyle='--',color='orange',linewidth=3)
if sheathbool == True:
    ax3.scatter((xSE5),uiSE5,color='orange',linewidth=2)
'''
#ax3.set_ylabel("$\\phi$")
ax3.set_xlabel("$x/\\lambda_{De}$")
#ax3[0,1].set_xlabel("$x/\\lambda_{de}$")
#ax3[1,0].set_xlabel("$x/\\lambda_{de}$")
ax3.set_title("Velocity [$C_s$]")
ax3.set_ylim([0,4])
#ax3[0,1].set_title("Velocity [$C_s$] Helium")
#ax3[1,0].set_title("Velocity [$C_s$] Hydrogen")
ax3.legend()
#ax3[0,1].legend()
#ax3[1,0].legend()
#ax3.set_yscale("log")
fig3.savefig(Data_path + "Velocities.png")

#print("ue(x=L):", ue[-1])
#print("uewall:", np.sqrt(MiMe*TeVal4/(2*np.pi*(TeVal4+TiTeVal4*TeVal4))))
#xxpt = xpts/sigma
fluchs = xpts/L

fig4, ax4 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig4.set_tight_layout(True)

ue = ueList[0]
ui = uiList[0]
ne = neList[0] 
ni = niList[0]

ax4.plot((xpts), ue*ne, label='$S_0=10^{28}$, $n_n=0$', linestyle='-',color='r',linewidth=3)
#ax4.plot((xpts), fluchs, label="Ana Flux",linestyle='--', color='k',linewidth=3)

ue = ueList[1]
ui = uiList[1]
ne = neList[1] 
ni = niList[1]
ax4.plot((xpts), ue*ne, label='$S_0=10^{28}$, $n_n=5$', linestyle='-',color='b',linewidth=3)

ue = ueList[2]
ui = uiList[2]
ne = neList[2] 
ni = niList[2]
ax4.plot((xpts), ue*ne, label='$S_0=10^{29}$, $n_n=5$', linestyle='-',color='g',linewidth=3)
'''
ue = ueList[3]
ui = uiList[3]
ne = neList[3] 
ni = niList[3]
ax4.plot((xpts), ue*ne, label='Finite Ion Temperature', linestyle='--',color='orange',linewidth=3)

ue = ueList[4]
ui = uiList[4]
ne = neList[4] 
ni = niList[4]
ax4.plot((xpts), ue*ne, label='Max', linestyle=':',color='orange',linewidth=3)
'''
#ax4.set_ylabel("$\\phi$")
ax4.set_xlabel("$x/\\lambda_{de}$")
ax4.set_title("Flux")
#ax4.set_yscale("log")
#ax4.legend()
#ax4.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
#fig.subplots_adjust(left=0.15)
fig4.savefig(Data_path + "Flux")#,bbox_inches='tight')

def Sauces(S,Te):
    #Tref = 10 #eV
    uref = (Te*q/mi)**(1/2)
    #nref = S0/uref
    lref = (Eps0*Te*uref/(S*L*q))**(1/3)
    nref = S*L*lref/uref
    barS0 = S*lref**4*L/uref
    barSion = Sion2(Te,15.8)*(lref*nref/uref)
    barSrec = Srec2(Te,15.8,1)/(lref**2*uref)

    return barS0, barSion, barSrec

barS01, barSion1, barSrec1 = Sauces(S0Min,TeMin)
barS02, barSion2, barSrec2 = Sauces(S0Max,TeMax)

fig5, ax5 = plt.subplots(num=4,nrows=2,ncols=3, clear=True)
fig5.set_tight_layout(True)

#Gammae = GammaeList[0]
#Gammai = GammaiList[0]
ne = neList[0] 
ni = niList[0] 
ax5[0,0].plot((xpts), nnMax*ne*barSion1*barS01, label='Ionization Source', linestyle='-',color='red',linewidth=3)
ax5[0,1].plot((xpts), ni*ne*barSrec1*barS01, label='Recombination Source', linestyle='-',color='blue',linewidth=3)
ax5[0,2].plot(xpts, [0.02 for i in range(len(xpts))],linestyle='-',color='g',linewidth=3)

ne = neList[1] 
ni = niList[1] 
ax5[1,0].plot((xpts), nnMax*ne*barSion2*barS02, label='Ionization Source', linestyle='-',color='red',linewidth=3)
ax5[1,1].plot((xpts), ni*ne*barSrec2*barS02, label='Recombination Source', linestyle='-',color='blue',linewidth=3)
ax5[1,2].plot(xpts,[0.02 for i in range(len(xpts))],linestyle='-',color='g',linewidth=3)


#ax4.set_ylabel("$\\phi$")
ax5[0,0].set_xlabel("$x/\\lambda_{de}$")
ax5[0,1].set_xlabel("$x/\\lambda_{de}$")
ax5[0,2].set_xlabel("$x/\\lambda_{de}$")
ax5[1,0].set_xlabel("$x/\\lambda_{de}$")
ax5[1,1].set_xlabel("$x/\\lambda_{de}$")
ax5[1,2].set_xlabel("$x/\\lambda_{de}$")
ax5[0,0].set_title("Ion low Te")
ax5[0,1].set_title("Rec low Te")
ax5[0,2].set_title("Source")
ax5[1,0].set_title("Ion high Te")
ax5[1,1].set_title("Rec high Te")
ax5[1,2].set_title("Source")
#ax4.set_yscale("log")
#ax5.legend()
fig5.savefig(Data_path + "Sources")

"""fig5, ax5 = plt.subplots(num=5,nrows=1,ncols=1, clear=True)
fig5.set_tight_layout(True)

ne = neList[2]
ax5.plot(xpts, Prho*np.ones(len(xpts)), label='$P_\\rho$', linestyle=':',color='black',linewidth=2)

ne = neList[0]
ni = niList[0]
ax5.plot(xpts, (ni-ne)/ne, label='$(n_i-n_e)/n_e$', linestyle='-',color='black',linewidth=2)

ne = neList[1]
ni = niList[1]
ax5.plot(xpts, (ni-ne)/ne, label='$(n_i-n_e)/n_e$', linestyle='-',color='blue',linewidth=2)

ne = neList[4]
ni = niList[4]
ax5.plot(xpts, (ni-ne)/ne, label='$(n_i-n_e)/n_e$', linestyle='-',color='red',linewidth=2)

#ax5.set_ylabel("$\\phi$")
ax5.set_xlabel("$x/\\lambda_{de}$")
ax5.set_title("$(n_i-n_e)/n_e$")
ax5.set_yscale("log")
#ax5.set_ylim([1.e-3,1.1])
#ax5.legend()
fig5.savefig(Data_path + "Sheath_Entrance.png")"""
#Te, TiTe, S0, Knud, nn
if sheathbool == True:
    fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
    fig7.set_tight_layout(True)

    ax7.plot(nnvec[0,:], niSEvec[0,:], 'r', label='Te=0eV',linewidth=3)
    ax7.plot(nnvec[0,:], niSEvec[-1,:], 'b', label='Te=20eV',linewidth=3)
    #ax7.plot(Knudvec[0,0,0,:,0], niSEvec[-1,0,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
    #ax7.plot(Knudvec[0,0,0,:,0], niSEvec[-1,-1,0,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)
    #ax7.set_ylabel("$\\phi$")
    ax7.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax7.set_title("Density Ratio")
    ax7.set_xscale("log")
    ax7.legend()
    fig7.savefig(Data_path + "Density_Ratio.png")

    fig8, ax8 = plt.subplots(num=8,nrows=1,ncols=1, clear=True)
    fig8.set_tight_layout(True)

    ax8.plot(nnvec[0,:], phiSEvec[0,:], 'r', label='Te=0eV',linewidth=3)
    ax8.plot(nnvec[0,:], phiSEvec[-1,:], 'b', label='Te=20eV',linewidth=3)
    #ax8.plot(Knudvec[0,0,0,:,0], phiSEvec[-1,0,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
    #ax8.plot(Knudvec[0,0,0,:,0], phiSEvec[-1,-1,0,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)
    #ax7.set_ylabel("$\\phi$")
    ax8.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax8.set_title("Potential Drop")
    ax8.set_xscale("log")
    #ax8.legend()
    fig8.savefig(Data_path + "Potential_Drop.png")

    fig9, ax9 = plt.subplots(num=9,nrows=1,ncols=1, clear=True)
    fig9.set_tight_layout(True)

    ax9.plot(nnvec[0,:], uiSEvec[0,:], 'r', label='Te=0eV, Ti=0',linewidth=3)
    ax9.plot(nnvec[0,:], uiSEvec[-1,:], 'b', label='Te=0eV, Ti=Te',linewidth=3)
    #ax9.plot(Knudvec[0,0,0,:,0], uiSEvec[-1,0,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
    #ax9.plot(Knudvec[0,0,0,:,0], uiSEvec[-1,-1,0,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)
    #ax7.set_ylabel("$\\phi$")
    ax9.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax9.set_title("Bohm Criterion")
    ax9.set_xscale("log")
    #ax9.legend()
    fig9.savefig(Data_path + "Bohm_Criterion.png")
    """PICSH = np.array([4.5,3.4,3.5,3.1,3.])
    NumSH = np.array([4.7,3.6,3.1,2.9,2.9])
    fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
    fig7.set_tight_layout(True)

    #ax7.plot(KnBevingSEvec[:,0,0], phiSEvec[:,0,0], '-k', linewidth=2, label='H - Low $T_i$')
    #ax7.plot(KnBevingSEvec[:,0,9], phiSEvec[:,0,9], '-k',linestyle='--', linewidth=2, label='H - $T_i=T_e$')
    #ax7.plot(KnBevingSEvec[:,3,0], phiSEvec[:,3,0], '-g', linewidth=2, label='He - Low $T_i$')
    ax7.plot(KnBevingSEvec[:,3,9], phiSEvec[:,3,9], '-b', linestyle='--', linewidth=2, label='He - $T_i=T_e$')
    #ax7.plot(KnBevingSEvec[:,39,0], phiSEvec[:,39,0], '-r', linewidth=2, label='Ar - Low $T_i$')
    ax7.plot(KnBevingSEvec[:,39,9], phiSEvec[:,39,9], '-r', linestyle='--', linewidth=2, label='Ar - $T_i=T_e$')
    #ax7.plot(KnudNumVec,PICSH, '-b', label="Bev_PIC")
    #ax7.scatter(KnudNumVec,PICSH,c='b')
    #ax7.plot(KnudNumVec,NumSH+0.6, '-r', label="Bev_Num")
    #ax7.scatter(KnudNumVec,NumSH+0.6,c='r')

    #ax7.set_ylabel("$\\phi$")
    ax7.set_xlabel("$\\lambda_{de}/\\lambda_{in}$")
    ax7.set_title("Potential drop")
    ax7.set_xscale("log")
    ax7.legend()
    fig7.savefig(Data_path + "Potential_Drop.png")

    def a1(ne,ni,knud):
        #E = -L*(ne-ni)
        E = 0.1
        return -(knud/(2*E))+np.sqrt(1+(knud/(2*E))**2) #Equation 12 (Fluid model version) from Beving
    
    Bev_Bohm = np.array([a1(neSE1,niSE1,KnudNumVec[0]),a1(neSE2,niSE2,KnudNumVec[1]),a1(neSE3,niSE3,KnudNumVec[2]),a1(neSE4,niSE4,KnudNumVec[3]),a1(neSE5,niSE5,KnudNumVec[4])])

    def h1(knud):
        return (0.5+0.5*knud)/(1 + 30*knud) #Equation 14 from beving

    Bev_ECDR = np.array([0.15,0.24,h1(KnudNumVec[2]),h1(KnudNumVec[3]),h1(KnudNumVec[4])])

    fig8, ax8 = plt.subplots(num=8,nrows=1,ncols=1, clear=True)
    fig8.set_tight_layout(True)

    ax8.plot(KnBevingSEvec[:,0,0], uiSEvec[:,0,0], '-k', linewidth=2, label='H')
    #ax8.plot(KnBevingSEvec[:,0,9], uiSEvec[:,0,9], '-k', linestyle='--', linewidth=2, label='H - $T_i=T_e$')
    ax8.plot(KnBevingSEvec[:,3,0], uiSEvec[:,3,0], '-b', linewidth=2, label='He')
    #ax8.plot(KnBevingSEvec[:,3,9], uiSEvec[:,3,9], '-b', linestyle='--', linewidth=2, label='He - $T_i=T_e$')
    ax8.plot(KnBevingSEvec[:,39,0], uiSEvec[:,39,0], '-r', linewidth=2, label='Ar')
    #ax8.plot(KnBevingSEvec[:,39,9], uiSEvec[:,39,9], '-r', linestyle='--', linewidth=2, label='Ar - $T_i=T_e$')
    #ax8.plot(KnudNumVec,Bev_Bohm,c='g',linewidth=2, linestyle='--',label = 'a1')
    #ax8.scatter(KnudNumVec,Bev_Bohm,c='g')

    #ax8.set_ylabel("$\\phi$")
    ax8.set_xlabel("$\\lambda_{de}/\\lambda_{in}$")
    ax8.set_title("Bohm Criterion")
    #ax8.set_xlim([0,3e-1])
    ax8.set_xscale("log")
    ax8.legend()
    fig8.savefig(Data_path + "Bohm_Criterion.png")

    fig9, ax9 = plt.subplots(num=9,nrows=1,ncols=1, clear=True)
    fig9.set_tight_layout(True)

    ax9.plot(KnBevingSEvec[:,0,0], neSEvec[:,0,0], '-k', linewidth=2, label='H - Low $T_i$')
    ax9.plot(KnBevingSEvec[:,0,9], neSEvec[:,0,9], '-k', linestyle='--', linewidth=2, label='H - $T_i=T_e$')
    ax9.plot(KnBevingSEvec[:,3,0], neSEvec[:,3,0], '-b', linewidth=2, label='He - Low $T_i$')
    ax9.plot(KnBevingSEvec[:,3,9], neSEvec[:,3,9], '-b', linestyle='--', linewidth=2, label='He - $T_i=T_e$')
    ax9.plot(KnBevingSEvec[:,39,0], neSEvec[:,39,0], '-r', linewidth=2, label='Ar - Low $T_i$')
    ax9.plot(KnBevingSEvec[:,39,9], neSEvec[:,39,9], '-r', linestyle='--', linewidth=2, label='Ar - $T_i=T_e$')
    ax9.plot(KnudNumVec, Bev_ECDR,color='g',linestyle='--',label='h1')
    ax9.scatter(KnudNumVec, Bev_ECDR,color='g')

    #ax9.set_ylabel("$\\phi$")
    ax9.set_xlabel("$\\lambda_{de}/\\lambda_{in}$")
    ax9.set_title("$n_{SE}$")
    ax9.set_xscale("log")
    ax9.legend()
    fig9.savefig(Data_path + "n_SE")

    Sheath_width = np.array([L-xSE5,L-xSE4,L-xSE3,L-xSE2,L-xSE1])
    PIC = np.array([32,16,10,9,8])
    Numer = np.array([18,12,9,8,8])"""
    #Sheath_width = np.array([L-xSE1,L-xSE2,L-xSE3,L-xSE4,L-xSE5])
    fig11, ax11 = plt.subplots(num=9,nrows=1,ncols=1, clear=True)
    fig11.set_tight_layout(True)

    ax11.plot(nnvec[0,:], L-xSEvec[0,:], 'r', label='Te=0eV, Ti=0',linewidth=3)
    ax11.plot(nnvec[0,:], L-xSEvec[-1,:], 'b', label='Te=0eV, Ti=Te',linewidth=3)
    #ax11.plot(Knudvec[0,0,0,:,0], L-xSEvec[-1,0,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
    #ax11.plot(Knudvec[0,0,0,:,0], L-xSEvec[-1,-1,0,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)

    #ax9.set_ylabel("$\\phi$")
    ax11.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax11.set_ylabel("# $\\lambda_{De}$")
    ax11.set_title("Sheath Width")
    ax11.set_xscale('log')
    #ax11.legend()
    fig11.savefig(Data_path + "Sheath_Width")
"""
fig10, ax10 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig10.set_tight_layout(True)

n_de = debye(0.9,nref)
nrefHC = nref*(SHC/S)**(2/3)
n_deHC = debye(0.326,nrefHC)

ne = neList[0]
ni = niList[0]
ax10.plot((xpts*n_deHC), ne*nref/1e17, label='$n_e$', linestyle='--',color='black',linewidth=2)
ax10.plot((xpts*n_deHC), ni*nref/1e17, label='$n_i$', linestyle='-',color='black',linewidth=2)

ne = neList[1]
ni = niList[1]
ax10.plot((xpts*n_de), ne*nref/1e17, label='$n_e$', linestyle='--',color='red',linewidth=2)
ax10.plot((xpts*n_de), ni*nref/1e17, label='$n_i$', linestyle='-',color='red',linewidth=2)

ne = neList[2]
ni = niList[2]
ax10.plot((xpts*n_de), ne*nref/1e17, label='$n_e$', linestyle='--',color='blue',linewidth=2)
ax10.plot((xpts*n_de), ni*nref/1e17, label='$n_i$', linestyle='-',color='blue',linewidth=2)

#ax2.set_ylabel("$\\phi$")
ax10.set_xlabel("x (m)")
ax10.set_title("Density (x$10^{17}$)")
#ax2.set_yscale("log")
ax10.legend()
fig10.savefig(Data_path + "Densities_True.png")
"""
"""fig15, ax15 = plt.subplots(num=15,nrows=1,ncols=1, clear=True)
fig15.set_tight_layout(True)
ax15.scatter(trainpts[:,0],trainpts[:,1],s=0.001)

#n, bins, patches = ax15.hist(xtrainpts[:,0], 50, density=False, facecolor='g')

#ax15.set_ylabel("$\\phi$")
ax15.set_xlabel("$x/\\lambda_{de}$")
ax15.set_title("Training points")
#ax15.set_yscale("log")
#ax15.legend()
fig15.savefig(Data_path + "Training_points.png")"""

def fixloss(l):
    teststeps = [0]
    loss2 = [l[0]]
    for i in range(len(l)-1):
        if l[i+1] != l[i]:
            loss2.append(l[i+1])
            teststeps.append(steps[i+1])
    return np.array(teststeps),loss2

max_index = loss[:,0].argmax()
print("Max index:", max_index)
train_pos = loss[:,0]#[:5045]#np.delete(loss[:,0], [5045:-1])
train_EC = loss[:,1]#[:5045]#np.delete(loss[:,1], [5045:-1])
train_IM = loss[:,2]#[:5045]#np.delete(loss[:,2], [5045:-1])
#steps = steps[:5045]#np.delete(steps, [5045:])


while max(train_pos) > 1e8 or max(train_EC) > 1e8 or max(train_IM) > 1e8:
    max_index = train_EC.argmax()
    try:
        train_pos = np.delete(train_pos, max_index)
        train_EC = np.delete(train_EC, max_index)
        train_IM = np.delete(train_IM, max_index)
        steps = np.delete(steps, max_index)
    except:
        break

#train_pos = loss[:,0]#np.delete(loss[:,0], [max_index,-11,-12,-10])
#train_EC = loss[:,1]#np.delete(loss[:,1], [max_index,-11,-12,-10])
#train_IM = loss[:,2]#np.delete(loss[:,2], [max_index,-11,-12,-10])

test_pos = np.delete(test_loss[:,0], 0)
test_EC = np.delete(test_loss[:,1], 0)
test_IM = np.delete(test_loss[:,2], 0)
test_steps = test_steps[1:]


fig16, ax16 = plt.subplots(num=16,nrows=1,ncols=1, clear=True)
fig16.set_tight_layout(True)

ax16.scatter(test_steps/1000*test_every, test_pos, color='r',s=40)
ax16.scatter(test_steps/1000*test_every, test_EC, color='b',s=40)
ax16.scatter(test_steps/1000*test_every, test_IM, color='k',s=40)

ax16.plot(steps[::100]/1000, train_pos[::100], label='Poisson', linestyle='-',color='red',linewidth=3)
ax16.plot(steps[::100]/1000, train_EC[::100], label='Electron Continuity', linestyle='-',color='blue',linewidth=3)
ax16.plot(steps[::100]/1000, train_IM[::100], label='Ion Momentum', linestyle='-',color='k',linewidth=3)

'''x,y = fixloss(loss[:,4])
ax16.scatter(x/1000, y, label='P Test Loss ',color='r',linewidth=2)
x,y = fixloss(loss[:,5])
ax16.scatter(x/1000, y, label='EC Test Loss ', color='b',linewidth=2)
x,y = fixloss(loss[:,6])
ax16.scatter(x/1000, y, label='IM Test Loss', color='k',linewidth=2)
'''
#ax16.set_ylabel("$\\phi$")
ax16.set_xlabel("Epochs (x1000)")
ax16.set_title("Losses")
#ax16.set_ylim([1e-7,5e-4])
ax16.set_yscale("log")
ax16.legend()
fig16.savefig(Data_path + "Losses.png")

Tepts = np.linspace(TeMin,TeMax,numxpts)
TiTepts = np.linspace(TiTeMin,TiTeMax,numxpts)
S0pts = np.linspace(S0Min,S0Max,numxpts)
Knudpts = np.linspace(KnudMin,KnudMax,numxpts)
nnpts = np.linspace(nnMin,nnMax,numxpts)

engine = SobolEngine(dimension=5, scramble=True, seed=4321)
X_test = engine.draw(10000).to(device)

Y = pde(model, X_test)
if not isinstance(Y, (tuple, list)):
    Y = [Y]
Y = [res.detach().cpu().numpy() for res in Y]

X_test = X_test.detach().cpu().numpy()
X = X_test[:,0]
Tepts = X_test[:,1]
TiTepts = X_test[:,2]
S0pts = X_test[:,3]
#Knudpts = X_test[:,4]
nnpts = X_test[:,4]

floor = 1e-4

#space_res = residual_pts(xpts,Tepts,TiTepts,S0pts,Knudpts,nnpts)
#print(Y[0].shape)
poisson_res = np.abs(Y[0]) #>= floor
ec_res = np.abs(Y[1]) #>= floor
im_res = np.abs(Y[2]) #>= floor

#Residual Plots
fig17,ax17 = plt.subplots(num=17,nrows=1,ncols=1, clear=True)
fig17.set_tight_layout(True)

'''ax17.plot(xpts, space_res[0], color='r',label='Poisson Residual')
ax17.plot(xpts, space_res[1], color='b',label='EC Residual')
ax17.plot(xpts, space_res[2], color='k',label='IM Residual')'''

ax17.scatter(X, ec_res, color='r',label="Electron Continuity",s=0.01)
ax17.scatter(X, im_res, color='b',label="Ion Momentum",s=0.01)
ax17.scatter(X, poisson_res, color='k',label="Poisson",s=0.01)

ax17.set_xlabel("$x/\\lambda_{de}$")
ax17.set_ylabel("$T_e$")
ax17.set_yscale("log")
ax17.set_title("Poisson")
'''ax17[0,1].set_xlabel("$x/\\lambda_{de}$")
ax17[0,1].set_ylabel("$T_e$")
ax17[0,1].set_title("Electron Continuity")
ax17[1,0].set_xlabel("$x/\\lambda_{de}$")
ax17[1,0].set_ylabel("$T_e$")
ax17[1,0].set_title("Ion Momentum")'''
ax17.legend()
fig17.savefig(Data_path + "Residuals.png")
plt.close()

def plot(x,y,sizes,labels,titles):
    fig,ax = plt.subplots(num=20,nrows=2,ncols=2, clear=True)
    fig.set_tight_layout(True)

    ax[0,0].scatter(x,y,color='k',s=sizes[0])
    ax[0,1].scatter(x,y,color='k',s=sizes[1])
    ax[1,0].scatter(x,y,color='k',s=sizes[2])
    ax[0,0].set_xlabel(labels[0])
    ax[0,0].set_ylabel(labels[1])
    ax[0,0].set_title(titles[0])
    ax[0,1].set_xlabel(labels[0])
    ax[0,1].set_ylabel(labels[1])
    ax[0,1].set_title(titles[1])
    ax[1,0].set_xlabel(labels[0])
    ax[1,0].set_ylabel(labels[1])
    ax[1,0].set_title(titles[2])

    plt.savefig(Data_path + "Residuals_"+labels[1]+"vs"+labels[0]+".png")
    plt.close()

plot(X,Tepts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["X","Te"],["Poisson","Electron Continuity","Ion Momentum"])
plot(X,TiTepts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["X","TiTe"],["Poisson","Electron Continuity","Ion Momentum"])
plot(X,S0pts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["X","S0"],["Poisson","Electron Continuity","Ion Momentum"])
#plot(X,Knudpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["X","Knud"],["Poisson","Electron Continuity","Ion Momentum"])
plot(X,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["X","nn"],["Poisson","Electron Continuity","Ion Momentum"])
plot(Tepts,TiTepts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Te","TiTe"],["Poisson","Electron Continuity","Ion Momentum"])
plot(Tepts,S0pts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Te","S0"],["Poisson","Electron Continuity","Ion Momentum"])
#plot(Tepts,Knudpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Te","Knud"],["Poisson","Electron Continuity","Ion Momentum"])
plot(Tepts,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Te","nn"],["Poisson","Electron Continuity","Ion Momentum"])
plot(TiTepts,S0pts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["TiTe","S0"],["Poisson","Electron Continuity","Ion Momentum"])
#plot(TiTepts,Knudpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["TiTe","Knud"],["Poisson","Electron Continuity","Ion Momentum"])
plot(TiTepts,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["TiTe","nn"],["Poisson","Electron Continuity","Ion Momentum"])
#plot(S0pts,Knudpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["S0","Knud"],["Poisson","Electron Continuity","Ion Momentum"])
plot(S0pts,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["S0","nn"],["Poisson","Electron Continuity","Ion Momentum"])
#plot(Knudpts,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Knud","nn"],["Poisson","Electron Continuity","Ion Momentum"])

'''
xpts = np.linspace(0,1,50)
Tepts = np.linspace(0,1,20)
TiTepts = np.linspace(0,1,5)
S0pts = np.linspace(0,1,5)
Knudpts = np.linspace(0,1,10)
xnew, Tenew,TiTenew, Knudnew, S0new = np.meshgrid(xpts,Tepts,TiTepts,Knudpts,S0pts)
X2 = np.vstack((np.ravel(xnew),np.ravel(Tenew),np.ravel(TiTenew),np.ravel(Knudnew),np.ravel(S0new))).T
y2 = model.predict(X2,operator=pde)
#print(y2.shape)
resphi = y2[0]
resec = y2[1]
resim = y2[2]
#resnw = y2[3]

resphi = resphi.reshape(xnew.shape)
resec = resec.reshape(xnew.shape)
resim = resim.reshape(xnew.shape)
#resnw = resnw.reshape(xnew.shape)
resphinew = resphi.T
resecnew = resec.T
resimnew = resim.T
print(resphinew.shape)
print(resphi.shape)
fig17,ax17 = plt.subplots(num=17,nrows=2,ncols=2, clear=True)
fig17.set_tight_layout(True)
C  = ax17[0,0].contourf(xpts[:],Tepts,  resphi[:,:,0,0,0],levels=30,cmap='plasma')
C1 = ax17[0,1].contourf(xpts[:],TiTepts,resphinew[0,0,:,:,0],levels=30,cmap='plasma')
C2 = ax17[1,0].contourf(xpts[:],S0pts,  resphinew[:,0,0,:,0],levels=30,cmap='plasma')
C3 = ax17[1,1].contourf(xpts[:],Knudpts,resphinew[0,:,0,:,0],levels=30,cmap='plasma')

fig17.colorbar(C,ax=ax17[0,0])
fig17.colorbar(C1,ax=ax17[0,1])
fig17.colorbar(C2,ax=ax17[1,0])
fig17.colorbar(C3,ax=ax17[1,1])

ax17[0,0].set_title("Te")
ax17[0,1].set_title("TiTe")
ax17[1,0].set_title("S0")
ax17[1,1].set_title("Knud")

#plt.plot(xpts,resphi[0,:,0,0,0],color='r',label='phi')
#plt.plot(xpts,resec[0,:,0,0,0],color='g',label='ec')
#plt.plot(xpts,resim[0,:,0,0,0],color='b',label='im')
#plt.plot(xpts,resnw[0,:,0,0],color='k',label='nw')
#plt.legend()
fig17.savefig(Data_path + 'Residual_phi.png') 

fig18,ax18 = plt.subplots(num=18,nrows=2,ncols=2, clear=True)
fig18.set_tight_layout(True)
C  = ax18[0,0].contourf(xpts[:],Tepts,  resec[:,:,0,0,0],levels=30,cmap='plasma')
C1 = ax18[0,1].contourf(xpts[:],TiTepts,resecnew[0,0,:,:,0],levels=30,cmap='plasma')
C2 = ax18[1,0].contourf(xpts[:],S0pts,  resecnew[:,0,0,:,0],levels=30,cmap='plasma')
C3 = ax18[1,1].contourf(xpts[:],Knudpts,resecnew[0,:,0,:,0],levels=30,cmap='plasma')

fig18.colorbar(C,ax=ax18[0,0])
fig18.colorbar(C1,ax=ax18[0,1])
fig18.colorbar(C2,ax=ax18[1,0])
fig18.colorbar(C3,ax=ax18[1,1])

ax18[0,0].set_title("Te")
ax18[0,1].set_title("TiTe")
ax18[1,0].set_title("S0")
ax18[1,1].set_title("Knud")

#plt.plot(xpts,resphi[0,:,0,0,0],color='r',label='phi')
#plt.plot(xpts,resec[0,:,0,0,0],color='g',label='ec')
#plt.plot(xpts,resim[0,:,0,0,0],color='b',label='im')
#plt.plot(xpts,resnw[0,:,0,0],color='k',label='nw')
#plt.legend()
fig18.savefig(Data_path + 'Residual_ec.png') 

fig19,ax19 = plt.subplots(num=19,nrows=2,ncols=2, clear=True)
fig19.set_tight_layout(True)
C  = ax19[0,0].contourf(xpts[:],Tepts,  resim[:,:,0,0,0],levels=30,cmap='plasma')
C1 = ax19[0,1].contourf(xpts[:],TiTepts,resimnew[0,0,:,:,0],levels=30,cmap='plasma')
C2 = ax19[1,0].contourf(xpts[:],S0pts,  resimnew[:,0,0,:,0],levels=30,cmap='plasma')
C3 = ax19[1,1].contourf(xpts[:],Knudpts,resimnew[0,:,0,:,0],levels=30,cmap='plasma')

fig19.colorbar(C,ax=ax19[0,0])
#fig19.colorbar(C1,ax=ax19[0,1])
fig19.colorbar(C2,ax=ax19[1,0])
fig19.colorbar(C3,ax=ax19[1,1])

ax19[0,0].set_title("Te")
ax19[0,1].set_title("TiTe")
ax19[1,0].set_title("S0")
ax19[1,1].set_title("Knud")

fig19.savefig(Data_path + 'Residual_im.png') 
#x,Te,TiTe,S0,Knud, nn
'''
phi2list = []
ni2list = []
ne2list = []
ui2list = []
ue2list = []

#xpts = np.linspace(0,L,numxpts)

phi,ni,ne,ui,ue,_ = pred(xpts,TeMax,0,S0Max,nnMax)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)

'''phi,ni,ne,ui,ue = pred(xpts,20,0,1e31,0)
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
ui2list.append(ui)'''

'''# sample x grid inside domain (avoid endpoints)
x_check = np.linspace(0.01, L-1e-3, 200)
x_check_norm = (x_check - xMin) / (xMax - xMin) 
phi, ni, ne, ui, ue, newall = pred(x_check, TeMax, 0, S0Max, nnMax)
# numeric E from PINN (consistent sign with RK: E = -dphi/dx)
x_t = torch.tensor(x_check_norm, device=device, dtype=torch.float64).reshape(-1,1).requires_grad_(True)
X = feature_transform(torch.cat((x_t, x_t*0 + 1, x_t*0 + 0, x_t*0 + 1, x_t*0 + 1), dim=1))
X2 = feature_transform(torch.cat((x_t*0 +1,x_t*0 +1,x_t*0 +0,x_t*0 +1,x_t*0 +1), dim=1))
outs = model(X)
outs_t = output_transform(X,outs)
phi_t, ni_t, uet_t,_ = outs_t.split(1, dim=1)

outs2_t = output_transform(X2,model(X2))
newall_t = outs2_t[:,3:4]

ue_t = torch.expm1(uet_t)
ne_t = newall_t * torch.exp(phi_t)
ui_t = ne_t*ue_t/ni_t
E_fd = -torch.autograd.grad(phi_t, x_t, grad_outputs=torch.ones_like(phi_t), create_graph=True)[0] / (xMax - xMin)
dE_dx = torch.autograd.grad(E_fd, x_t, grad_outputs=torch.ones_like(E_fd), create_graph=True)[0] / (xMax - xMin)
dui_dx_t = torch.autograd.grad(ui_t, x_t, grad_outputs=torch.ones_like(ui_t), create_graph=True)[0] / (xMax - xMin)
due_dx_t = torch.autograd.grad(ue_t, x_t, grad_outputs=torch.ones_like(ue_t), create_graph=True)[0] / (xMax - xMin)
#E_fd = -np.gradient(phi, x_check)

E_fd = E_fd.detach().cpu().numpy().flatten()
dE_dx_fd = dE_dx.detach().cpu().numpy().flatten()
dui_dx_fd = dui_dx_t.detach().cpu().numpy().flatten()
due_dx_fd = due_dx_t.detach().cpu().numpy().flatten()

# compute barSion per x (uses same formulas as RK)
uref = np.sqrt((TeMax*q/mi))
lref = (Eps0*TeMax*uref/(S0Max*L*q))**(1/3)
nref = S0Max*L*lref/uref
barSion = Sion2(TeMax, 15.8)*(lref*nref/uref)

# compute ne as used in RK (nw assumed from pred at wall X2)
nw = newall[0] if np.ndim(newall)>0 else newall
ne_from_nw = nw * np.exp(phi)

# build RK ODE RHS arrays (vectorized)
dphi_dx_rk = -E_fd
dE_dx_rk = ni - ne_from_nw
dui_dx_rk = E_fd/ui - ui/(ne_from_nw*ue*L) - nnMax*ui*barSion/(ue)
due_dx_rk = 1/(ne_from_nw*L)  + ue*E_fd + nnMax*barSion

# Now compute PDE-derived RHS from PINN (if you derive different algebraic form), or compare to finite-diff of PINN fields:
# finite-diff of ui and ni (for internal consistency)
#dui_dx_fd = np.gradient(ui, x_check)
#dni_dx_fd = np.gradient(ni, x_check)

# report diagnostics
print("Max abs differences (RK_RHS vs PINN finite-diff):")
print("dphi:", np.max(np.abs(dphi_dx_rk + E_fd)))    # should be ~0 (sign check)
print("dE:",   np.max(np.abs(dE_dx_rk - dE_dx_fd)))  # second derivative check
print("dui:",  np.max(np.abs(dui_dx_rk - dui_dx_fd)))
print("dni:",  np.max(np.abs(due_dx_rk - due_dx_fd)))

# also print some extreme values to locate bad terms
print("phi range:", phi.min(), phi.max())
print("ni range:", ni.min(), ni.max())
print("ne_from_nw range:", ne_from_nw.min(), ne_from_nw.max())
print("ui range:", ui.min(), ui.max())
print("nw (used):", nw, "barSion:", barSion)
'''

# ...after you have model loaded and pred defined...

# choose a few x points inside domain
x_check = np.linspace(0.01, L-1e-3, 10)

# get PINN outputs (phi, ni, ne, ui, ue, newall)
phi_t, ni_t, ne_t, ui_t, ue_t, newall_t = pred(x_check, TeMax, 0, S0Max, nnMax)

# compute barSion with numpy (same formula as RK)
uref = np.sqrt((TeMax*q/mi))
lref = (Eps0*TeMax*uref/(S0Max*L*q))**(1/3)
nref = S0Max*L*lref/uref
barSion_np = Sion2(TeMax, 15.8) * (lref * nref / uref)

# compute the term used in PINN: nn * ne * barSion  (vector)
term_pinn = nnMax * ne_t * barSion_np

# compute same term using torch implementation (callable analog)
Te_torch = torch.tensor(TeMax, dtype=torch.float64, device=device)
uref_t = torch.sqrt((Te_torch)*q/mi)
lref_t = (Eps0*Te_torch*uref_t/(S0Max*L*q))**(1/3)
nref_t = S0Max*L*lref_t/uref_t
barSion_t = Sion(Te_torch, torch.tensor(15.8, dtype=Te_torch.dtype, device=device)) * (lref_t * nref_t / uref_t)

# compare numeric values
print("barSion_np:", barSion_np)
print("barSion_t:", barSion_t.item())
print("max abs diff on term nn*ne*barSion:", np.max(np.abs(term_pinn - (nnMax * ne_t * barSion_t.item()))))

x0 = 0.01
S = 1/L

def runge(Te,S0,nn):
    #MiMe = 1836
    #Te = 1
    TiTe=0
    un = 0
    #nn = 1
    #Knud = 0.3
    E0 = 0
    phi,ni,ne,ui,ue,newall = pred(x0,Te,TiTe,S0,nn)
    #print("newall:", phi)
    #print("ne (x=L):", ne[-1])
    #exit()
    phi0 = phi[0]
    ui0 = ui[0] 
    ue0 = ue[0] 
    ni0 = ni[0]
    ne0 = ne[0]
    nw = newall[0]
    np.savetxt(Data_path + "initial.txt", [phi0,ui0,ue0,nw])

    uref = np.sqrt(Te*q/mi)
    lref = (Eps0*Te*uref/(S0*L*q))**(1/3)
    nref = S0*L*lref/uref
    barSion = Sion2(Te,15.8)*(lref*nref/uref)
    #print("tests:", uref, lref, nref)

    '''phi0 = float(phi0)
    ui0 = float(ui0)
    ni0 = float(ni0)
    nw = float(nw)
    Te_local = float(Te)
    S0_local = float(S0)
    nn_local = float(nn)

    # compute uref, lref, nref, barSion as in PDE
    uref = (Te_local*q/mi)**0.5
    lref = (Eps0*Te_local*uref/(S0_local*L*q))**(1/3)
    nref = S0_local*L*lref/uref
    barSion_rk = Sion2(Te_local, 15.8)*(lref*nref/uref)   # your RK45 uses Sion2 (numpy)
    Sauce_rk = 1.0/L + nn_local * (nw * np.exp(phi0)) * barSion_rk

    # build RHS from RK45 function
    ne0 = nw * np.exp(phi0)
    dphi_dx_rk = -E0  # E0 is your initial E (you set E0=0)
    dE_dx_rk = ni0 - ne0
    dui_dx_rk = dphi_dx_rk / ui0 - Sauce_rk/ni0
    dni_dx_rk = 2*Sauce_rk/ui0 - ni0 * dphi_dx_rk / ui0**2

    print("Initial values: phi0, E0, ui0, ni0, ne0, nw =", phi0, E0, ui0, ni0, ne0, nw)
    print("RK45 Sauce:", Sauce_rk, "barSion:", barSion_rk)
    print("RK45 rhs:", dphi_dx_rk, dE_dx_rk, dui_dx_rk, dni_dx_rk)

    # now compute the PDE RHS-derived expressions (from algebra) and compare:
    # from PDE: dui_dx_pde = E/ui - (1/ni)*(1/L + nn*ne*barSion)
    dui_dx_pde = dphi_dx_rk / ui0 - (1.0/ni0) * (1.0/L + nn_local*ne0*barSion_rk)
    # dni_dx_pde computed using the identity dni = 2*Sauce/ui - ni*E/ui^2 (same as RK45)
    dni_dx_pde = 2*(1.0/L + nn_local*ne0*barSion_rk)/ui0 - ni0*dphi_dx_rk/ui0**2

    print("PDE-derived rhs:", dphi_dx_rk, dE_dx_rk, dui_dx_pde, dni_dx_pde)
    exit()'''

    '''def system(x,y): 
        phi = y[0] 
        E = y[1]
        ui = y[2]
        ni = y[3]
        ne = nw*np.exp(phi)
        
        uref = (Te*q/mi)**(1/2)
        lref = (Eps0*Te*uref/(S0*L*q))**(1/3)
        nref = S0*L*lref/uref
        
        #barS0 = S0*lref**4*L/uref
        #Tet = torch.tensor(Te,dtype=torch.float64).to(device)
        barSion = Sion2(Te,15.8)*(lref*nref/uref)
        #print("tests:",uref,lref,nref,nw,barSion)
        #barSrec = Srec(Te,13.6,1)/(lref**2*uref)
        Sauce = S + nn*ne*barSion #*barS0 - ni*ne*barSrec*barS0
        
        
        dphi_dx = -E 
        dE_dx = ni-ne
        #dui_dx = E/ui - Sauce/ni #- Knud 
        dui_dx = E/ui - 1/(ni*L) - nn*ne*barSion/ni
        #dni_dx = 2*Sauce/ui - ni*E/ui**2 #+ ni/ui*Knud
        dni_dx = 2/(ui*L) + nn*2*ne*barSion/ui - ni*E/ui**2
        return dphi_dx, dE_dx, dui_dx, dni_dx'''
        
    '''
    def system(x,y): 
        phi = y[0] 
        E = y[1]
        ui = y[2]
        ue = y[3]
        ne = nw*np.exp(phi)
        #ni = ne*ue/ui
        
        uref = (Te*q/(mi))**(1/2)
        lref = (Eps0*Te*uref/(S0*L*q))**(1/3)
        nref = S0*L*lref/uref
        
        #barS0 = S0*lref**4*L/uref
        barSion = Sion2(Te,15.8)*(lref*nref/uref)
        print("tests:",uref,lref,nref,barSion)
        exit()
        #barSrec = Srec2(Te,13.6,1)/(lref**2*uref)
        Sauce = S + nn*ne*barSion #- ue*ne**2/ui*barSrec*barS0
        
        dphi_dx = -E 
        dE_dx = ne*(ue/ui-1)
        #dui_dx = E/ui - ui*Sauce/(ue*ne) #- nn*1*signn(0.5*(Te+TiTe*Te),lref,nref) #(ui-un)/ui = 1 #C_s/uref = 1
        dui_dx = E/ui - ui/(ne*ue*L) - nn*ui*barSion/(ue)
        #due_dx = Sauce/ne + E*ue
        due_dx = 1/(ne*L)  + ue*E + nn*barSion
        return dphi_dx, dE_dx, dui_dx, due_dx
        '''

    '''xlist = [x0] 
    philist = [phi0]
    Elist = [E0]
    uilist = [ui0]
    uelist = [ue0]
    nilist = [ni0]

    qwerk = scipy.integrate.RK45(system,x0,[phi0,E0,ui0,ni0],51,rtol = 1e-12, atol = 1e-30)
    while qwerk.status == 'running' and qwerk.y[0] > 0:
        qwerk.step()
        xlist.append(qwerk.t)
        philist.append(qwerk.y[0])
        Elist.append(qwerk.y[1])
        uilist.append(qwerk.y[2])
        nilist.append(qwerk.y[3])
        uelist.append((np.array(qwerk.y[2])*np.array(qwerk.y[3]))/(nw * np.exp(np.array(qwerk.y[0]))))
        #print(qwerk.status)
    ne = nw * np.exp(np.array(philist))'''
    #ni = np.array(uelist)*ne/np.array(uilist)

    def system(x, y):
        phi = y[0]
        E = y[1]
        ui = y[2]
        ue = y[3]
        ne = nw * np.exp(phi)

        Sauce = S + nn*ne*barSion

        dphi_dx = -E 
        dE_dx = ne*(ue/ui-1)
        #dui_dx = E/ui - ui*Sauce/(ue*ne) #- nn*1*signn(0.5*(Te+TiTe*Te),lref,nref) #(ui-un)/ui = 1 #C_s/uref = 1
        dui_dx = E/ui - ui/(ne*ue*L) - nn*ui*barSion/(ue) - nn*signn(0.5*(Te),lref,nref)
        #due_dx = Sauce/ne + E*ue
        due_dx = 1/(ne*L)  + ue*E + nn*barSion#*ue
        return np.array([dphi_dx, dE_dx, dui_dx, due_dx])

    xlist = [x0] 
    philist = [phi0]
    Elist = [E0]
    uilist = [ui0]
    uelist = [ue0]
    nilist = [ni0]

    # event to stop when phi -> 0 (terminal)
    def phi_event(x, y):
        return y[0]
    phi_event.terminal = True
    phi_event.direction = -1  # stop when phi decreases through zero

    # integrate with Radau (stiff solver)
    from scipy.integrate import solve_ivp
    t_span = (x0, L)            # integrate from x0 to L (or a suitable upper bound)
    y0 = [phi0, E0, ui0, ue0]

    sol = solve_ivp(system, t_span, y0,
                    method='Radau',
                    rtol=1e-12, atol=1e-30,
                    events=phi_event,
                    max_step=0.1)   # adjust max_step as needed

    xlist = sol.t.tolist()
    philist = sol.y[0].tolist()
    Elist = sol.y[1].tolist()
    uilist = sol.y[2].tolist()
    #nilist = sol.y[3].tolist()
    uelist = sol.y[3].tolist()
    #uelist = (sol.y[2] * sol.y[3]) / (nw * np.exp(sol.y[0]))
    ne = nw * np.exp(np.array(philist))
    nilist = (sol.y[3] * ne) / sol.y[2]

    np.savetxt(Data_path + "x.txt", xlist)
    np.savetxt(Data_path + "ni.txt", nilist)
    np.savetxt(Data_path + "ne.txt", ne)
    np.savetxt(Data_path + "E.txt", Elist)
    np.savetxt(Data_path + "ui.txt", uilist)
    np.savetxt(Data_path + "ue.txt", uelist)
    np.savetxt(Data_path + "input.txt", [Te,S0,nn])
        
    return xlist, philist, uilist, Elist, uelist, ne, nilist

phiRKlist = []
xRKlist = []
neRKlist = []
uiRKlist = []
ueRKlist = []
niRKlist = []
#Te,S0,Knud,nn
xl, phi, ui, E, ue, ne, ni = runge(TeMax,S0Max,nnMax)
phiRKlist.append(phi)
neRKlist.append(ne)
xRKlist.append(xl)
ueRKlist.append(ue)
uiRKlist.append(ui)
niRKlist.append(ni)

'''xl, phi, ui, E, ue, ne, ni = runge(20,0,1e31,0)
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
uiRKlist.append(ui)'''

#ni = ne*np.array(uelist)/(uilist)

print(len(xRKlist[0]))
ptsize = 100
plotonly = 6001
plotevery = 500
#xpts = np.linspace(0,L,numxpts)
fig20, ax20 = plt.subplots(num=20,nrows=1,ncols=1, clear=True)
fig20.set_tight_layout(True)
x = xRKlist[0]
phi = phi2list[0]
phiRK = phiRKlist[0]
#print(phi)
#exit()
ax20.scatter(x[-plotonly:][::plotevery],phiRK[-plotonly:][::plotevery],color='k',label="RK45",marker='x',s=ptsize)
ax20.plot(xpts,phi,color='k',linestyle='-',label='PINN',linewidth=2)
ax20.legend()
ax20.set_title("$e\\phi /T$")
ax20.set_xlabel("$\\lambda_{de}$")
fig20.savefig(Data_path + "phiRK.png")
plt.close()

#print(x)
fig21, ax21 = plt.subplots(num=21,nrows=1,ncols=1, clear=True)
fig21.set_tight_layout(True)
ui = ui2list[0]
uiRK = uiRKlist[0]
ax21.scatter(x[-plotonly:][::plotevery],uiRK[-plotonly:][::plotevery],color='k',label="RK45",marker='x',s=ptsize)
ax21.plot(xpts,ui,color='k',linestyle='-',label='PINN',linewidth=2)
ax21.legend()
ax21.set_title("Ion Velocity")
ax21.set_xlabel('$\\lambda_{De}$')
fig21.savefig(Data_path + "uiRK.png")
plt.close()

#print(x)
fig22, ax22 = plt.subplots(num=22,nrows=1,ncols=1, clear=True)
fig22.set_tight_layout(True)
ue = ue2list[0]
ueRK = ueRKlist[0]
ax22.scatter(x[-plotonly:][::plotevery],ueRK[-plotonly:][::plotevery],color='k',label="RK45",marker='x',s=ptsize)
ax22.plot(xpts,ue,color='k',linestyle='-',label='PINN',linewidth=2)
ax22.legend()
ax22.set_title("Electron Velocity")
ax22.set_xlabel('$\\lambda_{De}$')
fig22.savefig(Data_path + "ueRK.png")
plt.close()

fig23, ax23 = plt.subplots(num=23,nrows=1,ncols=1, clear=True)
fig23.set_tight_layout(True)
ne = ne2list[0]
ni = ni2list[0]
neRK = neRKlist[0]
niRK = niRKlist[0]
ax23.scatter(x[-plotonly:][::plotevery],neRK[-plotonly:][::plotevery],color='b',label="ne-RK45",marker='x',s=ptsize)
ax23.scatter(x[-plotonly:][::plotevery],niRK[-plotonly:][::plotevery],color='r',label="ni-RK45",marker='x',s=ptsize)
ax23.plot(xpts,ne,color='b',linestyle='-',label='ne-PINN',linewidth=2)
ax23.plot(xpts,ni,color='r',linestyle='-',label='ni-PINN',linewidth=2)
ax23.legend()
ax23.set_title("Density")
ax23.set_xlabel('$\\lambda_{De}$')
fig23.savefig(Data_path + "DensitiesRK.png")
plt.close()

'''fig23, ax23 = plt.subplots(num=23,nrows=1,ncols=1, clear=True)
fig23.set_tight_layout(True)
x = xRKlist[1]
phi = phi2list[1]
phiRK = phiRKlist[1]
ax23.plot(x,phiRK,color='b',label="RK45")
ax23.plot(xpts,phi,color='g',linestyle='--',label='PINN')
ax23.legend()
ax23.set_title("Phi")
ax23.set_xlabel('$\\lambda_{De}$')
fig23.savefig(Data_path + "phiRK1.png")
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
fig24.savefig(Data_path + "uiRK1.png")
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
fig25.savefig(Data_path + "ueRK1.png")
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
fig26.savefig(Data_path + "phiRK2.png")
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
fig27.savefig(Data_path + "uiRK2.png")
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
fig28.savefig(Data_path + "ueRK2.png")
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
fig29.savefig(Data_path + "phiRK3.png")
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
fig30.savefig(Data_path + "uiRK3.png")
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
fig31.savefig(Data_path + "ueRK3.png")
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
fig32.savefig(Data_path + "phiRK4.png")
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
fig33.savefig(Data_path + "uiRK4.png")
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
fig34.savefig(Data_path + "ueRK4.png")
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
fig35.savefig(Data_path + "phiRK5.png")
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
fig36.savefig(Data_path + "uiRK5.png")
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
fig37.savefig(Data_path + "ueRK5.png")
plt.close()'''
'''
numTepts = 20
numS0pts = 10
numKnudpts = 30
TeScan = np.linspace(TeMin,TeMax,numTepts)
S0Scan = np.linspace(S0Min, S0Max, numS0pts)
KnudScan = np.logspace(-4,-1, 15)
KnudScan = np.append(KnudScan, np.linspace(0.1, KnudMax, 15))
phivec = np.zeros([numTepts,numS0pts,numKnudpts])
uivec = np.zeros([numTepts,numS0pts,numKnudpts])
uevec = np.zeros([numTepts,numS0pts,numKnudpts])
nevec = np.zeros([numTepts,numS0pts,numKnudpts])
nivec = np.zeros([numTepts,numS0pts,numKnudpts])
TevecRK = np.zeros([numTepts,numS0pts,numKnudpts])
S0vecRK = np.zeros([numTepts,numS0pts,numKnudpts])
KnudvecRK = np.zeros([numTepts,numS0pts,numKnudpts])
xSEvecRK = np.zeros([numTepts,numS0pts,numKnudpts])

def sheathedge(Te,S0,Knud,x):
    xl, phi, ui, E, ue, ne, ni = runge(Te,S0,Knud)
    for i in range(len(ne)):
        tol = 1e-3
        dif = np.abs((ni[i]-ne[i]) / ne[i] - Prho)
        if dif <= tol:
            sheathenter = xl[i]
            print(dif,sheathenter)
    initguess = np.array([x for i in range(len(xl))])
    sheathx = np.where(np.isclose(xl,initguess,rtol=5e-4,atol=1e-6))[0][0]
    
    #dif = np.abs((ni[sheathx:]-ne[sheathx:]) / ne[sheathx:] - Prho)
    #sheathenter = np.where(min(dif)==dif)[0][0]
    return xl, phi, ui, E, ue, ne, ni,sheathx
counts = 0

for i in range(numTepts):
    for j in range(numS0pts):
        for k in range(numKnudpts):
            sol = optimize.root(SheathEntrance, [L-5], args = (TeScan[i],0,S0Scan[j],KnudScan[k]))
            xSE = sol.x
            #print("sheath:",xSE)
            #phiSE, niSE, neSE, uiSE, ueSE = pred1(xSE,MiMeScan[i],TiTeScan[j],KnudScan[k])
            xl, phi, ui, E, ue, ne, ni,edgepnt = sheathedge(TeScan[i],S0Scan[j],KnudScan[k],xSE[0])

            phivec[i,j,k] = phi[edgepnt]
            uivec[i,j,k] = ui[edgepnt]
            uevec[i,j,k] = ue[edgepnt]
            nivec[i,j,k] = ni[edgepnt] / ni[0]
            nevec[i,j,k] = ne[edgepnt] / ne[0]

            TevecRK[i,j,k] = TeScan[i]
            S0vecRK[i,j,k] = S0Scan[j]
            KnudvecRK[i,j,k] = KnudScan[k] 
            xSEvecRK[i,j,k] = xl[edgepnt]

            counts = counts +1
            if counts%100 == 0:
                print("Percent done:", counts*100/(numTepts*numKnudpts*numS0pts),'%')

                
fig40, ax40 = plt.subplots(num=40,nrows=1,ncols=1, clear=True)
fig40.set_tight_layout(True)

ax40.plot(KnudvecRK[0,0,:], phivec[0,0,:], 'c', label='Te=1eV, Ti=0',linewidth=3)
ax40.plot(KnudvecRK[0,0,:], phivec[-1,0,:], 'm', label='Te=20eV, Ti=0',linewidth=3)
ax40.plot(Knudvec[0,0,:,0], phiSEvec[0,0,:,0], 'k', label='Te=1eV, Ti=0',linewidth=3)
ax40.plot(Knudvec[0,0,:,0], phiSEvec[0,-1,:,0], 'b', label='Te=1eV, Ti=Te',linewidth=3)
ax40.plot(Knudvec[0,0,:,0], phiSEvec[-1,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
ax40.plot(Knudvec[0,0,:,0], phiSEvec[-1,-1,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)
#ax7.set_ylabel("$\\phi$")
ax40.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax40.set_title("Potential Drop [$e\\phi / T$]")
ax40.set_xscale("log")
#ax8.legend()
fig40.savefig(Data_path + "Potential_DropRK.png")

fig41, ax41 = plt.subplots(num=41,nrows=1,ncols=1, clear=True)
fig41.set_tight_layout(True)

ax41.plot(KnudvecRK[0,0,:], uivec[0,0,:], 'c', label='Te=1eV, Ti=0',linewidth=3)
ax41.plot(KnudvecRK[0,0,:], uivec[-1,0,:], 'm', label='Te=20eV, Ti=0',linewidth=3)
ax41.plot(Knudvec[0,0,:,0], uiSEvec[0,0,:,0], 'k', label='Te=1eV, Ti=0',linewidth=3)
ax41.plot(Knudvec[0,0,:,0], uiSEvec[0,-1,:,0], 'b', label='Te=1eV, Ti=Te',linewidth=3)
ax41.plot(Knudvec[0,0,:,0], uiSEvec[-1,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
ax41.plot(Knudvec[0,0,:,0], uiSEvec[-1,-1,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)

#ax7.set_ylabel("$\\phi$")
ax41.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax41.set_title("Ion Speed at Sheath Entrance [$C_s$]")
ax41.set_xscale("log")
#ax9.legend()
fig41.savefig(Data_path + "Bohm_CriterionRK.png")

#Sheath_width = np.array([L-xSE1,L-xSE2,L-xSE3,L-xSE4,L-xSE5])
fig42, ax42 = plt.subplots(num=42,nrows=1,ncols=1, clear=True)
fig42.set_tight_layout(True)

ax42.plot(KnudvecRK[0,0,:], L-xSEvecRK[0,0,:], 'c', label='Te=1eV, Ti=0',linewidth=3)
ax42.plot(KnudvecRK[0,0,:], L-xSEvecRK[-1,0,:], 'm', label='Te=20eV, Ti=0',linewidth=3)
ax42.plot(Knudvec[0,0,:,0], L-xSEvec[0,0,:,0], 'k', label='Te=1eV, Ti=0',linewidth=3)
ax42.plot(Knudvec[0,0,:,0], L-xSEvec[0,-1,:,0], 'b', label='Te=1eV, Ti=Te',linewidth=3)
ax42.plot(Knudvec[0,0,:,0], L-xSEvec[-1,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
ax42.plot(Knudvec[0,0,:,0], L-xSEvec[-1,-1,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)

#ax9.set_ylabel("$\\phi$")
#ax42.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax42.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax42.set_title("Sheath Width [$\\lambda_{De}$]")
ax42.set_xscale('log')
#ax11.legend()
fig42.savefig(Data_path + "Sheath_WidthRK")

fig47, ax47 = plt.subplots(num=47,nrows=1,ncols=1, clear=True)
fig47.set_tight_layout(True)

ax47.plot(KnudvecRK[0,0,:], nivec[0,0,:], 'c', label='Te=1eV, Ti=0',linewidth=3)
ax47.plot(KnudvecRK[0,0,:], nivec[-1,0,:], 'm', label='Te=20eV, Ti=0',linewidth=3)
ax47.plot(Knudvec[0,0,:,0], niSEvec[0,0,:,0], 'k', label='Te=1eV, Ti=0',linewidth=3)
ax47.plot(Knudvec[0,0,:,0], niSEvec[0,-1,:,0], 'b', label='Te=1eV, Ti=Te',linewidth=3)
ax47.plot(Knudvec[0,0,:,0], niSEvec[-1,0,:,0], 'r', label='Te=20eV, Ti=0',linewidth=3)
ax47.plot(Knudvec[0,0,:,0], niSEvec[-1,-1,:,0], 'g', label='Te=20eV, Ti=Te',linewidth=3)
#ax9.set_ylabel("$\\phi$")
#ax47.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax47.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
ax47.set_title("Edge-to-Center Density [$\\lambda_{De}$]")
ax47.set_xscale('log')
ax47.legend()
fig47.savefig(Data_path + "Density_ratioRK")

fig43, ax43 = plt.subplots(num=43,nrows=1,ncols=1, clear=True)
fig43.set_tight_layout(True)

ax43.plot(TiTevec[0,:,0,0], phiSEvec[0,:,0,0], 'r', label='Te=1eV, Ti=0',linewidth=3)
ax43.plot(TiTevec[0,0,0,0], phivec[0,0,0] , 'ok',linestyle='--', label='Te=1eV, Ti=0', ms=8)
ax43.set_ylim([4,4.75])
#ax7.set_ylabel("$\\phi$")
ax43.set_xlabel("$T_{i}/T_{e}$")
ax43.set_title("Potential Drop [$e\\phi /T$]")
#ax7.set_xscale("log")
#ax8.legend()
fig43.savefig(Data_path + "Potential_DropTiRK.png")


fig44, ax44 = plt.subplots(num=44,nrows=1,ncols=1, clear=True)
fig44.set_tight_layout(True)

ax44.plot(TiTevec[0,:,0,0], uiSEvec[0,:,0,0], 'r', label='Te=1eV, Ti=0',linewidth=3)
ax44.plot(TiTevec[0,0,0,0], uivec[0,0,0], 'ok',linestyle='--', label='Te=1eV, Ti=0',ms=8)
ax44.set_ylim([0.75,1.5])
#ax7.set_ylabel("$\\phi$")
ax44.set_xlabel("$T_{i}/T_{e}$")
ax44.set_title("Ion Speed at Sheath Entrance [$C_s$]")
#ax7.set_xscale("log")
#ax8.legend()
fig44.savefig(Data_path + "Bohm_CriteriaTiRK.png")

fig45, ax45 = plt.subplots(num=45,nrows=1,ncols=1, clear=True)
fig45.set_tight_layout(True)

ax45.plot(TiTevec[0,:,0,0], L-xSEvec[0,:,0,0], 'r', label='Te=1eV, Ti=0',linewidth=3)
ax45.plot(TiTevec[0,0,0,0], L-xSEvecRK[0,0,0], 'ok',linestyle='--', label='Te=1eV, Ti=0',ms=8)
ax45.set_ylim([6,7])
#ax7.set_ylabel("$\\phi$")
ax45.set_xlabel("$T_{i}/T_{e}$")
ax45.set_title("Sheath Width [$\\lambda_{De}$]")
#ax7.set_xscale("log")
#ax8.legend()
fig45.savefig(Data_path + "Sheath_WidthTiRK.png")

fig46, ax46 = plt.subplots(num=46,nrows=1,ncols=1, clear=True)
fig46.set_tight_layout(True)

ax46.plot(TiTevec[0,:,0,0], niSEvec[0,:,0,0], 'r', label='PINN',linewidth=3)
ax46.plot(TiTevec[0,0,0,0], nivec[0,0,0], 'ok',linestyle='--', label='RK45',ms=8)
ax46.set_ylim([0.5,0.6])
#ax7.set_ylabel("$\\phi$")
ax46.set_xlabel("$T_{i}/T_{e}$")
ax46.set_title("Edge-to-Center Density")
#ax7.set_xscale("log")
ax46.legend()
fig46.savefig(Data_path + "Density_ratioTiRK.png")'''