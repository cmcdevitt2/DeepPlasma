import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize
import scipy.integrate
import math

from MagicNeutrals import *

sheathbool = False #Turns off/on sheath entrance calculations as well as graphs that use those calculations

Data_path = '/path/to/saved/model/and/for/figure/saving/'
ckpt_save_path = Data_path + "model_final.pt"

loss = np.loadtxt(Data_path + 'loss_history.txt')
test_loss = np.loadtxt(Data_path + 'test_history.txt')
steps = np.arange(0,len(loss[:,0]),1)
test_steps = np.arange(0,len(test_loss[:,0]),1)
Data_path = Data_path + 'figures/'

numxpts = 1000 # only used when model is loaded

#Parameters at which to evaluate the model
TeVal1 = TeMax
TiTeVal1 = TiTeMin
S0Val1 = S0Min
nnVal1 = nnMin

TeVal2 = TeMax
TiTeVal2 = TiTeMin
S0Val2 = S0Min
nnVal2 = nnMax

TeVal3 = TeMax
TiTeVal3 = TiTeMin
S0Val3 = S0Max
nnVal3 = nnMax

TeVal4 = TeMax
TiTeVal4 = TiTeMin
S0Val4 = S0Max
nnVal4 = nnMax

TeVal5 = TeMax
TiTeVal5 = TiTeMax
S0Val5 = S0Max
nnVal5 = 2

#Combine to form lists
TeVec = np.array([TeVal1,TeVal2,TeVal3,TeVal4,TeVal5])
TiTeVec = np.array([TiTeVal1,TiTeVal2,TiTeVal3,TiTeVal4,TiTeVal5])
S0Vec = np.array([S0Val1,S0Val2,S0Val3,S0Val4,S0Val5])
nnVec = np.array([nnVal1,nnVal2,nnVal3,nnVal4,nnVal5])

def Sion2(T,Ez): 
    #Fit to ionization rate from NRL Formulary
    #This version is for numpy arrays and other non-tensor variables
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * np.exp(-Ez/T) #m^3/s

def Srec2(T,Ez,Z):
    #Fit to recombination rate from NRL Formulary
    #This version is for numpy arrays and other non-tensor variables
    return 5.2e-20 * Z * (Ez/T)**(1/2) * ( 0.43 + 1/2*np.log(Ez/T) + 0.469*(Ez/T)**(-1.3) ) #m^3/s

interiorpts = [2,2]

Prho = 0.0685 

#Load the model
model.load_state_dict(torch.load(ckpt_save_path, map_location=device))

xpts = np.linspace(0,L,numxpts)

def pred(xVal,Te,TiTe,S0,nn):
    #Function to predict the variable profiles given a set of inputs
    #Arguments are normalized and need to be scaled to fit the network inputs
    model.eval()
    TeNorm = (Te - TeMin ) / ( TeMax - TeMin )
    TiTeNorm = (TiTe-TiTeMin) / (TiTeMax-TiTeMin) 
    S0Norm = (S0 - S0Min ) / ( S0Max - S0Min )
    nnNorm = (nn - nnMin) / (nnMax - nnMin)
    xNorm = (xVal-xMin)/(xMax-xMin)

    #create a tensor for all space and for the right boundary (x=L)
    x = torch.tensor(xNorm,device=device).reshape(-1,1)
    x2 = torch.tensor(1,device=device).reshape(-1,1)
    X = torch.cat((x,TeNorm + x*0, TiTeNorm + x*0, S0Norm + x*0, nnNorm + x*0),dim=1)
    X2 = torch.cat((x2,TeNorm +x2*0, TiTeNorm +x2*0, S0Norm +x2*0, nnNorm +x2*0),dim=1)

    #Pass the inputs through the network to get the outputs
    with torch.no_grad():
        X_trans = feature_transform(X)
        outputs = model(X_trans)
        outputs_trans = output_transform(X, outputs).cpu().numpy()

        outputs2 = model(feature_transform(X2))
        outputs2_trans = output_transform(X2, outputs2).cpu().numpy()

    #profiles across space
    phi, ni, uet = outputs_trans[:,0], outputs_trans[:,1], outputs_trans[:,2]

    #electron density at the wall, which was learning by the network
    newall = outputs2_trans[:,3]

    #compute electron density and un-transform ue to get the normalized electron velocity
    ne = newall*np.exp(phi)
    ue = np.expm1(uet)
    ui = ne*ue/ni

    return phi,ni,ne,ui,ue,newall

def res(test_pts):
    #Compute residuals for test_pts
    model.zero_grad() 
    residuals = pde(model, test_pts)
   
    if not isinstance(residuals, (tuple, list)):
        residuals = [residuals]
    losses = [torch.mean(res**2).detach().cpu().numpy() for res in residuals]
    loss = sum(losses)

    return losses, loss

def SheathEntrance(xVal,TeVal,TiTeVal,S0Val,nnVal):
    #Compute the sheath entrance
    _, ni, ne, _, _,_ = pred(xVal,TeVal,TiTeVal,S0Val,nnVal)

    return (ni-ne) / ne - Prho

#Build arrays of input parameters to explore
numTepts = 10
numTiTepts = 10
numS0pts = 10
numnnpts = 10
TeScan = np.linspace(TeMin,TeMax,numTepts)
TiTeScan = np.linspace(TiTeMin, TiTeMax, numTiTepts)
S0Scan = np.linspace(S0Min, S0Max, numS0pts)
nnScan = np.linspace(nnMin, nnMax, numnnpts)
xSEvec = np.zeros([numTiTepts,numnnpts])
phiSEvec = np.zeros([numTiTepts,numnnpts])
uiSEvec = np.zeros([numTiTepts,numnnpts])
neSEvec = np.zeros([numTiTepts,numnnpts])
TiTevec = np.zeros([numTiTepts,numnnpts])
nnvec = np.zeros([numTiTepts,numnnpts])
niSEvec = np.zeros([numTiTepts,numnnpts])


phiList = []
neList = []
niList = []
ueList = []
uiList = []

#predict the sheath profiles for the given set of inputs
phi, ni, ne, ui, ue,_ = pred(xpts,TeVal1,TiTeVal1,S0Val1,nnVal1)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_= pred(xpts,TeVal2,TiTeVal2,S0Val2,nnVal2)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_ = pred(xpts,TeVal3,TiTeVal3,S0Val3,nnVal3)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_ = pred(xpts,TeVal4,TiTeVal4,S0Val4,nnVal4)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

phi, ni, ne, ui, ue,_ = pred(xpts,TeVal5,TiTeVal5,S0Val5,nnVal5)

phiList.append(phi)
neList.append(ne)
niList.append(ni)
ueList.append(ue)
uiList.append(ui)

def residual_pts(x,Te,TiTe,S0,nn):
    #predict the residuals for the trained network at a given set of input parameters
    xNorm = (x - xMin) / (xMax - xMin)
    TeNorm = (Te - TeMin ) / ( TeMax - TeMin )
    TiTeNorm = (TiTe - TiTeMin) / (TiTeMax - TiTeMin)
    S0Norm = (S0 - S0Min ) / ( S0Max - S0Min )
    nnNorm = (nn - nnMin) / ( nnMax - nnMin )

    x_t = torch.tensor(xNorm,device=device).reshape(-1,1)
    Te_t = torch.tensor(TeNorm,device=device).reshape(-1,1)
    TiTe_t = torch.tensor(TiTeNorm,device=device).reshape(-1,1)
    S0_t = torch.tensor(S0Norm,device=device).reshape(-1,1)
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
    #compute the mean free path of the ions
    uref = np.sqrt((Te+Te*TiTe)*q/(mi))
    lref = (Eps0*Te*uref/(S0*L*q))**(1/3)
    nref = S0*L*lref/uref
    ui = 1
    un = 0

    return nn*(ui-un)*signn(TiTe*Te,lref,nref) 

meanfp = []

#build an array of mean free paths for various ion temperatures
#remember that the ion temp is related to the electron temp through the Ti/Te ratio
#With TiTeMax, the ion temp equals the electron temp
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
    #Find the ion flux at the wall for different neutral densities, source strengths, and electron temperatures
    sol = optimize.root(SheathEntrance, [L-5], args = (TeVal1,TiTeVal1,S0Val1,i))
    xSE = sol.x
    phiedge1, niedge1, needge1, uiedge1, ueedge1,_ = pred(L,TeVal1,TiTeVal1,S0Val4,i)
    testFlux1.append(uiedge1*niedge1)
    phiedge2, niedge2, needge2, uiedge2, ueedge2,_ = pred(L,TeVal2,TiTeVal1,S0Val4,i)
    testFlux2.append(uiedge2*niedge2)
    phiedge3, niedge3, needge3, uiedge3, ueedge3,_ = pred(L,TeVal1,TiTeVal1,S0Val1,i)
    testFlux3.append(uiedge3*niedge3)
    phiedge4, niedge4, needge4, uiedge4, ueedge4,_ = pred(L,TeVal2,TiTeVal1,S0Val1,i)
    testFlux4.append(uiedge4*niedge4)

#Plot the flux as a function of neutral density
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
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal2,TiTeVal2,S0Val2,nnVal2))
xSE2 = sol.x
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal3,TiTeVal3,S0Val3,nnVal3))
xSE3 = sol.x
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal4,TiTeVal4,S0Val4,nnVal4))
xSE4 = sol.x
sol = optimize.root(SheathEntrance, [L-5], args = (TeVal5,TiTeVal5,S0Val5,nnVal5))
xSE5 = sol.x
print("xSE1, xSE2, xSE3, xSE4:", xSE1, xSE2, xSE3, xSE4)
    
# Compute values at sheath entrance
phiSE1, niSE1, neSE1, uiSE1, ueSE1,_ = pred(xSE1,TeVal1,TiTeVal1,S0Val1,nnVal1)
phiSE2, niSE2, neSE2, uiSE2, ueSE2,_ = pred(xSE2,TeVal2,TiTeVal2,S0Val2,nnVal2)
phiSE3, niSE3, neSE3, uiSE3, ueSE3,_ = pred(xSE3,TeVal3,TiTeVal3,S0Val3,nnVal3)
phiSE4, niSE4, neSE4, uiSE4, ueSE4,_ = pred(xSE4,TeVal4,TiTeVal4,S0Val4,nnVal4)
phiSE5, niSE5, neSE5, uiSE5, ueSE5,_ = pred(xSE5,TeVal5,TiTeVal5,S0Val5,nnVal5)
if sheathbool == True:
    counts = 0

    #build arrays for different values at the sheath edge
    for j in range(0,numTiTepts):  
        for h in range(0,numnnpts):
            #Find the sheath edge
            sol = optimize.root( SheathEntrance, [L-5], args = (TeMax,TiTeScan[j],S0Max,nnScan[h]))
            xSE = sol.x
            #Predict values at the sheath edge
            phiSE, niSE, neSE, uiSE, ueSE,_ = pred(xSE,TeMax,TiTeScan[j],S0Max,nnScan[h])   
            #Predict values at the center 
            phiCenter, niCenter, neCenter, uiCenter, ueCenter,_ = pred(0,TeMax,TiTeScan[j],S0Max,nnScan[h]) 

            phiSEvec[j,h] = phiSE
            uiSEvec[j,h] = uiSE
            neSEvec[j,h] = neSE / niCenter
            niSEvec[j,h] = niSE / niCenter
            xSEvec[j,h] = xSE
            Tevec[j,h] = TeMax
            TiTevec[j,h] = TiTeScan[j]
            S0vec[j,h] = S0Max
            nnvec[j,h] = nnScan[h]

            #visual tracking because this can take several minutes with large numbers of predictions
            #Currently, only makes 100 predictions, which takes less than a minute
            counts = counts +1
            if counts%10000 == 0:
                print("Percent done:", counts*100/(numTiTepts*numnnpts),'%')

ptsize = 100

#########################
#        Plots
#########################

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

ax2.set_xlabel("$x/\\lambda_{de}$")
ax2.set_title("Density")
ax2.legend()
fig2.savefig(Data_path + "Densities.png")

fig3, ax3 = plt.subplots(num=3,nrows=1,ncols=1, clear=True)
fig3.set_tight_layout(True)

ax3.plot((xpts), np.ones(len(xpts)), label='$C_s$', linestyle=':',color='black',linewidth=3)

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

ax3.set_xlabel("$x/\\lambda_{De}$")
ax3.set_title("Velocity [$C_s$]")
ax3.set_ylim([0,4])
ax3.legend()
fig3.savefig(Data_path + "Velocities.png")

fluchs = xpts/L

fig4, ax4 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig4.set_tight_layout(True)

ue = ueList[0]
ui = uiList[0]
ne = neList[0] 
ni = niList[0]

ax4.plot((xpts), ue*ne, label='$S_0=10^{28}$, $n_n=0$', linestyle='-',color='r',linewidth=3)

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

#ax4.set_ylabel("$\\phi$")
ax4.set_xlabel("$x/\\lambda_{de}$")
ax4.set_title("Flux")
#ax4.set_yscale("log")
#ax4.legend()
#ax4.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5))
#fig.subplots_adjust(left=0.15)
fig4.savefig(Data_path + "Flux")#,bbox_inches='tight')

def Sauces(S,Te):
    #Computes the source, the ionization rate, and the recombination rate
    uref = (Te*q/mi)**(1/2)
    lref = (Eps0*Te*uref/(S*L*q))**(1/3)
    nref = S*L*lref/uref
    barS0 = S*lref**4*L/uref
    barSion = Sion2(Te,15.8)*(lref*nref/uref)
    barSrec = Srec2(Te,15.8,1)/(lref**2*uref)

    return barS0, barSion, barSrec

#Call the function to compute the source and sink terms for the smallest and largest source cases
barS01, barSion1, barSrec1 = Sauces(S0Min,TeMin)
barS02, barSion2, barSrec2 = Sauces(S0Max,TeMax)

fig5, ax5 = plt.subplots(num=4,nrows=2,ncols=3, clear=True)
fig5.set_tight_layout(True)

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

if sheathbool == True:
    #These plots are for the various derived quantites at the sheath entrance
    fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
    fig7.set_tight_layout(True)

    ax7.plot(nnvec[0,:], niSEvec[0,:], 'r', label='Te=0eV',linewidth=3)
    ax7.plot(nnvec[0,:], niSEvec[-1,:], 'b', label='Te=20eV',linewidth=3)
    ax7.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax7.set_title("Density Ratio")
    ax7.set_xscale("log")
    ax7.legend()
    fig7.savefig(Data_path + "Density_Ratio.png")

    fig8, ax8 = plt.subplots(num=8,nrows=1,ncols=1, clear=True)
    fig8.set_tight_layout(True)

    ax8.plot(nnvec[0,:], phiSEvec[0,:], 'r', label='Te=0eV',linewidth=3)
    ax8.plot(nnvec[0,:], phiSEvec[-1,:], 'b', label='Te=20eV',linewidth=3)
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
    #ax7.set_ylabel("$\\phi$")
    ax9.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax9.set_title("Bohm Criterion")
    ax9.set_xscale("log")
    #ax9.legend()
    fig9.savefig(Data_path + "Bohm_Criterion.png")
    
    fig11, ax11 = plt.subplots(num=9,nrows=1,ncols=1, clear=True)
    fig11.set_tight_layout(True)

    ax11.plot(nnvec[0,:], L-xSEvec[0,:], 'r', label='Te=0eV, Ti=0',linewidth=3)
    ax11.plot(nnvec[0,:], L-xSEvec[-1,:], 'b', label='Te=0eV, Ti=Te',linewidth=3)
    
    #ax9.set_ylabel("$\\phi$")
    ax11.set_xlabel("$\\lambda_{De}/\\lambda_{in}$")
    ax11.set_ylabel("# $\\lambda_{De}$")
    ax11.set_title("Sheath Width")
    ax11.set_xscale('log')
    #ax11.legend()
    fig11.savefig(Data_path + "Sheath_Width")

def fixloss(l):
    #Adjust the test loss so that it only plots at the iterations it was evaluated at
    teststeps = [0]
    loss2 = [l[0]]
    for i in range(len(l)-1):
        if l[i+1] != l[i]:
            loss2.append(l[i+1])
            teststeps.append(steps[i+1])
    return np.array(teststeps),loss2

#max_index = loss[:,0].argmax()
#print("Max index:", max_index)
train_pos = loss[:,0]
train_EC = loss[:,1]
train_IM = loss[:,2]

#initial steps can sometimes jump really high as the optimizer feels around for the optimal direction to pursue
#This loop removes residual points that are exceptionally high so they don't swallow the graph when matplotlib tries to scale the graph
#If this fails for any reason, we just pretend we didn't try to delete a point and move on to the rest of the script without throwing an error
while max(train_pos) > 1e8 or max(train_EC) > 1e8 or max(train_IM) > 1e8:
    max_index = train_EC.argmax()
    try:
        train_pos = np.delete(train_pos, max_index)
        train_EC = np.delete(train_EC, max_index)
        train_IM = np.delete(train_IM, max_index)
        steps = np.delete(steps, max_index)
    except:
        break

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

#ax16.set_ylabel("$\\phi$")
ax16.set_xlabel("Epochs (x1000)")
ax16.set_title("Losses")
#ax16.set_ylim([1e-7,5e-4])
ax16.set_yscale("log")
ax16.legend()
fig16.savefig(Data_path + "Losses.png")

#Create arrays for the input parameters
Tepts = np.linspace(TeMin,TeMax,numxpts)
TiTepts = np.linspace(TiTeMin,TiTeMax,numxpts)
S0pts = np.linspace(S0Min,S0Max,numxpts)
Knudpts = np.linspace(KnudMin,KnudMax,numxpts)
nnpts = np.linspace(nnMin,nnMax,numxpts)

#Create a set of test points to evaluate the residuals at
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
nnpts = X_test[:,4]

poisson_res = np.abs(Y[0]) 
ec_res = np.abs(Y[1]) 
im_res = np.abs(Y[2]) 

#Residual Plots
fig17,ax17 = plt.subplots(num=17,nrows=1,ncols=1, clear=True)
fig17.set_tight_layout(True)

ax17.scatter(X, ec_res, color='r',label="Electron Continuity",s=0.01)
ax17.scatter(X, im_res, color='b',label="Ion Momentum",s=0.01)
ax17.scatter(X, poisson_res, color='k',label="Poisson",s=0.01)

ax17.set_xlabel("$x/\\lambda_{de}$")
ax17.set_ylabel("$T_e$")
ax17.set_yscale("log")
ax17.set_title("Poisson")
ax17.legend()
fig17.savefig(Data_path + "Residuals.png")
plt.close()

def plot(x,y,sizes,labels,titles):
    #There are a lot of combinations of inputs to represent with 2D graphs, so this function helps streamline the plotting process
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
plot(X,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["X","nn"],["Poisson","Electron Continuity","Ion Momentum"])
plot(Tepts,TiTepts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Te","TiTe"],["Poisson","Electron Continuity","Ion Momentum"])
plot(Tepts,S0pts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Te","S0"],["Poisson","Electron Continuity","Ion Momentum"])
plot(Tepts,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["Te","nn"],["Poisson","Electron Continuity","Ion Momentum"])
plot(TiTepts,S0pts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["TiTe","S0"],["Poisson","Electron Continuity","Ion Momentum"])
plot(TiTepts,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["TiTe","nn"],["Poisson","Electron Continuity","Ion Momentum"])
plot(S0pts,nnpts,[0.01*poisson_res,0.01*ec_res,0.01*im_res],["S0","nn"],["Poisson","Electron Continuity","Ion Momentum"])


phi2list = []
ni2list = []
ne2list = []
ui2list = []
ue2list = []

phi,ni,ne,ui,ue,_ = pred(xpts,TeMax,0,S0Max,nnMax)
phi2list.append(phi)
ne2list.append(ne)
ni2list.append(ni)
ue2list.append(ue)
ui2list.append(ui)

#704 to 730 describes a verification that the torch version and the numpy version of the ionization and recombination rates calculate the same value
#choose a few x points inside domain
x_check = np.linspace(0.01, L-1e-3, 10)

#get PINN outputs (phi, ni, ne, ui, ue, newall)
phi_t, ni_t, ne_t, ui_t, ue_t, newall_t = pred(x_check, TeMax, 0, S0Max, nnMax)

#compute barSion with numpy (same formula as RK)
uref = np.sqrt((TeMax*q/mi))
lref = (Eps0*TeMax*uref/(S0Max*L*q))**(1/3)
nref = S0Max*L*lref/uref
barSion_np = Sion2(TeMax, 15.8) * (lref * nref / uref)

#compute the term used in PINN: nn * ne * barSion  (vector)
term_pinn = nnMax * ne_t * barSion_np

#compute same term using torch implementation (callable analog)
Te_torch = torch.tensor(TeMax, dtype=torch.float64, device=device)
uref_t = torch.sqrt((Te_torch)*q/mi)
lref_t = (Eps0*Te_torch*uref_t/(S0Max*L*q))**(1/3)
nref_t = S0Max*L*lref_t/uref_t
barSion_t = Sion(Te_torch, torch.tensor(15.8, dtype=Te_torch.dtype, device=device)) * (lref_t * nref_t / uref_t)

#compare numeric values
print("barSion_np:", barSion_np)
print("barSion_t:", barSion_t.item())
print("max abs diff on term nn*ne*barSion:", np.max(np.abs(term_pinn - (nnMax * ne_t * barSion_t.item()))))

x0 = 0.01
S = 1/L

def runge(Te,S0,nn):
    #Building the RK model
    TiTe=0
    un = 0
    E0 = 0
    #Predict initial values for the starting point of the RK solver using the PINN, rather than using backwards differencing to estimate the values
    #This ensures the RK solver starts at the same place as the PINN, which allows us to verify that the PINN solved the equations correctly
    phi,ni,ne,ui,ue,newall = pred(x0,Te,TiTe,S0,nn)
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

    def system(x, y):
        #Define the governing ODEs
        phi = y[0]
        E = y[1]
        ui = y[2]
        ue = y[3]
        ne = nw * np.exp(phi)

        Sauce = S + nn*ne*barSion

        dphi_dx = -E 
        dE_dx = ne*(ue/ui-1)
        dui_dx = E/ui - ui/(ne*ue*L) - nn*ui*barSion/(ue) - nn*signn(0.5*(Te),lref,nref)
        due_dx = 1/(ne*L)  + ue*E + nn*barSion
        return np.array([dphi_dx, dE_dx, dui_dx, due_dx])

    xlist = [x0] 
    philist = [phi0]
    Elist = [E0]
    uilist = [ui0]
    uelist = [ue0]
    nilist = [ni0]

    #Use scipy to solve the system
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
    uelist = sol.y[3].tolist()
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

#Predict the profiles as calculated by the RK method
xl, phi, ui, E, ue, ne, ni = runge(TeMax,S0Max,nnMax)
phiRKlist.append(phi)
neRKlist.append(ne)
xRKlist.append(xl)
ueRKlist.append(ue)
uiRKlist.append(ui)
niRKlist.append(ni)


#print(len(xRKlist[0]))
ptsize = 100
plotonly = 6001
plotevery = 500

#############################
# RK Plots comparing to PINN
#############################

fig20, ax20 = plt.subplots(num=20,nrows=1,ncols=1, clear=True)
fig20.set_tight_layout(True)
x = xRKlist[0]
phi = phi2list[0]
phiRK = phiRKlist[0]
ax20.scatter(x[-plotonly:][::plotevery],phiRK[-plotonly:][::plotevery],color='k',label="RK45",marker='x',s=ptsize)
ax20.plot(xpts,phi,color='k',linestyle='-',label='PINN',linewidth=2)
ax20.legend()
ax20.set_title("$e\\phi /T$")
ax20.set_xlabel("$\\lambda_{de}$")
fig20.savefig(Data_path + "phiRK.png")
plt.close()

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