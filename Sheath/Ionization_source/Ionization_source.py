import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_optimizer.optimizer.soap import SOAP
from torch.quasirandom import SobolEngine
from NN_modules import *

save_path = '/blue/cmcdevitt/ewebb2/PyTorch/models/test2/'

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)

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
TiTeMin = 0
TiTeMax = 1
KnudMin = 0 # collisionality parameters for ion-neutral collisions
KnudMax = 1.5e-1
nnMin = 0.0
nnMax = 5.0
MiMe = 1836*40
niMax = 25

#reduced dimension study
#S0 = 1e32
#Te = 20#20
#TiTe = 0.0
#nn = 4.0 #4.0

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

#for hydrogen, 
#first ionization potential
#Einf = 13.6 #eV

#ion mass
mi = MiMe*me#1.6735575e-27 #kg

#T and Ez in eV
def Sion(T,Ez): 
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * torch.exp(-Ez/T) 

def Srec(T,Ez,Z):
    return 5.2e-20*Z*(Ez/T)**(1/2)*(0.43 + 0.5*torch.log(Ez/T) + 0.469*(Ez/T)**(-1/3))

def Sion2(T,Ez): 
    return 1e-11 * ( (T/Ez)**(1/2) ) / ( (Ez)**(3/2)*(6.0+T/Ez) ) * np.exp(-Ez/T) 

def Srec2(T,Ez,Z):
    return 5.2e-20*Z*(Ez/T)**(1/2)*(0.43 + 0.5*np.log(Ez/T) + 0.469*(Ez/T)**(-1/3))

def signn(E,xr,nr):
    E = E + 1e-4
    return (2e-19/((2*E)**0.5*(1+2*E)) + 3e-19*2*E/(1+2/3*E)**2.3) * xr*nr

#dummy vars
#nn = 1

def feature_transform(inputs):
    #xNorm, TeNorm, TiTeNorm, S0Norm, nnNorm = inputs.split(1,dim=1)
    #xNorm,TeNorm,TiTeNorm, KnudNorm = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4]
    xNorm,TeNorm,TiTeNorm,S0Norm,nnNorm = inputs.split(1,dim=1)
    #xNorm = inputs
    xSq = xNorm * xNorm

    return torch.cat((xSq,TeNorm,TiTeNorm,S0Norm,nnNorm),dim=1)
    #return xSq

def output_transform(inputs, outputs):
    #xNorm, TeNorm, TiTeNorm = inputs.split(1,dim=1)
    xNorm,TeNorm,TiTeNorm = inputs[:,0:1],inputs[:,1:2],inputs[:,2:3]
    #xNorm = inputs
    phi, ni, ue, nw = outputs.split(1,dim=1)
    
    x = xMin + ( xMax - xMin ) * xNorm
    Te = TeMin + (TeMax - TeMin ) * TeNorm
    TiTe = TiTeMin + (TiTeMax - TiTeMin ) *TiTeNorm
    #TiTe = TiTeMin

    #uewall = np.sqrt(MiMe*8*Te/(np.pi*(Te+TiTe*Te)))
    #uewall = torch.sqrt(MiMe*Te/(2*np.pi*(Te+TiTe*Te)))
    uewall = np.sqrt(MiMe/(2*np.pi)) #correct
    #uewall = torch.sqrt(MiMe*(Te+TiTe*Te)/2*np.pi)
    #enforce a floor on density over the domain
    #or multiply continuity by L

    phiFinal = (L-x)*(L+x)/L**2 * phi 
    #niFinal = (niMax)*0.5*( 1 + torch.tanh(ni) ) #+ niMin*(1-(x/L)**2) 
    #niFinal = torch.log(1 + torch.exp(ni))
    niFinal = F.softplus(ni) + 1e-8
    #beta = 20.0
    #ueFinal = (L-x)*(L+x)/L**2*(x/L) * (ue - F.softplus(beta*(ue-2))/beta) + ((1-torch.tanh(5-x/(0.2*L)))+(torch.tanh(5+x/(0.2*L))-1))*np.log(uewall+1)
    ueFinal = (L-x)*(L+x)/L**2*(x/L) * ue + ((1-torch.tanh(5-x/(0.2*L)))+(torch.tanh(5+x/(0.2*L))-1))*np.log(uewall+1)
    #ueFinal = (x/L) * ue 
    #nwFinal = torch.log(1+torch.exp(nw))
    #nwFinal = nw 
    nwFinal = F.softplus(nw) 

    #niFinal = torch.clamp(niFinal, max=200)

    return torch.cat((phiFinal,niFinal,ueFinal,nwFinal), dim=1)

def pde(model,inputs):
    inputs = inputs.requires_grad_(True)
    inputs_trans = feature_transform(inputs)
    xNorm, TeNorm, TiTeNorm, S0Norm, nnNorm = inputs.split(1,dim=1)
    #xNorm,TeNorm,TiTeNorm,S0Norm,nnNorm= inputs[:,0:1], inputs[:,1:2], inputs[:,2:3],inputs[:,3:4],inputs[:,4:5]
    #xNorm = inputs

    #Xne = torch.ones(pts,1,device=device).requires_grad_(True)
    X = torch.cat((xNorm*0 +1.0, TeNorm, TiTeNorm, S0Norm, nnNorm),dim=1)
    #X = torch.cat((Xne,TeNorm,TiTeNorm, KnudNorm),dim=1)
    #X = xNorm*0 +1.0
    
    #print(X)
    X2 = feature_transform(X)

    x = xMin + ( xMax - xMin ) * xNorm
    #Te = TeMin
    #TiTe = TiTeMin
    #Knud = 0.2
    #S0 = S0Min
    Te = TeMin + (TeMax - TeMin ) * TeNorm
    TiTe = TiTeMin + (TiTeMax - TiTeMin ) * TiTeNorm
    #Knud = KnudMin + (KnudMax - KnudMin) * KnudNorm
    S0 = S0Min + (S0Max - S0Min ) * S0Norm
    nn = nnMin + (nnMax - nnMin) * nnNorm
    un = 0.0
    Ti = Te * TiTe

    #print(xNorm)

    outputs = model(inputs_trans)
    outputs_trans = output_transform(inputs, outputs)
    phi, ni, uet,_ = outputs_trans.split(1,dim=1)

    outputs2 = model(X2)
    outputs2_trans = output_transform(X,outputs2)
    _,_,_,nw = outputs2_trans.split(1,dim=1)
    #print(nwt)
    #exit()

    #coordinate transforms
    ne = nw*torch.exp(phi)
    #ni = ni.clamp_min(1e-6)
    #s = nwt + phi
    #s = torch.clamp(s, max=700.0)   # 700 is safe for float64; use ~88 for float32
    #ne = torch.exp(s)
    #ne = torch.exp(nwt+phi)# * torch.exp(phi)
    #uet_clamped = torch.clamp(uet, max=700.0)
    ue = torch.expm1(uet)   # = exp(uet)-1, more stable for small uet
    #ue = uet

    ui = ne*ue/ni
    #denom = ni.clamp_min(1e-8)
    #ui = ne * ue / denom

    #normalizations
    #uref = torch.sqrt((Te+Te*TiTe)*q/(mi))
    uref = torch.sqrt((Te)*q/(mi))
    #uref = np.sqrt((Te+Te*TiTe)*q/(1836*me)) #this is wrong, but for the sake of comparison
    #lref = torch.pow(Eps0*Te*uref/(S0*L*q), 1/3)
    lref = (Eps0*Te*uref/(S0*L*q))**(1/3)
    nref = S0*L*lref/uref
    #barS0 = S0*lref**4*L/uref
    #H-13.6 | Ar-15.8
    #barSion = Sion2(Te,15.8)/(lref**2*uref)
    barSion = Sion(Te,15.8)*(lref*nref/uref)
    #barSrec = Srec2(Te,15.8,1)*(lref*nref/uref)

    dphi_x = torch.autograd.grad(phi, inputs, create_graph=True, grad_outputs=torch.ones_like(phi))[0][:,0:1] / (xMax - xMin)
    dphi_xx = torch.autograd.grad(dphi_x, inputs, create_graph=True, grad_outputs=torch.ones_like(dphi_x))[0][:,0:1] / (xMax - xMin)
    dne_x = torch.autograd.grad(ne, inputs, create_graph=True, grad_outputs=torch.ones_like(ne))[0][:,0:1] / (xMax - xMin)
    due_x = torch.autograd.grad(ue, inputs, create_graph=True, grad_outputs=torch.ones_like(ue))[0][:,0:1] / (xMax - xMin)
    dni_x = torch.autograd.grad(ni, inputs, create_graph=True, grad_outputs=torch.ones_like(ni))[0][:,0:1] / (xMax - xMin)
    dui_x = torch.autograd.grad(ui, inputs, create_graph=True, grad_outputs=torch.ones_like(ui))[0][:,0:1] / (xMax - xMin)

    lossb1 = dphi_xx - (ne - ni) # Poisson
    lossb2 = L*ne*due_x + L*ue*dne_x - 1 - L*nn*ne*barSion # electron continuity- nn*ne*barSion
    lossb5 = TiTe*dni_x + ni*dphi_x + ui*(1/L + nn*ne*barSion) + ni*ui*dui_x + nn*ni*1*ui*signn(0.5*(Te+TiTe*Te),lref,nref) #C_s/uref=1  #ni*ui*Knud #ni*ui*nn*Knud/(nnMax) #+ ni*ui*nn*0.3/2#+ ni*ui*Knud  #ion momentum
    
    #+ nn*ne*barSion*barS0 - ni*ne*barSrec*barS0
    #soft BCs
    #uewall = np.sqrt(MiMe*Te/(2*np.pi*(Te+TiTe*Te)))
    #BC0 = torch.ones(1,1,device=device).requires_grad_(True)
    #right_bound = torch.cat((BC0, TeNorm, TiTeNorm, S0Norm, KnudNorm, nnNorm),dim=1)
    #_,_,ue,_ = output_transform(right_bound, model(feature_transform(right_bound))).split(1,dim=1)

    #lossb3 = uewt - np.log(uewall)
    '''BC0 = torch.zeros(1,1,device=device)
    _,ni,_,_ = output_transform(BC0, model(feature_transform(BC0))).split(1,dim=1)

    dni_dx = torch.autograd.grad(ni, BC0, create_graph=True, grad_outputs=torch.ones_like(ni))[0][:,0:1] / (xMax - xMin)

    lossb3 = dni_dx'''

    return lossb1,lossb2,lossb5#,lossb3

model = FNN(in_dim=5, layers=[32]*5, out_dim=4).to(device)


def main():

    '''Xs = torch.tensor([[0.5]], dtype=torch.float64, device=device).requires_grad_(True)
    outs = model(feature_transform(Xs))
    outs_tr = output_transform(Xs, outs)
    phi, ni, uet, nw = outs_tr.split(1,1)
    print("phi finite:", torch.isfinite(phi).all(), "ni finite:", torch.isfinite(ni).all())'''
    engine = SobolEngine(dimension=5, scramble=True, seed=1234)
    engine2 = SobolEngine(dimension=4, scramble=True, seed=5678)
    X_bound = engine2.draw(bdypts).to(device)
    #X_bound = torch.cat((X_bound[:,0:1]*0.0+1.0, X_bound),dim=1)
    X_rbound = torch.cat((X_bound[:,0:1]*0.0+1.0, X_bound),dim=1)
    X_lbound = torch.cat((X_bound[:,0:1]*0.0, X_bound),dim=1)
    #X_corner = engine.draw(corner_points).to(device)
    #X_corner = torch.cat((0.8 + 0.2*X_corner[:,0:1], 0.8 + 0.2*X_corner[:,1:2], 0.2*X_corner[:,2:3], 0.8 + 0.2*X_corner[:,3:4], 0.8+0.2*X_corner[:,4:5]),dim=1)
    X_train = engine.draw(pts).to(device)
    X_train = torch.cat((X_train, X_rbound, X_lbound),dim=0)
    #X_train = torch.cat((X_train),dim=0)
    #X_sheath = 0.8 + 0.2*torch.rand(sheath_pts, 1).to(device)
    #X_sheath = engine.draw(sheath_pts).to(device)
    #X_sheath = torch.cat((0.8+0.2*X_sheath[:,0:1], X_sheath[:,1:]),dim=1)
    #X_train = torch.cat((X_train, X_sheath),dim=0)

    engine3 = SobolEngine(dimension=5, scramble=True, seed=4321)
    X_test = engine3.draw(100000).to(device)

    optimizer = SOAP(model.parameters(), lr=lr, betas=(.999, .999), weight_decay=0e-2, precondition_frequency=1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99975)

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

    torch.save(model.state_dict(), save_path + f"model_final.pt")
if __name__ == "__main__":
    main()
