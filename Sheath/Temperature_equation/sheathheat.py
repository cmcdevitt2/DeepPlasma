import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch

import time
import sys

save_path = '/path/to/model/'

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
nr = S0*L*xr/ur                     #m^-3 
                                    

Epsnormsq = Eps0**2*Ts**2/(q**2*xr*nr)

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

def log10(y):
    return torch.log10(y)

def egyrofreq(B):
    #B in Tesla
    #NRL formulary fit for electron gyrofrequency in rad/s
    return 1.76e3*B

def coulog(n,T):
    #n in m^-3 and T in eV
    #Coulomb logarithm 
    n_dim = n*nr/1e6 #convert to cm^-3
    T_dim = T*Ts
    return 23.4 - 1.15*torch.log10(n_dim) + 3.45*torch.log10(T_dim)

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

def boundaryRight(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

def boundaryLeft(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


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
    dTe_xx = dde.grad.hessian(Te,inputs, i=0, j=0) / (xMax-xMin)**2
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
    Rui = 1/MiMe*(ue-ui)*ne/Taue(ni,Te) *0.51
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

def main():
    geom = dde.geometry.Hypercube([0,0], [1,1])

    #geom = dde.geometry.Hypercube([0, 0], [1, 1])
    #extreme_region = dde.geometry.Hypercube([0.7, 0.5, 0.9, 0.5,0.5], [1, 1, 1,1,1])
    #mid_region = dde.geometry.Hypercube([0,0.5],[1,0.75])
    #left_corner = dde.geometry.Hypercube([0, 0.75], [0.25, 1])
    #right_corner = dde.geometry.Hypercube([0.75, 0.75], [1, 1])
    #bl_corner = dde.geometry.Hypercube([0,0],[0.25,0.25])
    #br_corner = dde.geometry.Hypercube([0.75,0],[1,0.25])
    #tight_center = dde.geometry.Hypercube([0.4,0.4],[0.6,0.6])
    #broad_center = dde.geometry.Hypercube([0.2,0.2],[0.8,0.8])
    sheath_region = dde.geometry.Hypercube([0.8,0],[1,1])
    sheath_points = sheath_region.random_points(10000)

    #uniform_points = geom.random_points(500000)
    #extreme_points = extreme_region.random_points(500000)
    #mid_points = mid_region.random_points(interiorpts[2])
    #left_corner_points = left_corner.random_points(interiorpts[3])
    #right_corner_points = right_corner.random_points(interiorpts[4])
    #bl_corner_points = bl_corner.random_points(interiorpts[4])
    #br_corner_points = br_corner.random_points(interiorpts[5])
    #tight_center_points = tight_center.random_points(interiorpts[6])
    #broad_center_points = broad_center.random_points(interiorpts[7])

    #points = np.append(uniform_points, extreme_points, axis = 0)
    #points = np.append(points,extreme_points,axis =0)
    #points = np.append(points, left_corner_points, axis = 0)
    #points = np.append(points, right_corner_points, axis = 0)

    n = 5
    activision = f"LAAF-{n} tanh"
    net = dde.maps.FNN([1]+[32]*3+[5], "tanh", "Glorot normal")
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    losses = []

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
        [bc_wall],
        num_domain=pts,
        num_boundary=bdypts,
        num_test=pts,
        train_distribution='Hammersley',
        #test_distribution='uniform',
        #anchors=sheath_points 
    )

    model = dde.Model(data, net)

    loss_weights = [10,1,1,1,1] 

    PDE_Resampler = dde.callbacks.PDEPointResampler(period=500,pde_points=True)

    loss = ["MSE"] * 5
    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights,external_trainable_variables=Tewall)
    variable = dde.callbacks.VariableValue(Tewall, period=1000)
    losshistory, train_state = model.train(epochs=0,callbacks=[variable,PDE_Resampler], model_save_path = save_path + 'model.pt')

    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights,external_trainable_variables=Tewall)
    losshistory, train_state = model.train(epochs=epochsADAM,callbacks=[variable,PDE_Resampler], model_save_path = save_path + 'model.pt')

    save_solution(geom, model, "solution0")
    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir= save_path + 'data')

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

        #callbacks=[PDE_Resampler],
        losshistory, train_state = model.train(display_every=1000,callbacks=[variable,PDE_Resampler],model_save_path = save_path + 'model.pt')

        
        save_solution(geom, model, "solution0")

        dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir= save_path + 'data')

        """Xtrain = np.loadtxt(f'./models/3000_latest/data/train.dat')
        Restrain = model.predict(Xtrain,operator=pde)
        Restrain1 = Restrain[0]
        Restrain2 = Restrain[1]
        Restrain3 = Restrain[2]
        Restrain4 = Restrain[3]
        print(f'MSE_train = {np.mean(Restrain1**2):.2e}, MSE_train = {np.mean(Restrain2**2):.2e}, MSE_train = {np.mean(Restrain3**2):.2e}, MSE_train = {np.mean(Restrain4**2):.2e}')

    
        Xtest = np.loadtxt(f'./models/3000_latest/data/test.dat')[:,:2]
        Restest = model.predict(Xtest,operator=pde)
        Restest1 = Restest[0]
        Restest2 = Restest[1]
        Restest3 = Restest[2]
        Restest4 = Restest[3]
        print(f'MSE_test = {np.mean(Restest1**2):.2e}, MSE_test = {np.mean(Restest2**2):.2e}, MSE_test = {np.mean(Restest3**2):.2e}, MSE_test = {np.mean(Restest4**2):.2e}')
        '''print(f'MSE_train = {np.mean(Restrain**2):.2e}, MSE_test = {np.mean(Restest**2):.2e}')'''
        """
        Tew = variable.get_value()
        print(Tew)
if __name__ == "__main__":
    main()
