import torch
import torch.nn as nn
import tqdm
import numpy as np
from scipy.optimize import minimize

class FNN(nn.Module):
    def __init__(self, in_dim, layers, out_dim):
        super().__init__()
        net = []
        dims = [in_dim] + layers + [out_dim]
        for i in range(len(dims)-2):
            net.append(nn.Linear(dims[i], dims[i+1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*net)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

def TrainSOAPorADAM(NumEpochs,optimizer,model,loss_fn,lr,X_train,X_test,test_every,save_every,SavePath):
    
    t = tqdm.trange(NumEpochs)

    dummy = loss_fn(model, X_train)
    if not isinstance(dummy, (tuple, list)):
        dummy = [dummy]
    num_out = len(dummy)
    
    loss_hist = [[] for _ in range(num_out)]
    loss_test = [[] for _ in range(num_out)]
    
    for epoch in t:
        
        # Adaptive sampling
#         if resample_train_points and (epoch % resample_every == 0) and epoch > 0:
#             X_train = adaptive_resample(
#             model, pde, N_target=N,
#             k=k, c=c, frac=frac, pool_mult=pool_mult,
#             chunk_size=65536, device=device
#         )
        
        optimizer.zero_grad()     
        residuals = loss_fn(model, X_train)
        if not isinstance(residuals, (tuple, list)):
            residuals = [residuals]

        losses = [torch.mean(res**2) for res in residuals]
        total_losses = sum(losses)

        for i, l in enumerate(losses):
            loss_hist[i].append(l.detach())
        t.set_postfix({f"loss{i+1}": f"{l.detach():.2e}" for i, l in enumerate(losses)})
        t.refresh()

        total_losses.backward()
        optimizer.step()
        if lr is not None:
            lr.step()

        if (test_every is not None) and (epoch % test_every == 0):
            test_residuals = loss_fn(model, X_test)
            if not isinstance(test_residuals, (tuple, list)):
                test_residuals = [test_residuals]
            test_losses = [torch.mean(res**2) for res in test_residuals]

            for i, l in enumerate(test_losses):
                loss_test[i].append(l.detach())
        
        # Model saving
        if (save_every is not None) and (epoch % save_every == 0):
            ckp = model.state_dict()
            PATH = SavePath + f"model{epoch}.pt"
            torch.save(ckp, PATH)

    total_loss = [torch.stack(hist).cpu().numpy() for hist in loss_hist]
    if test_every is not None:
        test_loss = [torch.stack(hist).cpu().numpy() for hist in loss_test]
    else:
        test_loss = 0

    return total_loss, test_loss

def TrainScipy(model, loss_fn, method, X_train, X_test, test_every, epochs, maxiter, save_every, SavePath):

    dummy = loss_fn(model, X_train)
    if not isinstance(dummy, (tuple, list)):
        dummy = [dummy]
    num_out = len(dummy)

    loss_hist = [[] for _ in range(num_out)]
    loss_test = [[] for _ in range(num_out)]
    pbar = tqdm.trange(maxiter)

    def get_flat_params(model): 
        params = []
        for param in model.parameters():
            params.append(param.detach().cpu().numpy().reshape(-1))
        return np.concatenate(params)

    def set_flat_params(model, flat_params):
        idx = 0
        for param in model.parameters():
            numel = param.numel()
            param_np = flat_params[idx:idx+numel].reshape(param.shape)
            param.data.copy_(torch.tensor(param_np, dtype=param.dtype, device=param.device))
            idx += numel

    def loss_and_grad(flat_params, model, loss_fn, X_train):
        set_flat_params(model, flat_params)
        model.zero_grad()
        residuals = loss_fn(model, X_train)
        if not isinstance(residuals, (tuple, list)):
            residuals = [residuals]
        losses = [torch.mean(res**2) for res in residuals]
        loss = sum(losses)
        loss.backward()
        # ensure CUDA ops complete before reading param.grad
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        grads = []

        for i, l in enumerate(losses):
            loss_hist[i].append(l.detach())
        pbar.set_postfix({f"loss{i+1}": f"{l.detach():.2e}" for i, l in enumerate(losses)})

        pbar.update(1)
        if (test_every is not None) and (len(loss_hist[0]) % test_every == 0):
            test_residuals = loss_fn(model, X_test)
            if not isinstance(test_residuals, (tuple, list)):
                test_residuals = [test_residuals]
            test_losses = [torch.mean(res**2) for res in test_residuals]

            for i, l in enumerate(test_losses):
                loss_test[i].append(l.detach())

        if (save_every is not None) and (len(loss_hist[0]) % save_every == 0):
            ckp = model.state_dict()
            path = SavePath + f'model{len(loss_hist[0])+epochs}.pt'
            torch.save(ckp, path)

        for param in model.parameters():
            grads.append(param.grad.detach().cpu().numpy().reshape(-1))
        flat_grad = np.concatenate(grads)
        return loss.item(), flat_grad


    flat_params_init = get_flat_params(model)
    #nfeval = epochs
    
    """def callback(params):
        print("Callback called")
        set_flat_params(model, params)
        residuals = loss_fn(model, X_train)
        if not isinstance(residuals, (tuple, list)):
            residuals = [residuals]

        losses = [torch.mean(res**2) for res in residuals]
        for i, l in enumerate(losses):
            loss_hist[i].append(l.detach())
        pbar.set_postfix({f"loss{i+1}": f"{l.detach():.2e}" for i, l in enumerate(losses)})
        #pbar.refresh()
        pbar.update(1)
        if len(loss_hist[0]) % 500 == 0:
            ckp = model.state_dict()
            path = SavePath + f'model{len(loss_hist[0])+epochs}.pt'
            torch.save(ckp, path) """       

    if method=='l-bfgs-b':
        scipy_options={'maxiter': maxiter,
                       'maxfun': maxiter,
                       'maxls': 200,
                       'maxcor':100,
                       'gtol': np.finfo(np.float64).eps,
                       'ftol': np.finfo(np.float64).eps}
    if method=='BFGS':
        scipy_options={'maxiter': maxiter,
                       'gtol': np.finfo(np.float64).eps,
                       'method_bfgs': 'SSBroyden2'}

    res = minimize(
        loss_and_grad,
        flat_params_init,
        args=(model, loss_fn, X_train),
        method=method,
        jac=True,
        #callback=callback,
        options=scipy_options,        
        )
    set_flat_params(model, res.x)

    total_loss = [torch.stack(hist).cpu().numpy() for hist in loss_hist]
    if test_every is not None:
        test_loss = [torch.stack(hist).cpu().numpy() for hist in loss_test]
    else:
        test_loss = 0
    return total_loss, test_loss
