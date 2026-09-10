import torch
import math

def _phi_and_derphi(eval_fg, xk, pk, alpha: float):
    x = xk + alpha * pk
    f, g = eval_fg(x)
    return f, g, torch.dot(g, pk)

def _zoom(eval_fg, xk, pk, phi0, derphi0, c1, c2,
          alo, ahi, phi_alo, derphi_alo, maxiter=20):
    for _ in range(maxiter):
        aj = 0.5 * (alo + ahi)
        phi_aj, g_aj, derphi_aj = _phi_and_derphi(eval_fg, xk, pk, aj)
        if (phi_aj > phi0 + c1*aj*derphi0) or (phi_aj >= phi_alo):
            ahi = aj
        else:
            if abs(float(derphi_aj)) <= -c2 * float(derphi0):
                return aj, phi_aj, g_aj
            if float(derphi_aj) * (ahi - alo) >= 0:
                ahi = alo
            alo, phi_alo, derphi_alo = aj, phi_aj, derphi_aj
    phi_alo2, g_alo2, _ = _phi_and_derphi(eval_fg, xk, pk, alo)
    return alo, phi_alo2, g_alo2

def strong_wolfe_line_search(eval_fg, xk, pk, gfk, old_fval,
                              c1=1e-4, c2=0.9, amax=None,
                              maxiter=10, zoom_maxiter=20, alpha1=1.0):
    derphi0 = torch.dot(gfk, pk)
    if (not torch.isfinite(derphi0)) or float(derphi0) >= 0.0:
        pk      = -gfk
        derphi0 = -torch.dot(gfk, gfk)
    phi0   = old_fval
    alpha0 = 0.0
    alpha1 = float(alpha1)
    if amax is not None:
        alpha1 = min(alpha1, float(amax))
    phi_a0, derphi_a0 = phi0, derphi0
    phi_a1, g_a1, derphi_a1 = _phi_and_derphi(eval_fg, xk, pk, alpha1)
    for i in range(maxiter):
        if (phi_a1 > phi0 + c1*alpha1*derphi0) or (i > 0 and phi_a1 >= phi_a0):
            a_s, p_s, g_s = _zoom(eval_fg, xk, pk, phi0, derphi0,
                                   c1, c2, alpha0, alpha1, phi_a0, derphi_a0,
                                   maxiter=zoom_maxiter)
            return float(a_s), xk + a_s*pk, p_s, g_s
        if abs(float(derphi_a1)) <= -c2 * float(derphi0):
            return float(alpha1), xk + alpha1*pk, phi_a1, g_a1
        if float(derphi_a1) >= 0.0:
            a_s, p_s, g_s = _zoom(eval_fg, xk, pk, phi0, derphi0,
                                   c1, c2, alpha1, alpha0, phi_a1, derphi_a1,
                                   maxiter=zoom_maxiter)
            return float(a_s), xk + a_s*pk, p_s, g_s
        alpha0, phi_a0, derphi_a0 = alpha1, phi_a1, derphi_a1
        alpha1 *= 2.0
        if amax is not None:
            alpha1 = min(alpha1, float(amax))
        phi_a1, g_a1, derphi_a1 = _phi_and_derphi(eval_fg, xk, pk, alpha1)
    return float(alpha1), xk + alpha1*pk, phi_a1, g_a1

class SSBroyden2(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gtol=1e-10, xrtol=0.0,
                 line_search="strong_wolfe", c1=1e-4, c2=0.9, backtrack=0.5,
                 ls_max_steps=25, wolfe_maxiter=10, zoom_maxiter=20,
                 amax=None, initial_scale=False, eps=1e-30,
                 dtype=torch.float64, device=None):
        super().__init__(params, defaults={})
        self.lr, self.gtol, self.xrtol           = lr, gtol, xrtol
        self.line_search, self.c1, self.c2       = line_search, c1, c2
        self.backtrack                           = backtrack
        self.ls_max_steps, self.wolfe_maxiter    = ls_max_steps, wolfe_maxiter
        self.zoom_maxiter                        = zoom_maxiter
        self.amax, self.initial_scale, self.eps  = amax, initial_scale, eps
        self.dtype                               = dtype
        self._params  = list(self.param_groups[0]["params"])
        self.device   = device if device is not None else self._params[0].device
        self._numels  = [p.numel() for p in self._params]
        self._P       = int(sum(self._numels))
        self.state["H"] = torch.eye(self._P, device=self.device, dtype=self.dtype)
        self.state["x"] = self._gather_flat_params().detach().clone()
        self.state["k"] = 0
        # Warm-start cache: (loss, grad) at the CURRENT parameters, carried
        # forward from the end of the previous step() call. Full-batch/
        # deterministic closures make this an exact reuse, not an
        # approximation -- avoids one redundant forward+backward pass
        # (an entire extra closure() call) on every iteration, which is
        # what scipy's own BFGS loop does via its `gfk = gfkp1` carry-over.
        self._fg_cache = None

    def _gather_flat_params(self):
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    def _set_flat_params_(self, flat):
        with torch.no_grad():
            off = 0
            for p, n in zip(self._params, self._numels):
                p.copy_(flat[off:off+n].view_as(p)); off += n

    def _gather_flat_grad(self):
        return torch.cat([
            p.grad.detach().reshape(-1) if p.grad is not None
            else torch.zeros(n, device=self.device, dtype=self.dtype)
            for p, n in zip(self._params, self._numels)
        ])

    def _eval_loss_and_grad(self, closure):
        loss = closure()
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=self.device, dtype=self.dtype)
        return loss.detach(), self._gather_flat_grad()

    def step(self, closure):
        if self._fg_cache is not None:
            fk, gk = self._fg_cache
            self._fg_cache = None
        else:
            self.zero_grad(set_to_none=True)
            fk, gk = self._eval_loss_and_grad(closure)
        xk, Hk, N = self._gather_flat_params().detach(), self.state["H"], self._P
        if torch.linalg.vector_norm(gk).item() <= self.gtol:
            self._fg_cache = (fk, gk)
            return fk
        pk = -(Hk @ gk)

        def eval_fg_at(x_flat):
            self._set_flat_params_(x_flat)
            self.zero_grad(set_to_none=True)
            return self._eval_loss_and_grad(closure)

        alpha, xkp1, fkp1, gkp1 = strong_wolfe_line_search(
            eval_fg_at, xk, pk, gk, fk,
            c1=self.c1, c2=self.c2, amax=self.amax,
            maxiter=self.wolfe_maxiter, zoom_maxiter=self.zoom_maxiter,
            alpha1=float(self.lr)
        )
        self._set_flat_params_(xkp1)

        sk, yk = xkp1 - xk, gkp1 - gk
        ys = torch.dot(yk, sk).item()
        if abs(ys) < self.eps:
            self.state["x"] = xkp1.detach().clone(); self.state["k"] += 1
            self._fg_cache = (fkp1, gkp1)
            return fkp1

        rhok   = 1.0 / ys
        Hkyk   = Hk @ yk
        ykHkyk = torch.dot(yk, Hkyk).item()
        if abs(ykHkyk) < self.eps:
            self.state["x"] = xkp1.detach().clone(); self.state["k"] += 1
            self._fg_cache = (fkp1, gkp1)
            return fkp1

        hk    = ykHkyk * rhok
        bk    = -alpha * rhok * torch.dot(sk, gk).item()
        ak    = bk * hk - 1.0
        denom = (1.0 + ak) or self.eps
        rad   = max(abs(ak) / denom, 0.0)
        rhokm = min(1.0, hk * (1.0 - math.sqrt(rad)))

        thetakm = 0.0 if abs(ak) < self.eps else (rhokm - 1.0) / ak
        thetakp = (1.0 / rhokm) if abs(rhokm) > self.eps else (1.0 / self.eps)
        inner   = thetakp if abs(bk) < self.eps else (1.0 - bk) / bk
        thetak  = max(thetakm, min(thetakp, inner))

        rhokk    = min(1.0, (1.0 / bk) if abs(bk) > self.eps else (1.0 / self.eps))
        sigmak   = 1.0 + thetak * ak
        exp      = 1.0 / (1.0 - float(N))
        sigmaknm1= abs(sigmak)**exp if abs(sigmak) > 0 else 0.0

        tauk = (min(rhokk * sigmaknm1, sigmak) if thetak <= 0.0
                else rhokk * min(sigmaknm1, 1.0 / thetak))

        vk     = (sk * rhok) - (Hkyk / ykHkyk)
        denom3 = (1.0 + ak * thetak) or self.eps
        phik   = (1.0 - thetak) / denom3
        tauk   = tauk if abs(tauk) > self.eps else (self.eps if tauk >= 0 else -self.eps)

        H_term  = Hk - torch.outer(Hkyk, Hkyk)/ykHkyk + (phik*ykHkyk)*torch.outer(vk, vk)
        Hk_new  = (H_term / tauk) + (rhok * torch.outer(sk, sk))

        self.state["H"] = Hk_new
        self.state["x"] = xkp1.detach().clone()
        self.state["k"] += 1
        self._fg_cache = (fkp1, gkp1)
        return fkp1

        