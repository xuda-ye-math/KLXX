import torch
from zflows.potential import Potential
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin

from parameters import L, KAPPA, LAMBDA, H


# 2D phi^4 lattice action on an L x L periodic lattice, flattened to R^{L*L}
class Phi4(Potential):
    def __init__(self, kappa=KAPPA, lam=LAMBDA, h=H):
        super().__init__()
        self.kappa, self.lam, self.h = kappa, lam, h
    def forward(self, x):
        phi = x.view(-1, L, L)
        hop = phi * (torch.roll(phi, 1, dims=1) + torch.roll(phi, 1, dims=2))
        s = (-2.0 * self.kappa * hop + phi.square()
             + self.lam * (phi.square() - 1.0).square() + self.h * phi)
        return s.sum(dim=(1, 2))


def magnetization(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)            # m(phi) = (1/D) sum_x phi_x


# forward KL
def loss_KL(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    return z.mean()

# X functional
def loss_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return (z - z[perm]).abs().mean() # autograd graph of z

# Quench and Temper (QT) algorithm for mode discovery
#   requires target.enable_grad() and target.enable_eval() to be called beforehand
def quench_and_temper(x: torch.Tensor, target: Potential, sigma: float, opt_step, opt_iters, mc_step, mc_iters):
    x = x + sigma * torch.randn_like(x)                                     # diffusion (melt):     scatter samples across R^d
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)       # optimization (quench): drive each sample to a mode center of target
    x = langevin(x, target, step=mc_step, iters=mc_iters)                   # rejuvenation (temper): spread samples around each mode
    return x

# coverage metric (Naeem et al. 2020): fraction of reference points x_i whose k-NN
# ball (within x) contains at least one candidate y_j
def coverage(y: torch.Tensor, x: torch.Tensor, k: int = 5) -> float:
    dxx = torch.cdist(x, x)
    dxx.fill_diagonal_(float('inf'))
    nnd_k = dxx.topk(k, dim=1, largest=False).values[:, -1] # [P]: distance to k-th nearest neighbor in x
    dxy = torch.cdist(x, y)                                 # [P, N]
    return (dxy < nnd_k.unsqueeze(1)).any(dim=1).float().mean().item()
