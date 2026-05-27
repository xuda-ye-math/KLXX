import torch
from zflows.potential import Potential
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin

# Annulus potential: U(x) = 20 * (r^6 - 8 r^4 + 16 r^2 + 1)^(1/3), with r^2 = x1^2 + x2^2
class Annulus(Potential):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        r2 = x[:, 0].square() + x[:, 1].square()
        return 20 * (r2.pow(3) - 8 * r2.pow(2) + 16 * r2 + 1).pow(1.0 / 3.0)

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

# forward KL + X functional
def loss_KL_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform, lambda_: float = 1.0):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return z.mean() + lambda_ * (z - z[perm]).abs().mean()

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
