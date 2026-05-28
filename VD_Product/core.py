import math
import torch
from zflows.potential import Potential
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin

# Product multi-well potential in d = 2**k dimensions:
#   U(x) = 1/2 sum_i x_i^2 + 12 sum_{i<k} exp(-x_i^2)
# The first k coordinates are double wells (minima at +-sqrt(ln 24) ~ +-1.783,
# barrier ~9.9 at the origin); the rest are standard Gaussian. U factorizes,
# so the target is a product measure with exactly 2**k modes. The exp(-x^2)
# coefficient was raised 6 -> 8 -> 12 to deepen the barrier and harden the benchmark.
WELL_CENTER = math.sqrt(math.log(24.0))  # ~1.7831


class MultiWell(Potential):
    def __init__(self, k: int):
        super().__init__()
        self.k = k  # d = 2**k

    def forward(self, x):  # x: [N, d] -> [N]
        quad = 0.5 * (x * x).sum(dim=-1)
        well = 12.0 * torch.exp(-(x[:, :self.k] ** 2)).sum(dim=-1)  # deeper wells -> harder
        return quad + well


# forward KL
def loss_KL(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    x, ladj = G.call_and_ladj(y)  # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    return z.mean()


# X functional (pairwise variation of the log-ratio under the batch's measure)
def loss_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return (z - z[perm]).abs().mean()


# Quench and Temper (QT) for mode discovery; needs target.enable_grad()+enable_eval()
def quench_and_temper(x, target, sigma, opt_step, opt_iters, mc_step, mc_iters):
    x = x + sigma * torch.randn_like(x)                                   # melt
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)     # quench
    x = langevin(x, target, step=mc_step, iters=mc_iters)                 # temper
    return x


# coverage metric (Naeem et al. 2020): fraction of reference points x_i whose
# k-NN ball (within x) contains at least one candidate y_j
def coverage(y: torch.Tensor, x: torch.Tensor, k: int = 5) -> float:
    dxx = torch.cdist(x, x)
    dxx.fill_diagonal_(float('inf'))
    nnd_k = dxx.topk(k, dim=1, largest=False).values[:, -1]
    dxy = torch.cdist(x, y)
    return (dxy < nnd_k.unsqueeze(1)).any(dim=1).float().mean().item()


# exact mode coverage, enabled by separability: each sample belongs to one of the
# 2**kk well-combinations by the sign pattern of its first kk coordinates.
# Returns (modes_found / 2**kk, mode_balance) where mode_balance is the total-variation
# distance of the occupancy histogram to the uniform 2**-kk (0 = perfectly balanced).
def mode_coverage(y: torch.Tensor, kk: int, frac: float = 0.01):
    N = y.shape[0]
    nmodes = 2 ** kk
    signs = (y[:, :kk] > 0).long()                          # [N, kk] in {0,1}
    powers = (2 ** torch.arange(kk, device=y.device))       # [kk]
    bucket = (signs * powers).sum(dim=-1)                    # [N] in [0, 2**kk)
    counts = torch.bincount(bucket, minlength=nmodes).float()
    threshold = max(1.0, frac * N / nmodes)
    modes_found = int((counts >= threshold).sum().item())
    p = counts / counts.sum()
    tv = 0.5 * (p - 1.0 / nmodes).abs().sum().item()
    return modes_found / nmodes, tv
