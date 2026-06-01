"""Core for the AIS-ladder study. The MultiWell target, the QT mode-discovery
helper, and the coverage metrics are copied verbatim from HD_Product. The new
piece is `fused_loss`, a single-tensor loss for KL+X_mu+X_mix that is
`loss_compile`-friendly (one varying input, the rest captured) so the training
step can be torch.compile'd via zflows.loss.loss_compile."""
import math
import torch
from zflows.potential import Potential
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin

# Product multi-well potential in d = 2**k dimensions (verbatim from HD_Product):
#   U(x) = 1/2 sum_i x_i^2 + 12 sum_{i<k} exp(-x_i^2)
# First k coords are double wells (minima at +-sqrt(ln 24) ~ +-1.783, barrier
# ~9.9 at the origin); the rest are standard Gaussian. U factorizes -> a product
# measure with exactly 2**k modes.
WELL_CENTER = math.sqrt(math.log(24.0))  # ~1.7831


class MultiWell(Potential):
    def __init__(self, k: int):
        super().__init__()
        self.k = k  # d = 2**k

    def forward(self, x):  # x: [N, d] -> [N]
        quad = 0.5 * (x * x).sum(dim=-1)
        well = 12.0 * torch.exp(-(x[:, :self.k] ** 2)).sum(dim=-1)
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


# ---------------------------------------------------------------------------
# Fused single-tensor loss for KL + LAMBDA*X_mu + X_mix, loss_compile-friendly.
# ---------------------------------------------------------------------------
def fused_loss(y_packed: torch.Tensor, source: Potential, target: Potential,
               G: ComposedTransform, lam: float):
    """KL+X_mu+X_mix in one pass on a single packed batch.

        y_packed = cat([y_mu, y_mix], dim=0)   # [2B, d], the two halves equal-size

    The log-ratio z(y) = source(G(y)) - target(y) - log|det J_G(y)| is evaluated
    once on the whole [2B, d] batch; the first B rows feed forward KL + X_mu and
    the last B rows feed X_mix. The X terms pair each z_i with z_{i-1} via a
    cyclic roll(1) instead of a fresh random permutation: this is permutation-
    invariant-equivalent PROVIDED each half is independently shuffled before
    packing (train.py does this), and unlike torch.randperm it introduces no RNG
    op inside the compiled graph. Returns  KL + lam * X_mu + X_mix."""
    x, ladj = G.call_and_ladj(y_packed)
    z = source(x) - target(y_packed) - ladj           # [2B]
    B = z.shape[0] // 2
    z_mu = z[:B]
    z_mix = z[B:]
    kl = z_mu.mean()
    x_mu = (z_mu - z_mu.roll(1)).abs().mean()
    x_mix = (z_mix - z_mix.roll(1)).abs().mean()
    return kl + lam * x_mu + x_mix


# Quench and Temper (QT) for mode discovery; needs target.enable_grad()+enable_eval()
def quench_and_temper(x, target, sigma, opt_step, opt_iters, mc_step, mc_iters):
    x = x + sigma * torch.randn_like(x)                                   # melt
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)     # quench
    x = langevin(x, target, step=mc_step, iters=mc_iters)                 # temper
    return x


# coverage metric (Naeem et al. 2020)
def coverage(y: torch.Tensor, x: torch.Tensor, k: int = 5) -> float:
    dxx = torch.cdist(x, x)
    dxx.fill_diagonal_(float('inf'))
    nnd_k = dxx.topk(k, dim=1, largest=False).values[:, -1]
    dxy = torch.cdist(x, y)
    return (dxy < nnd_k.unsqueeze(1)).any(dim=1).float().mean().item()


# exact mode coverage via the sign pattern of the first kk coordinates.
def mode_coverage(y: torch.Tensor, kk: int, frac: float = 0.01):
    N = y.shape[0]
    nmodes = 2 ** kk
    signs = (y[:, :kk] > 0).long()
    powers = (2 ** torch.arange(kk, device=y.device))
    bucket = (signs * powers).sum(dim=-1)
    counts = torch.bincount(bucket, minlength=nmodes).float()
    threshold = max(1.0, frac * N / nmodes)
    modes_found = int((counts >= threshold).sum().item())
    p = counts / counts.sum()
    tv = 0.5 * (p - 1.0 / nmodes).abs().sum().item()
    return modes_found / nmodes, tv
