"""Fourier-field Bayesian inverse problem with a sign-flip-symmetry-induced
multimodal posterior (idea_C).

An unknown field on the 2D torus is parameterized by its lowest Fourier modes
(|k|^2 <= 2 -> exactly d=9 real coefficients theta). The likelihood observes the
SQUARED partial field per mode-group on a spatial grid, which is invariant under
flipping the sign of any one group -> 2^G = 4 well-separated posterior modes that
the centered Gaussian smoothness prior (the source mu_0) does not cover.

This module is self-contained (the loss / QT / coverage helpers are copied
locally, matching the 2D_Benchmark folders). The only new code vs the templates
is FourierFieldPosterior + the basis / grid / mode helpers.
"""
import math
import itertools

import torch
from zflows.potential import Potential
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin


# ---------------------------------------------------------------------------
# Fourier basis / grid / mode helpers (the genuinely new code)
# ---------------------------------------------------------------------------

# d = 9 real Fourier coefficients, |k|^2 <= 2, ordered exactly as idea_C.md table:
#   m | k=(k1,k2) | parity | |k|^2 | group
#   0 | (0,0)  | cos | 0 | 0
#   1 | (0,1)  | cos | 1 | 0
#   2 | (0,1)  | sin | 1 | 0
#   3 | (1,-1) | cos | 2 | 0
#   4 | (1,-1) | sin | 2 | 0
#   5 | (1,0)  | cos | 1 | 1
#   6 | (1,0)  | sin | 1 | 1
#   7 | (1,1)  | cos | 2 | 1
#   8 | (1,1)  | sin | 2 | 1
MODES = [
    (0, 0, 'cos'), (0, 1, 'cos'), (0, 1, 'sin'), (1, -1, 'cos'), (1, -1, 'sin'),
    (1, 0, 'cos'), (1, 0, 'sin'), (1, 1, 'cos'), (1, 1, 'sin'),
]
DIM = len(MODES)            # 9
GROUPS = 2                  # G; group_id[m] = 0 if m < (d+1)//2 else 1  -> sizes 5, 4
NMODES = 2 ** GROUPS        # 4 sign-flip posterior modes

GAMMA2 = 2.0               # prior magnitude
PRIOR_S = 1.0              # prior smoothness exponent


def group_id():
    """[d] long tensor: contiguous group assignment (0 for first 5, 1 for last 4)."""
    return torch.tensor([0 if m < (DIM + 1) // 2 else 1 for m in range(DIM)], dtype=torch.long)


def prior_var():
    """[d] per-mode prior variance GAMMA2 / (1 + |k|^2)^s."""
    return torch.tensor(
        [GAMMA2 / (1.0 + (k1 * k1 + k2 * k2)) ** PRIOR_S for (k1, k2, p) in MODES],
        dtype=torch.get_default_dtype(),
    )


def cell_grid(n):
    """n x n cell-centered grid on [0,1]^2 -> [n*n, 2]."""
    c = (torch.arange(n, dtype=torch.get_default_dtype()) + 0.5) / n
    g = torch.stack(torch.meshgrid(c, c, indexing='ij'), dim=-1).reshape(-1, 2)
    return g


def eval_basis(pts):
    """[J, 2] grid points -> [J, d] basis matrix Phi (Phi[j, m] = phi_m(x_j))."""
    J = pts.shape[0]
    Phi = torch.zeros(J, DIM, dtype=pts.dtype)
    for m, (k1, k2, parity) in enumerate(MODES):
        arg = 2.0 * math.pi * (k1 * pts[:, 0] + k2 * pts[:, 1])
        Phi[:, m] = torch.cos(arg) if parity == 'cos' else torch.sin(arg)
    return Phi


def group_basis(pts):
    """[J, 2] -> [G, J, d] group basis Phi_g[g, j, m] = phi_m(x_j) * 1[m in group g]."""
    Phi = eval_basis(pts)                       # [J, d]
    gid = group_id()
    Phi_g = torch.zeros(GROUPS, pts.shape[0], DIM, dtype=Phi.dtype)
    for g in range(GROUPS):
        mask = (gid == g).to(Phi.dtype)
        Phi_g[g] = Phi * mask.unsqueeze(0)
    return Phi_g


def make_data(theta_star, Phi_g, sigma_obs, seed):
    """Synthetic observation data_{g,j} = v_g(x_j; theta*)^2 + N(0, sigma_obs^2),
    clamped to >= 0. Phi_g defines the (fine) generation grid. -> [G, J]."""
    v = torch.einsum('gjd,nd->ngj', Phi_g, theta_star.unsqueeze(0)).squeeze(0)  # [G, J]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = sigma_obs * torch.randn(v.shape, generator=gen, dtype=v.dtype)
    return (v ** 2 + noise).clamp(min=0.0)


def true_theta(seed=42, amp=2.5):
    """The fixed true field coefficients theta* = AMP * sqrt(prior_var) * z, z~N(0,I)."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    z = torch.randn(DIM, generator=gen, dtype=torch.get_default_dtype())
    return amp * torch.sqrt(prior_var()) * z


def mode_centers(theta_star):
    """The 4 known sign-flip mode centers: theta* with each group independently
    flipped. Returns [NMODES, d]. Bit b of index toggles group g = bit position."""
    gid = group_id()
    centers = []
    for b in range(NMODES):
        fl = theta_star.clone()
        for g in range(GROUPS):
            if (b >> g) & 1:
                fl[gid == g] *= -1.0
        centers.append(fl)
    return torch.stack(centers, dim=0)


class FourierFieldPosterior(Potential):
    """U(theta) = U_prior(theta) + U_like(theta), the negative log posterior.

    U_like uses the SQUARED partial field per group on the (Phi_g) grid:
        U_like = 0.5 * sum_{g,j} (v_g(x_j)^2 - data_{g,j})^2 / sigma_obs^2
    Squaring carries the per-group sign-flip symmetry -> 2^G modes.

    Pass the COARSE group basis for the training target; pass the FINE group
    basis (and fine data) for the honest post-training evaluation target.
    """

    def __init__(self, Phi_g, data, prior_var_, sigma_obs):
        super().__init__()
        self.register_buffer('Phi_g', Phi_g.clone())                 # [G, J, d]
        self.register_buffer('data', data.clone())                   # [G, J]
        self.register_buffer('prior_prec', 1.0 / prior_var_.clone())  # [d]
        self.noise2 = float(sigma_obs) ** 2

    def forward(self, theta):                                        # [N, d] -> [N]
        v = torch.einsum('gjd,nd->ngj', self.Phi_g, theta)          # [N, G, J]
        resid = v ** 2 - self.data.unsqueeze(0)                      # [N, G, J]
        U_like = 0.5 * (resid ** 2).sum(dim=(-1, -2)) / self.noise2
        U_prior = 0.5 * (theta ** 2 * self.prior_prec.unsqueeze(0)).sum(dim=-1)
        return U_prior + U_like


# ---------------------------------------------------------------------------
# Loss helpers (copied verbatim from the 2D_Benchmark / HD_Product templates)
# ---------------------------------------------------------------------------

# forward KL
def loss_KL(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    x, ladj = G.call_and_ladj(y)              # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    return z.mean()


# X functional (pairwise variation of the log-ratio under the batch's measure)
def loss_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return (z - z[perm]).abs().mean()


# Quench and Temper (QT) for mode discovery; needs target.enable_grad()+enable_eval().
# On this u^2 quartic the LBFGS step MUST be small (~0.05) with armijo; the default
# 1.0 overshoots and diverges to NaN (reviewer's single most important QT fix).
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


# ---------------------------------------------------------------------------
# Custom mode coverage (the HD_Product per-coordinate-sign version is WRONG here;
# assign each sample to its NEAREST of the 4 known sign-flip centers instead).
# ---------------------------------------------------------------------------

def mode_coverage_nearest(y: torch.Tensor, centers: torch.Tensor, frac: float = 0.05):
    """Assign each sample to its nearest of the NMODES sign-flip centers.

    Returns (modes_found / NMODES, occupancy_fraction[NMODES], counts[NMODES]).
    A mode counts as found if it holds >= `frac` of the samples (default 5%).
    """
    d = torch.cdist(y, centers)                              # [N, NMODES]
    assign = d.argmin(dim=1)                                  # [N]
    counts = torch.bincount(assign, minlength=centers.shape[0]).float()
    N = float(y.shape[0])
    occ = counts / max(N, 1.0)
    modes_found = int((occ >= frac).sum().item())
    return modes_found / centers.shape[0], occ.cpu(), counts.cpu()
