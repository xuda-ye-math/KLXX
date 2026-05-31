"""Sensor-array source-localization Bayesian inverse problem with a
permutation-symmetry-induced multimodal posterior.

A line of N_SENSORS fixed sensors observes the superposed signal of N_SRC
identical point sources at unknown positions theta = (theta_1, ..., theta_n).
Each sensor s_j reads the sum of the sources' Gaussian footprints,

    g_j(theta) = sum_{i=1}^{n} exp( -(s_j - theta_i)^2 / (2 ell^2) ),

and the data are y_j = g_j(theta*) + noise. Because the sources are identical,
the forward map g -- and hence the likelihood -- is exactly invariant under any
permutation of the source positions. A generic theta* therefore has an orbit of
n! = NMODES degenerate, well-separated posterior modes (the permutations of
theta*), while the centered Gaussian prior (the source mu_0) sits on their common
permutation-symmetry saddle at the origin and covers none of them.

This module is self-contained (the loss / QT / coverage helpers are copied
locally, matching the 2D_Benchmark and Fourier_Modes folders). The only new code
vs the templates is SensorArrayPosterior + the sensor / mode helpers.
"""
import itertools

import torch
from zflows.potential import Potential
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin


# ---------------------------------------------------------------------------
# Sensor-array forward model / mode helpers (the genuinely new code)
# ---------------------------------------------------------------------------

def sensor_positions(n_sensors, lim):
    """[J] sensors equispaced on [-lim, lim]."""
    return torch.linspace(-lim, lim, n_sensors, dtype=torch.get_default_dtype())


def true_theta(theta_star):
    """The fixed true source positions as a tensor."""
    return torch.tensor(theta_star, dtype=torch.get_default_dtype())


def mode_centers(theta_star):
    """The n! known permutation mode centers: every permutation of theta*.
    Returns [NMODES, n]."""
    t = torch.as_tensor(theta_star, dtype=torch.get_default_dtype())
    perms = list(itertools.permutations(range(t.shape[0])))
    return torch.stack([t[list(p)] for p in perms], dim=0)


def forward_signal(theta, sensors, ell):
    """Superposed Gaussian-footprint signal.
    theta [N, n], sensors [J] -> g [N, J], g[:, j] = sum_i exp(-(s_j-theta_i)^2/2ell^2)."""
    diff = theta.unsqueeze(-1) - sensors                 # [N, n, J]
    bumps = torch.exp(-0.5 * diff ** 2 / ell ** 2)       # [N, n, J]
    return bumps.sum(dim=-2)                              # [N, J]


def make_data(theta_star, sensors, ell, sigma_obs, seed):
    """Synthetic observation y_j = g_j(theta*) + N(0, sigma_obs^2). -> [J]."""
    t = torch.as_tensor(theta_star, dtype=torch.get_default_dtype()).unsqueeze(0)
    g = forward_signal(t, sensors, ell).squeeze(0)       # [J]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = sigma_obs * torch.randn(g.shape, generator=gen, dtype=g.dtype)
    return g + noise


class SensorArrayPosterior(Potential):
    """U(theta) = U_prior(theta) + U_like(theta), the negative log posterior.

        U_like  = 0.5 * sum_j ( g_j(theta) - y_j )^2 / sigma_obs^2
        U_prior = 0.5 * ||theta||^2 / sigma_prior^2

    The permutation invariance of g carries through to U -> n! degenerate modes.
    """

    def __init__(self, sensors, data, ell, sigma_obs, sigma_prior):
        super().__init__()
        self.register_buffer('sensors', sensors.clone())     # [J]
        self.register_buffer('data', data.clone())           # [J]
        self.ell = float(ell)
        self.noise2 = float(sigma_obs) ** 2
        self.prior_prec = 1.0 / float(sigma_prior) ** 2

    def forward(self, theta):                                 # [N, n] -> [N]
        g = forward_signal(theta, self.sensors, self.ell)    # [N, J]
        resid = g - self.data.unsqueeze(0)                   # [N, J]
        U_like = 0.5 * (resid ** 2).sum(dim=-1) / self.noise2
        U_prior = 0.5 * (theta ** 2).sum(dim=-1) * self.prior_prec
        return U_prior + U_like


# ---------------------------------------------------------------------------
# Loss helpers (copied verbatim from the 2D_Benchmark / Fourier_Modes templates)
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
# Sharp modes (small sigma_obs) -> use a SMALL armijo LBFGS step so the quench does
# not overshoot the narrow basins and diverge to NaN.
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
# Mode coverage: assign each sample to its NEAREST of the n! permutation centers.
# ---------------------------------------------------------------------------

def mode_coverage_nearest(y: torch.Tensor, centers: torch.Tensor, frac: float = 0.05):
    """Assign each sample to its nearest of the NMODES permutation centers.

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
