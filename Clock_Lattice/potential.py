# pyright: reportArgumentType=false
"""p-state clock model on the L x L periodic square lattice (potential-specific).

    U(theta) = -J sum_<ij> cos(theta_i - theta_j) - H sum_i cos(P theta_i),

one angle theta in [-pi, pi) per site, periodic boundary conditions.

Physics: the canonical bridge between Ising (P=2) and XY (P->inf). For P >= 5
in 2D it hosts two BKT transitions enclosing a critical quasi-long-range-
ordered phase (Jose-Kadanoff-Kirkpatrick-Nelson 1977). The cold Boltzmann
measure has exactly P symmetry-broken sectors (all spins near 2*pi*k/P) --
P equal-weight modes beyond the trivial Z2 case, with domain-wall saddles and
vortex metastability in between, so the posterior is genuinely non-trivial
while the mode set stays exactly enumerable for QT.
"""
import math

import torch
from zflows.potential import Potential


class Clock(Potential):
    def __init__(self, L: int, P: int, J: float, H: float):
        super().__init__()
        self.L, self.P = int(L), int(P)
        self.J, self.H = float(J), float(H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, L*L] angles -> U(x): [N]."""
        L = self.L
        th = x.view(-1, L, L)
        right = torch.roll(th, shifts=-1, dims=2)
        down = torch.roll(th, shifts=-1, dims=1)
        coupling = torch.cos(th - right) + torch.cos(th - down)
        aniso = torch.cos(self.P * th)
        return -(self.J * coupling + self.H * aniso).flatten(1).sum(-1)


def magnetization(y: torch.Tensor) -> torch.Tensor:
    """Complex global magnetization m = mean_j exp(i theta_j) per sample."""
    return torch.exp(1j * y.to(torch.float32)).mean(dim=1)


def sector_occupancy(y: torch.Tensor, P: int, frac: float = 0.01):
    """Assign each sample to the nearest of the P clock sectors via the phase
    of the global magnetization. Returns (coverage, tv, counts, mean_abs_m):
    coverage = fraction of the P sectors holding >= max(1, frac*N/P) samples,
    tv = total-variation distance of the sector histogram from uniform,
    mean_abs_m = mean |m| (order parameter; ~1 deep in the ordered phase)."""
    m = magnetization(y)
    ang = torch.angle(m)                                  # [-pi, pi)
    sector = torch.round(ang * P / (2.0 * math.pi)).long() % P
    counts = torch.bincount(sector, minlength=P).float()
    N = y.shape[0]
    threshold = max(1.0, frac * N / P)
    found = int((counts >= threshold).sum().item())
    p = counts / counts.sum()
    tv = 0.5 * (p - 1.0 / P).abs().sum().item()
    return found / P, tv, counts.tolist(), m.abs().mean().item()


def torus_coverage(y: torch.Tensor, x: torch.Tensor, k: int = 5) -> float:
    """kNN coverage (Naeem et al. 2020) with the torus geometry: embed each
    angle as (cos, sin) so Euclidean distance is the chord metric (monotone in
    the torus distance), then the usual k-nearest-neighbor ball test."""
    def emb(t):
        return torch.cat([torch.cos(t), torch.sin(t)], dim=1)
    ye, xe = emb(y), emb(x)
    dxx = torch.cdist(xe, xe)
    dxx.fill_diagonal_(float('inf'))
    nnd_k = dxx.topk(k, dim=1, largest=False).values[:, -1]
    dxy = torch.cdist(xe, ye)
    return (dxy < nnd_k.unsqueeze(1)).any(dim=1).float().mean().item()
