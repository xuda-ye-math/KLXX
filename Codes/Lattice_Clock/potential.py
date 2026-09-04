"""p-state clock model on the L x L periodic square lattice (potential-specific).

    U(theta) = -J sum_<ij> cos(theta_i - theta_j) - H sum_i cos(P theta_i),

one angle theta in [-pi, pi) per site, periodic boundary conditions.

Physics: the canonical interpolation between Ising (P=2) and XY (P->inf). For P >= 5
in 2D it hosts two BKT transitions enclosing a critical quasi-long-range-
ordered phase (Jose-Kadanoff-Kirkpatrick-Nelson 1977). The cold Boltzmann
measure has exactly P symmetry-broken sectors (all spins near 2*pi*k/P) --
P equal-weight modes beyond the trivial Z2 case, with domain-wall saddles and
vortex metastability in between, so the posterior is genuinely non-trivial
while the mode set stays exactly enumerable for QT.

All observables are wrap-invariant (they enter through exp(i*theta) or the
chord embedding (cos, sin)), so unwrapped angles from Langevin / L-BFGS need
no explicit reduction into the box.
"""

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from jflows.potential import Potential


class Clock(Potential):
    """p-state clock model, one angle per site of the periodic L x L lattice."""

    L: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    J: float = eqx.field(static=True)
    H: float = eqx.field(static=True)

    def __init__(self, L: int, P: int, J: float, H: float):
        self.L = int(L)
        self.P = int(P)
        self.J = float(J)
        self.H = float(H)

    def __call__(self, x: Array) -> Array:   # Array [N, L*L] -> Array [N]
        L = self.L
        th = x.reshape(-1, L, L)
        right = jnp.roll(th, shift=-1, axis=2)
        down = jnp.roll(th, shift=-1, axis=1)
        coupling = jnp.cos(th - right) + jnp.cos(th - down)
        aniso = jnp.cos(self.P * th)
        return -(self.J * coupling + self.H * aniso).reshape(x.shape[0], -1).sum(-1)


def magnetization(y: np.ndarray) -> np.ndarray:
    """Complex global magnetization m = mean_j exp(i theta_j) per sample."""
    return np.exp(1j * y.astype(np.float64)).mean(axis=1)


def sector_occupancy(y: np.ndarray, P: int, frac: float = 0.01):
    """Assign each sample to the nearest of the P clock sectors via the phase
    of the global magnetization. Returns (coverage, tv, counts, mean_abs_m):
    coverage = fraction of the P sectors holding >= max(1, frac*N/P) samples,
    tv = total-variation distance of the sector histogram from uniform,
    mean_abs_m = mean |m| (order parameter; ~1 deep in the ordered phase)."""
    m = magnetization(y)
    ang = np.angle(m)                                     # [-pi, pi)
    sector = np.round(ang * P / (2.0 * math.pi)).astype(np.int64) % P
    counts = np.bincount(sector, minlength=P).astype(np.float64)
    N = y.shape[0]
    threshold = max(1.0, frac * N / P)
    found = int((counts >= threshold).sum())
    p = counts / counts.sum()
    tv = 0.5 * np.abs(p - 1.0 / P).sum()
    return found / P, float(tv), counts.tolist(), float(np.abs(m).mean())


def torus_coverage(y: np.ndarray, x: np.ndarray, k: int = 5) -> float:
    """kNN coverage (Naeem et al. 2020) with the torus geometry: embed each
    angle as (cos, sin) so Euclidean distance is the chord metric (monotone in
    the torus distance), then the usual k-nearest-neighbor ball test —
    the fraction of reference points x with a generated sample y inside
    their k-th-nearest-neighbor ball."""
    def emb(t):
        t = t.astype(np.float64)
        return np.concatenate([np.cos(t), np.sin(t)], axis=1)

    def cdist(a, b):
        d2 = ((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * (a @ b.T))
        return np.sqrt(np.clip(d2, 0.0, None))

    ye, xe = emb(y), emb(x)
    dxx = cdist(xe, xe)
    np.fill_diagonal(dxx, np.inf)
    nnd_k = np.sort(dxx, axis=1)[:, k - 1]                # k-th NN distance
    dxy = cdist(xe, ye)
    return float((dxy < nnd_k[:, None]).any(axis=1).mean())
