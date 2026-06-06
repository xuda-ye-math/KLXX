# pyright: reportArgumentType=false
"""Screened-Poisson Bayesian source inversion (potential-specific).

Unknown field on the periodic unit square, parameterized by a real Fourier basis:

    v(x; theta) = sum_m theta_m phi_m(x),

phi_m = sqrt(2) cos(2 pi k.x) / sqrt(2) sin(2 pi k.x) over a half-plane of wave
vectors k (constant mode included once), enumerated LOW BLOCK FIRST: all modes
with max(|k1|,|k2|) <= m_low/2 precede the extension modes of the full m_full set.

Forward (spectral, exact on the grid):

    (-Lap + c^2) u = G0 (1 + delta cos(alpha v)) + eps_tilt v,
    y_s = u(x_s) at N_SENSORS ring sensors (bilinear),
    Phi(theta)  = |y_obs - F(theta)|^2 / (2 sigma_obs^2).

WHITENING IS A HARD RULE: every public Potential acts on whitened xi with
theta = sqrt(prior_var) * xi, so the prior is N(0, I) and all Langevin/QT/SMC
steps are isotropic. Raw theta never leaves this module.

    U_full(xi)      = 0.5|xi|^2 + Phi(S^{1/2} xi)          on R^{d_full}
    U_low(xi_low)   = 0.5|xi_low|^2 + Phi(S^{1/2}[xi_low; 0])  on R^{d_low}
    bridge: (1-t) U_0 + t U_low = 0.5|xi|^2 + t Phi_low    (likelihood tempering)

Multimodality: cos(alpha v) is invariant under v -> -v (theta -> -theta) and
v -> v + 2 pi n / alpha (constant-mode shift); eps_tilt*v breaks the sign
symmetry so the well weights are nontrivial.
"""
import math

import torch
from zflows.potential import Potential


# ---------------------------------------------------------------------------
# mode enumeration: half-plane real basis, low block first
# ---------------------------------------------------------------------------
def mode_list(m_low: int, m_full: int):
    """Real-basis mode table [(k1, k2, parity)]: the m_full^2 lowest-frequency
    real modes of the half-plane enumeration sorted by (|k|^2, lex), with the
    m_low^2 lowest forming the trained block (isotropic cutoff; cos before sin
    at equal k). Exactly m^2 real coefficients per 'm x m' set by construction."""
    kmax = m_full        # generous lattice; the sort + truncation does the cutoff
    pairs = [(0, 0)] + [(k1, k2) for k1 in range(0, kmax + 1)
                        for k2 in range(-kmax, kmax + 1)
                        if not (k1 == 0 and k2 <= 0)]
    pairs.sort(key=lambda t: (t[0] ** 2 + t[1] ** 2, t))
    out = [(0, 0, 'cos')]
    for k1, k2 in pairs[1:]:
        out.append((k1, k2, 'cos'))
        out.append((k1, k2, 'sin'))
    modes = out[:m_full ** 2]
    assert len(modes) == m_full ** 2
    return modes, m_low ** 2


def prior_std(modes, amp: float, s: int) -> torch.Tensor:
    """Per-mode prior std sqrt(amp / (1+|k|^2)^s)."""
    k2 = torch.tensor([float(k1 * k1 + k2_ * k2_) for k1, k2_, _ in modes])
    return (amp / (1.0 + k2) ** s).sqrt()


def eval_basis(modes, pts: torch.Tensor) -> torch.Tensor:
    """Phi[n_pts, n_modes] with phi_m(x) = sqrt(2) cos/sin(2 pi k.x) (1 for k=0)."""
    n_pts = pts.shape[0]
    Phi = torch.empty(n_pts, len(modes), dtype=pts.dtype)
    for m, (k1, k2, parity) in enumerate(modes):
        arg = 2.0 * math.pi * (k1 * pts[:, 0] + k2 * pts[:, 1])
        col = torch.cos(arg) if parity == 'cos' else torch.sin(arg)
        scale = 1.0 if (k1 == 0 and k2 == 0) else math.sqrt(2.0)
        Phi[:, m] = scale * col
    return Phi


# ---------------------------------------------------------------------------
# spectral solve helpers (verbatim logic from Darcy_sweep/T001, c^2 added)
# ---------------------------------------------------------------------------
def cell_grid(n: int) -> torch.Tensor:
    ax = (torch.arange(n, dtype=torch.get_default_dtype()) + 0.5) / n
    gx, gy = torch.meshgrid(ax, ax, indexing='ij')
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def inv_helmholtz_mult(n: int, c2: float, dtype, device) -> torch.Tensor:
    """1 / (|2 pi k|^2 + c^2) on the fft2 grid (c^2 > 0: no k=0 special case)."""
    kx = torch.fft.fftfreq(n, d=1.0 / n).to(dtype=dtype, device=device)
    KX, KY = torch.meshgrid(kx, kx, indexing='ij')
    return 1.0 / ((2.0 * math.pi) ** 2 * (KX ** 2 + KY ** 2) + c2)


def sensor_indices(n: int, center, radius: float, n_sensors: int):
    cx, cy = center
    thetas = 2.0 * math.pi * torch.arange(n_sensors, dtype=torch.get_default_dtype()) / n_sensors
    xs = cx + radius * torch.cos(thetas)
    ys = cy + radius * torch.sin(thetas)
    fi, fj = xs * n - 0.5, ys * n - 0.5
    i0, j0 = fi.floor().long() % n, fj.floor().long() % n
    i1, j1 = (i0 + 1) % n, (j0 + 1) % n
    wx, wy = fi - fi.floor(), fj - fj.floor()
    return i0, j0, i1, j1, wx, wy


def sensor_bilinear(u: torch.Tensor, idx) -> torch.Tensor:
    i0, j0, i1, j1, wx, wy = idx
    return ((1 - wx) * (1 - wy) * u[:, i0, j0] + wx * (1 - wy) * u[:, i1, j0]
            + (1 - wx) * wy * u[:, i0, j1] + wx * wy * u[:, i1, j1])


def forward_observation(theta, Phi_flat, inv_mult, n_grid, sensor_idx,
                        g0, delta, alpha, eps_tilt):
    """theta: [N, d] raw coefficients -> u at sensors [N, n_sensors]."""
    N = theta.shape[0]
    v = theta @ Phi_flat.T                                       # [N, n_grid^2]
    g = g0 * (1.0 + delta * torch.cos(alpha * v)) + eps_tilt * v
    G = torch.fft.fft2(g.view(N, n_grid, n_grid))
    u = torch.fft.ifft2(G * inv_mult.unsqueeze(0)).real
    return sensor_bilinear(u, sensor_idx)


# ---------------------------------------------------------------------------
# whitened potentials
# ---------------------------------------------------------------------------
class PoissonInverse(Potential):
    """U(xi) = 0.5|xi|^2 + temper * Phi(S^{1/2}_active xi), acting on whitened
    xi over the FIRST n_active modes of the table (n_active = d_low for the
    training potential, = d_full for the referee/extension potential); the
    remaining modes of the table are pinned to zero inside the forward."""

    def __init__(self, modes, n_active, std_all, data, n_grid, c2,
                 g0, delta, alpha, eps_tilt, sigma_obs, temper: float = 1.0):
        super().__init__()
        pts = cell_grid(n_grid)
        Phi_flat = eval_basis(modes, pts)                        # [n_grid^2, d_table]
        self.register_buffer('Phi_act', Phi_flat[:, :n_active].clone())
        self.register_buffer('std_act', std_all[:n_active].clone())
        self.register_buffer('inv_mult',
                             inv_helmholtz_mult(n_grid, c2, Phi_flat.dtype, 'cpu'))
        idx = sensor_indices(n_grid, SENSOR_RING_CENTER_, SENSOR_RING_RADIUS_,
                             N_SENSORS_)
        for name, t in zip(('si0', 'sj0', 'si1', 'sj1'), idx[:4]):
            self.register_buffer(name, t.clone())
        self.register_buffer('swx', idx[4].clone())
        self.register_buffer('swy', idx[5].clone())
        self.register_buffer('data', data.clone())
        self.n_grid, self.n_active = int(n_grid), int(n_active)
        self.g0, self.delta = float(g0), float(delta)
        self.alpha, self.eps_tilt = float(alpha), float(eps_tilt)
        self.noise2 = float(sigma_obs) ** 2
        self.temper = float(temper)

    def misfit(self, xi: torch.Tensor) -> torch.Tensor:
        """Phi(S^{1/2} xi): [N, n_active] whitened -> [N]."""
        theta = xi * self.std_act.unsqueeze(0)
        idx = (self.si0, self.sj0, self.si1, self.sj1, self.swx, self.swy)
        obs = forward_observation(theta, self.Phi_act, self.inv_mult,
                                  self.n_grid, idx, self.g0, self.delta,
                                  self.alpha, self.eps_tilt)
        resid = obs - self.data.unsqueeze(0)
        return 0.5 * (resid * resid).sum(dim=-1) / self.noise2

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return 0.5 * (xi * xi).sum(dim=-1) + self.temper * self.misfit(xi)


# module-level sensor constants filled by build() (kept out of __init__ args
# to mirror the parameters.py single-source-of-truth pattern)
SENSOR_RING_CENTER_ = (0.5, 0.5)
SENSOR_RING_RADIUS_ = 0.3
N_SENSORS_ = 12


def build(p, device):
    """From a parameters module p: mode table, stds, truth, data, and the two
    whitened potentials (U_low on d_low, U_full on d_full). Returns a dict."""
    global SENSOR_RING_CENTER_, SENSOR_RING_RADIUS_, N_SENSORS_
    SENSOR_RING_CENTER_ = p.SENSOR_RING_CENTER
    SENSOR_RING_RADIUS_ = p.SENSOR_RING_RADIUS
    N_SENSORS_ = p.N_SENSORS

    modes, d_low = mode_list(p.M_LOW, p.M_FULL)
    d_full = len(modes)
    std_all = prior_std(modes, p.PRIOR_AMP, p.PRIOR_S)

    gen = torch.Generator().manual_seed(p.SEED_TRUTH)
    xi_truth = torch.randn(d_full, generator=gen)
    theta_truth = xi_truth * std_all

    pts = cell_grid(p.N_GRID)
    Phi_flat = eval_basis(modes, pts)
    inv_mult = inv_helmholtz_mult(p.N_GRID, p.C2, Phi_flat.dtype, 'cpu')
    idx = sensor_indices(p.N_GRID, p.SENSOR_RING_CENTER, p.SENSOR_RING_RADIUS,
                         p.N_SENSORS)
    obs_clean = forward_observation(theta_truth.unsqueeze(0), Phi_flat, inv_mult,
                                    p.N_GRID, idx, p.G0, p.DELTA, p.ALPHA,
                                    p.EPS_TILT)[0]
    gen_n = torch.Generator().manual_seed(p.SEED_NOISE)
    data = obs_clean + p.SIGMA_OBS * torch.randn(p.N_SENSORS, generator=gen_n)

    def make(n_active, temper=1.0):
        pot = PoissonInverse(modes, n_active, std_all, data, p.N_GRID, p.C2,
                             p.G0, p.DELTA, p.ALPHA, p.EPS_TILT, p.SIGMA_OBS,
                             temper=temper).to(device)
        return pot

    return dict(modes=modes, d_low=d_low, d_full=d_full, std_all=std_all,
                xi_truth=xi_truth, theta_truth=theta_truth, data=data,
                obs_clean=obs_clean, make=make,
                u_low=make(d_low), u_full=make(d_full))


# ---------------------------------------------------------------------------
# observables / diagnostics
# ---------------------------------------------------------------------------
def well_coordinates(xi: torch.Tensor, std_all: torch.Tensor, alpha: float):
    """(s, n): sign and shift labels of the well containing each sample.
    The shift acts on the CONSTANT mode theta_0 (table position 0): wells sit
    at theta_0 ~ theta_0* + 2 pi n / alpha with v -> -v doubling. We label by
    the projection of theta_0 onto the shift lattice and the sign of the
    dominant nonconstant low mode (set after the pilot freezes truth)."""
    theta0 = xi[:, 0] * std_all[0]
    n = torch.round(theta0 * alpha / (2.0 * math.pi)).long()
    return n


def quench_and_temper(target: Potential, n: int, d: int, sigma: float,
                      opt_step: float, opt_iters: int,
                      mc_step: float, mc_iters: int, device) -> torch.Tensor:
    """Algorithm 2 on R^d (whitened space): Gaussian melt at scale sigma about
    the origin (prior = N(0, I): melt sigma ~ a few prior stds), L-BFGS quench,
    Langevin temper. Mirrors the 2D pipeline's quench_and_temper."""
    from zflows.utils import langevin, lbfgs
    x = sigma * torch.randn(n, d, device=device)
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)
    x = langevin(x, target, step=mc_step, iters=mc_iters)
    return x
