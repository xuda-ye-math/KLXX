"""Darcy_2D — Bayesian inversion for the source term of a periodic-domain elliptic PDE.

Forward model:
  -Laplacian u(x) = g(x; theta),   x in [0,1]^2 periodic,   mean(u) = 0,
  g(x; theta) = G0 * (1 + delta * cos(alpha * v(x; theta))),
  v(x; theta) = sum_m theta_m phi_m(x).

Spectral exact solve:
  hat u(k) = hat g(k) / |2 pi k|^2   for k != 0,    hat u(0) = 0.

Observation: bilinear-interpolated u(x_s) at N_SENSORS sensor positions.

Source/prior on theta: mu_0 = N(0, Sigma_0), (Sigma_0)_{mm} = 2 / (1 + |k_m|^2).
"""
import math
import torch
from zflows.potential import Potential
from zflows.flow import ComposedTransform
from zflows.utils import langevin, resample

MODES = [
    (0, 0, 'cos'), (0, 1, 'cos'), (0, 1, 'sin'), (1, -1, 'cos'), (1, -1, 'sin'),
    (1, 0, 'cos'), (1, 0, 'sin'), (1, 1, 'cos'), (1, 1, 'sin'),
]
DIM = len(MODES)
NMODES = 6
SIGN_FLIPS = (+1, -1)
SHIFT_KS   = (-1, 0, +1)


def cell_grid(n):
    c = (torch.arange(n, dtype=torch.get_default_dtype()) + 0.5) / n
    g = torch.stack(torch.meshgrid(c, c, indexing='ij'), dim=-1).reshape(-1, 2)
    return g


def eval_basis(pts):
    J = pts.shape[0]
    Phi = torch.zeros(J, DIM, dtype=pts.dtype)
    for m, (k1, k2, parity) in enumerate(MODES):
        arg = 2.0 * math.pi * (k1 * pts[:, 0] + k2 * pts[:, 1])
        Phi[:, m] = torch.cos(arg) if parity == 'cos' else torch.sin(arg)
    return Phi


def mode_centers(theta_star, alpha):
    T = 2.0 * math.pi / float(alpha)
    out = []
    for s in SIGN_FLIPS:
        for k in SHIFT_KS:
            w = s * theta_star.clone()
            w[0] = w[0] + k * T
            out.append(w)
    return torch.stack(out, dim=0)


def mode_coverage_nearest(samples, centers, frac=0.01):
    d2 = torch.cdist(samples, centers)
    assign = d2.argmin(dim=-1)
    counts = torch.bincount(assign, minlength=centers.shape[0]).float()
    N = float(samples.shape[0])
    threshold = max(1.0, frac * N / centers.shape[0])
    found = int((counts >= threshold).sum().item())
    occ = (counts / N).tolist()
    return found / centers.shape[0], occ, counts.long().tolist()


def prior_var():
    return torch.tensor(
        [2.0 / (1.0 + (k1 * k1 + k2 * k2)) for (k1, k2, _) in MODES],
        dtype=torch.get_default_dtype(),
    )


def inverse_lap_mult(n, dtype, device):
    """Spectral inverse-Laplacian multiplier: 1 / |2 pi k|^2 for k != 0, 0 for k = 0."""
    kx = torch.fft.fftfreq(n, d=1.0 / n).to(dtype=dtype, device=device)
    ky = torch.fft.fftfreq(n, d=1.0 / n).to(dtype=dtype, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = (2.0 * math.pi) ** 2 * (KX * KX + KY * KY)
    inv = torch.where(K2 > 0, 1.0 / K2, torch.zeros_like(K2))
    return inv


def sensor_indices(n, center, radius, n_sensors):
    cx, cy = center
    thetas = 2.0 * math.pi * torch.arange(n_sensors, dtype=torch.get_default_dtype()) / n_sensors
    xs = cx + radius * torch.cos(thetas)
    ys = cy + radius * torch.sin(thetas)
    fx = xs * n - 0.5; fy = ys * n - 0.5
    i0 = torch.floor(fx).long(); j0 = torch.floor(fy).long()
    wx = (fx - i0.to(fx.dtype)); wy = (fy - j0.to(fy.dtype))
    i0 = i0 % n; j0 = j0 % n
    i1 = (i0 + 1) % n; j1 = (j0 + 1) % n
    return i0, j0, i1, j1, wx, wy


def sensor_bilinear(u, sensor_idx):
    i0, j0, i1, j1, wx, wy = sensor_idx
    a00 = u[:, i0, j0]; a10 = u[:, i1, j0]; a01 = u[:, i0, j1]; a11 = u[:, i1, j1]
    one_mx = 1.0 - wx; one_my = 1.0 - wy
    return one_mx * one_my * a00 + wx * one_my * a10 + one_mx * wy * a01 + wx * wy * a11


def forward_observation(theta, Phi_flat, inv_lap, n_grid, sensor_idx,
                        g0, delta, alpha):
    """theta: [N, d] -> u_at_sensors [N, n_sensors].

    Source g(x;theta) = g0 * (1 + delta * cos(alpha * v(x;theta))).
    Elliptic spectral solve: hat u(k) = hat g(k) / |2 pi k|^2  (with 1/|2 pi k|^2 := 0 at k=0).
    """
    N_batch = theta.shape[0]
    v = (theta @ Phi_flat.T)                                              # [N, J]
    g = (g0 * (1.0 + delta * torch.cos(alpha * v))).view(N_batch, n_grid, n_grid)
    G = torch.fft.fft2(g)
    U = G * inv_lap                                                       # [N, n, n] complex
    u = torch.fft.ifft2(U).real                                           # [N, n, n]
    return sensor_bilinear(u, sensor_idx)                                 # [N, n_sensors]


class DarcyInverse(Potential):
    """Bayesian elliptic source-inversion on [0,1]^2 with periodic-cosine g."""

    def __init__(self, Phi_flat, inv_lap, sensor_idx, data, prior_var_,
                 n_grid, g0, delta, alpha, sigma_obs):
        super().__init__()
        self.register_buffer('Phi_flat', Phi_flat.clone())
        self.register_buffer('inv_lap',  inv_lap.clone())
        i0, j0, i1, j1, wx, wy = sensor_idx
        self.register_buffer('sens_i0', i0.clone()); self.register_buffer('sens_j0', j0.clone())
        self.register_buffer('sens_i1', i1.clone()); self.register_buffer('sens_j1', j1.clone())
        self.register_buffer('sens_wx', wx.clone()); self.register_buffer('sens_wy', wy.clone())
        self.register_buffer('data',       data.clone())
        self.register_buffer('prior_prec', 1.0 / prior_var_.clone())
        self.n_grid = int(n_grid)
        self.g0     = float(g0)
        self.delta  = float(delta)
        self.alpha  = float(alpha)
        self.noise2 = float(sigma_obs) ** 2

    def forward(self, theta):
        sensor_idx = (self.sens_i0, self.sens_j0, self.sens_i1, self.sens_j1,
                      self.sens_wx, self.sens_wy)
        obs = forward_observation(
            theta, self.Phi_flat, self.inv_lap, self.n_grid, sensor_idx,
            self.g0, self.delta, self.alpha,
        )
        resid = obs - self.data.unsqueeze(0)
        U_like  = 0.5 * (resid * resid).sum(dim=-1) / self.noise2
        U_prior = 0.5 * (theta * theta * self.prior_prec.unsqueeze(0)).sum(dim=-1)
        return U_prior + U_like


# Loss helpers (shared X-functional template) ---------------------------------

def loss_KL(y, source, target, G: ComposedTransform):
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    return z.mean()


def loss_X(y, source, target, G: ComposedTransform):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return (z - z[perm]).abs().mean()


def quench_and_temper(x, target, sigma, opt_step, opt_iters, mc_step, mc_iters, prior_std=None):
    """Anisotropic QT: when prior_std is provided ([d] tensor, sqrt of per-coord prior variance),
    the melt-step diffusion and Langevin temper are scaled per-coordinate so that high-variance
    modes get larger displacement and low-variance modes get tiny. Without prior_std the function
    reverts to the original isotropic QT.

    Reference: scaled-volatility MCMC on function-space Bayesian inverse problems; see also
    Jeffreys flow (Lin et al. 2026).
    """
    if prior_std is None:
        x = (x + sigma * torch.randn_like(x)).detach().clone().requires_grad_(True)
    else:
        # anisotropic diffusion: each coordinate diffused by sigma * prior_std[m]
        x = (x + sigma * prior_std.unsqueeze(0) * torch.randn_like(x)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=opt_step)
    for _ in range(opt_iters):
        opt.zero_grad()
        U = target(x).sum()
        U.backward()
        with torch.no_grad():
            x.grad.nan_to_num_(0.0).clamp_(-10.0, 10.0)
        opt.step()
    x = x.detach()
    if prior_std is None:
        x = langevin(x, target, step=mc_step, iters=mc_iters)
    else:
        x = preconditioned_langevin(x, target, prior_std, mc_step, mc_iters)
    return x


def preconditioned_langevin(x, target, prior_std, step, iters):
    """Anisotropic Langevin in the original coordinates, with diagonal preconditioner C = Sigma_0.
    Wrapped in enable_grad so it works when called from inside a torch.no_grad() context.
    """
    s_sq = prior_std * prior_std
    s    = prior_std
    noise_scale = (2.0 * step) ** 0.5
    for _ in range(iters):
        with torch.enable_grad():
            x_ = x.detach().clone().requires_grad_(True)
            U = target(x_).sum()
            grad = torch.autograd.grad(U, x_)[0]
        with torch.no_grad():
            grad = grad.nan_to_num_(0.0).clamp_(-1e6, 1e6)
            x = x_.detach() - step * s_sq.unsqueeze(0) * grad + noise_scale * s.unsqueeze(0) * torch.randn_like(x_)
    return x


def ais_M_pass(y, source, target, flow_t, M, step_size, iters_per_rung, prior_std=None):
    """M-rung annealed IS from nu = G^{-1}_# mu_0 to mu (paper Section 3.2).

    When prior_std is supplied, the inner Langevin runs with a diagonal preconditioner equal
    to the prior covariance — small step on low-variance modes, large step on high-variance ones.
    For the Fourier-coefficient targets with prior variance 2/(1+|k|^2), this avoids over-driving
    high-frequency modes while leaving the low-frequency modes (which dominate the likelihood)
    free to relax.
    """
    with torch.no_grad():
        x_curr, ladj_fwd = flow_t.call_and_ladj(y)
        current_logw = -target(y) + source(x_curr) - ladj_fwd
        for k in range(M):
            inc_logw = current_logw / M
            w = (inc_logw - inc_logw.max()).exp()
            y = resample(y, w)
            if prior_std is None:
                y = langevin(y, target, step=step_size, iters=iters_per_rung)
            else:
                y = preconditioned_langevin(y, target, prior_std, step_size, iters_per_rung)
            x_curr, ladj_fwd = flow_t.call_and_ladj(y)
            current_logw = -target(y) + source(x_curr) - ladj_fwd
    return y


def coverage(y: torch.Tensor, x: torch.Tensor, k: int = 5) -> float:
    dxx = torch.cdist(x, x)
    dxx.fill_diagonal_(float('inf'))
    nnd_k = dxx.topk(k, dim=1, largest=False).values[:, -1]
    dxy = torch.cdist(x, y)
    return (dxy < nnd_k.unsqueeze(1)).any(dim=1).float().mean().item()
