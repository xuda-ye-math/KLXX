# pyright: reportArgumentType=false
"""Truncation-ceiling gate (PLAN.md Sec. 6) -- runs BEFORE any training.

(a) whitened curvature ratio per mode (finite differences of the misfit),
(b) Var_{xi_m ~ N(0,1)} U_full vs |k| (other modes at truth),
(c) training-free ceiling ESS: xi_low from a PT-MALA posterior run, xi_high
    from the prior, full reweighting,
(c') stress: xi_low from heated chains (t = 0.7, 0.5).

Writes diagnostic.md + figures/diagnostic.png. Smoke geometry (4x4 in 8x8) by
default; --m-low/--m-full override.  Runtime target: ~1-2 min on GPU.
"""
import argparse
import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zflows.utils import compute_ESS_log, langevin

import parameters as p
import potential as pot

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

cli = argparse.ArgumentParser()
cli.add_argument('--m-low', type=int, default=p.M_LOW)
cli.add_argument('--m-full', type=int, default=p.M_FULL)
cli.add_argument('--n-samp', type=int, default=4096)
cli.add_argument('--prior-s', type=int, default=p.PRIOR_S)
cli.add_argument('--sigma-obs', type=float, default=p.SIGMA_OBS)
cli.add_argument('--c2', type=float, default=p.C2)
cli.add_argument('--delta', type=float, default=p.DELTA)
cli.add_argument('--n-sensors', type=int, default=p.N_SENSORS)
cli.add_argument('--tilt', type=float, default=p.EPS_TILT)
cli.add_argument('--tag', type=str, default='')
args = cli.parse_args()
p.M_LOW, p.M_FULL = args.m_low, args.m_full
p.PRIOR_S, p.SIGMA_OBS, p.C2 = args.prior_s, args.sigma_obs, args.c2
p.DELTA, p.N_SENSORS, p.EPS_TILT = args.delta, args.n_sensors, args.tilt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

B = pot.build(p, device)
modes, d_low, d_full = B['modes'], B['d_low'], B['d_full']
std_all = B['std_all']
u_full, u_low = B['u_full'], B['u_low']
xi_truth = B['xi_truth'].to(device)
kabs = torch.tensor([math.sqrt(k1 * k1 + k2 * k2) for k1, k2, _ in modes])

print(f"geometry: {args.m_low}x{args.m_low} in {args.m_full}x{args.m_full} "
      f"(d_low={d_low}, d_full={d_full}), grid {p.N_GRID}, device {device}")

# ---------------- (a) whitened Gauss-Newton curvature ratio ------------------
# rho_m = sum_s (d obs_s / d xi_m)^2 / sigma_obs^2 against whitened prior
# curvature 1. Exact autograd Jacobian (finite differences in float32 are
# cancellation noise at this scale).
def obs_of_xi(xi_vec):
    theta = (xi_vec * u_full.std_act).unsqueeze(0)
    idx = (u_full.si0, u_full.sj0, u_full.si1, u_full.sj1, u_full.swx, u_full.swy)
    return forward_observation(theta, u_full.Phi_act, u_full.inv_mult,
                               u_full.n_grid, idx, u_full.g0, u_full.delta,
                               u_full.alpha, u_full.eps_tilt)[0]

from potential import forward_observation
J = torch.autograd.functional.jacobian(obs_of_xi, xi_truth.clone(),
                                       vectorize=True)        # [N_S, d_full]
rho = (J ** 2).sum(dim=0).cpu() / p.SIGMA_OBS ** 2
ext = slice(d_low, d_full)
print(f"(a) curvature ratio: low-block max {rho[:d_low].max():.3g}, "
      f"extension max {rho[ext].max():.3g}, extension mean {rho[ext].mean():.3g}")

# ---------------- (b) Var over one prior-drawn mode --------------------------
nv = 512
varU = torch.zeros(d_full)
for m in range(d_full):
    X = xi_truth.unsqueeze(0).repeat(nv, 1)
    X[:, m] = torch.randn(nv, device=device)
    varU[m] = u_full.misfit(X).var().item()
print(f"(b) Var U: low max {varU[:d_low].max():.3g}, ext max {varU[ext].max():.3g}")

# ---------------- (c)/(c') ceiling ESS ---------------------------------------
def pt_mala_lowmode(temper: float, n_keep: int, n_chains: int = 256,
                    iters: int = 2000, step: float = 5e-3):
    """Cheap posterior sampler on the LOW block at likelihood temper t:
    Langevin on U = prior + t*Phi_low from overdispersed starts (this is a
    diagnostic sampler, not the referee; multimodality is handled by the
    overdispersed init covering the shift/sign wells)."""
    u_t = B['make'](d_low, temper=temper).enable_grad()
    x = 2.0 * torch.randn(n_chains, d_low, device=device)
    x = langevin(x, u_t, step=step, iters=iters)
    reps = (n_keep + n_chains - 1) // n_chains
    xs = [x]
    for _ in range(reps - 1):
        x = langevin(x, u_t, step=step, iters=200)
        xs.append(x)
    return torch.cat(xs)[:n_keep]

def ceiling_ess(xi_low_samples: torch.Tensor) -> float:
    """ESS of [xi_low; xi_high~prior] reweighted low-posterior x prior -> full."""
    n = xi_low_samples.shape[0]
    xi_h = torch.randn(n, d_full - d_low, device=device)
    xi = torch.cat([xi_low_samples, xi_h], dim=1)
    # proposal density: pi_low(xi_L) * N(xi_H; 0, I); target: pi_full
    # log w = [U_low(xi_L) + 0.5|xi_H|^2] - U_full(xi)
    #       = Phi_low(xi_L) - Phi_full(xi)        (priors cancel exactly)
    logw = u_low.misfit(xi_low_samples) - u_full.misfit(xi)
    return compute_ESS_log(logw).item()

results = {}
for tag, t in [('c_t1.0', 1.0), ('cp_t0.7', 0.7), ('cp_t0.5', 0.5)]:
    xs = pt_mala_lowmode(t, args.n_samp)
    results[tag] = ceiling_ess(xs)
    print(f"({tag}) ceiling ESS = {results[tag]:.4f}")

# ---------------- gate verdict + artifacts -----------------------------------
gate_a = rho[ext].max().item() < 0.1
gate_c = results['c_t1.0'] > 0.5
gate_cp = results['cp_t0.7'] > 0.2
verdict = 'PASS' if (gate_a and gate_c and gate_cp) else 'FAIL'
print(f"GATE: {verdict}  (a<0.1: {gate_a}, c>0.5: {gate_c}, c'>0.2: {gate_cp})")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
ax = axes[0]
ax.semilogy(kabs[:d_low], rho[:d_low].clamp(min=1e-12), 'o', ms=4, label='low block')
ax.semilogy(kabs[ext], rho[ext].clamp(min=1e-12), 's', ms=4, color='tab:red',
            label='extension')
ax.axhline(0.1, color='black', ls='--', lw=1)
ax.set_xlabel(r'$|k|$'); ax.set_ylabel('whitened curvature ratio')
ax.legend(fontsize=8); ax.set_title('(a) likelihood vs prior curvature')
ax = axes[1]
ax.semilogy(kabs[:d_low], varU[:d_low].clamp(min=1e-12), 'o', ms=4)
ax.semilogy(kabs[ext], varU[ext].clamp(min=1e-12), 's', ms=4, color='tab:red')
ax.set_xlabel(r'$|k|$'); ax.set_ylabel(r'Var$_{\xi_m}\,\Phi$')
ax.set_title('(b) per-mode misfit variance')
plt.tight_layout()
plt.savefig(FIG / 'diagnostic.png', dpi=400, bbox_inches='tight', pad_inches=0.02)

out_md = HERE / (f'diagnostic{args.tag}.md' if args.tag else 'diagnostic.md')
with open(out_md, 'w') as f:
    f.write(f"# Truncation-ceiling gate: {verdict}\n\n"
            f"Geometry {args.m_low}x{args.m_low} in {args.m_full}x{args.m_full} "
            f"(d_low={d_low}, d_full={d_full}); grid {p.N_GRID}; "
            f"alpha={p.ALPHA}, c^2={p.C2}, sigma_obs={p.SIGMA_OBS}, "
            f"prior A={p.PRIOR_AMP}, s={p.PRIOR_S}, tilt={p.EPS_TILT}.\n\n"
            f"| check | value | gate |\n|---|---|---|\n"
            f"| (a) max extension curvature ratio | {rho[ext].max():.4g} | < 0.1: {gate_a} |\n"
            f"| (b) max extension Var Phi | {varU[ext].max():.4g} | (info) |\n"
            f"| (c) ceiling ESS, t=1.0 | {results['c_t1.0']:.4f} | > 0.5: {gate_c} |\n"
            f"| (c') ceiling ESS, t=0.7 | {results['cp_t0.7']:.4f} | > 0.2: {gate_cp} |\n"
            f"| (c') ceiling ESS, t=0.5 | {results['cp_t0.5']:.4f} | (info) |\n\n"
            f"Figure: figures/diagnostic.png. Low-block max curvature ratio "
            f"{rho[:d_low].max():.4g} (should be >> extension: the data must "
            f"constrain the trained block, else the posterior is trivial).\n")
print("wrote diagnostic.md, figures/diagnostic.png")
