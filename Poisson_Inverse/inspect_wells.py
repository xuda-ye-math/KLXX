# pyright: reportArgumentType=false
"""Well-structure inspection of the LOW-MODE posterior at full temper (t=1).

(1) QT (melt -> L-BFGS quench -> temper) enumerates the minima; greedy
    radius clustering of the quenched pool gives the well census.
(2) Long MALA from overdispersed starts gives rough occupancies.
(3) kNN coverage (Naeem et al., as in the 2D benchmark) cross-checks that the
    MALA samples reach every QT-discovered well and vice versa.

Writes wells.md + figures/wells.png.  Usage:
    ~/.envs/torch/bin/python inspect_wells.py --m-low 4 --m-full 8
    ~/.envs/torch/bin/python inspect_wells.py --m-low 6 --m-full 12 --n-grid 48
"""
import argparse
import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zflows.utils import langevin, lbfgs

import parameters as p
import potential as pot

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

cli = argparse.ArgumentParser()
cli.add_argument('--m-low', type=int, default=4)
cli.add_argument('--m-full', type=int, default=8)
cli.add_argument('--n-grid', type=int, default=p.N_GRID)
cli.add_argument('--n-pool', type=int, default=8000)
cli.add_argument('--cluster-radius', type=float, default=1.5)
cli.add_argument('--sigma-obs', type=float, default=p.SIGMA_OBS)
cli.add_argument('--delta', type=float, default=p.DELTA)
cli.add_argument('--n-sensors', type=int, default=p.N_SENSORS)
cli.add_argument('--alpha', type=float, default=p.ALPHA)
cli.add_argument('--tilt', type=float, default=p.EPS_TILT)
cli.add_argument('--prior-s', type=int, default=p.PRIOR_S)
cli.add_argument('--cluster', choices=['quench', 'mala', 'sym'], default='sym',
                 help='mala: cluster posterior mass (recommended); quench: raw minima')
args = cli.parse_args()
p.M_LOW, p.M_FULL, p.N_GRID = args.m_low, args.m_full, args.n_grid
p.SIGMA_OBS, p.DELTA, p.N_SENSORS, p.ALPHA = (args.sigma_obs, args.delta,
                                              args.n_sensors, args.alpha)
p.EPS_TILT = args.tilt
p.PRIOR_S = args.prior_s

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
B = pot.build(p, device)
d, std_all = B['d_low'], B['std_all'].to(device)
u = B['u_low'].enable_grad().enable_eval()
xi_truth = B['xi_truth'][:d].to(device)

# ---------------- (1) QT census --------------------------------------------
melt = 2.0 * torch.randn(args.n_pool, d, device=device)
quenched = lbfgs(melt, u, step=p.OPT_STEP, iters=200, armijo=True)
qt = langevin(quenched, u, step=p.MC_STEP, iters=p.MC_ITERS)

# greedy radius clustering of the QUENCHED minima (pre-temper, tight)
order = torch.argsort(u.eval(quenched).clone())
centers, labels = [], torch.full((args.n_pool,), -1, dtype=torch.long)
for i in order.tolist():
    x = quenched[i]
    hit = False
    for c, xc in enumerate(centers):
        if (x - xc).norm() < args.cluster_radius:
            labels[i] = c; hit = True; break
    if not hit:
        labels[i] = len(centers); centers.append(x.clone())
centers_t = torch.stack(centers)
n_wells = len(centers)
Uc = u.eval(centers_t).clone()
phic = u.misfit(centers_t)
prc = 0.5 * (centers_t ** 2).sum(-1)
theta0 = centers_t[:, 0] * std_all[0]
# sign label: projection onto the truth's nonconstant direction
tr_nc = xi_truth.clone(); tr_nc[0] = 0.0
proj = (centers_t * tr_nc.unsqueeze(0)).sum(-1) / (tr_nc.norm() + 1e-9)
qt_counts = torch.bincount(labels, minlength=n_wells)

# ---------------- (2) MALA occupancy ----------------------------------------
chains = 2.5 * torch.randn(512, d, device=device)
x = langevin(chains, u, step=2e-3, iters=4000, adjust=True)
samp = [x]
for _ in range(7):
    x = langevin(x, u, step=2e-3, iters=500, adjust=True)
    samp.append(x)
mala = torch.cat(samp)                                   # 4096 samples
if args.cluster == 'sym':
    # label each MALA sample by the SYMMETRY coordinates: shift index
    # n = round(theta0 * alpha / 2pi) and sign of the projection on the
    # truth's nonconstant direction -- the (n, s) lattice IS the well set.
    th0_m = mala[:, 0] * std_all[0]
    n_lab = torch.round(th0_m * p.ALPHA / (2.0 * math.pi)).long()
    s_lab = ((mala * tr_nc.unsqueeze(0)).sum(-1) > 0).long()
    pair = n_lab * 2 + s_lab
    uniq, inv = pair.unique(return_inverse=True)
    n_wells = len(uniq)
    centers_t = torch.stack([mala[inv == c].mean(0) for c in range(n_wells)])
    Uc = u.eval(centers_t).clone()
    phic = u.misfit(centers_t)
    prc = 0.5 * (centers_t ** 2).sum(-1)
    theta0 = centers_t[:, 0] * std_all[0]
    proj = (centers_t * tr_nc.unsqueeze(0)).sum(-1) / (tr_nc.norm() + 1e-9)
    occ = torch.bincount(inv, minlength=n_wells).float()
    occ = occ / occ.sum()
    qt_counts = (torch.cdist(qt, centers_t).argmin(dim=1)
                 .bincount(minlength=n_wells))
    print('sym labels (n, s):', [( (u.item() - u.item() % 2)//2, u.item() % 2) for u in uniq])
elif args.cluster == 'mala':
    # re-cluster on the POSTERIOR MASS: greedy radius clustering of MALA
    # samples ordered by potential (the quench census above is kept only as
    # the QT-reachability reference for the coverage metric below)
    Um = u.eval(mala).clone()
    order_m = torch.argsort(Um)
    centers, labels_m = [], torch.full((mala.shape[0],), -1, dtype=torch.long)
    for i in order_m.tolist():
        x = mala[i]
        hit = False
        for c, xc in enumerate(centers):
            if (x - xc).norm() < args.cluster_radius:
                labels_m[i] = c; hit = True; break
        if not hit:
            labels_m[i] = len(centers); centers.append(x.clone())
    centers_t = torch.stack(centers)
    n_wells = len(centers)
    Uc = u.eval(centers_t).clone()
    phic = u.misfit(centers_t)
    prc = 0.5 * (centers_t ** 2).sum(-1)
    theta0 = centers_t[:, 0] * std_all[0]
    proj = (centers_t * tr_nc.unsqueeze(0)).sum(-1) / (tr_nc.norm() + 1e-9)
    occ = torch.bincount(labels_m, minlength=n_wells).float()
    occ = occ / occ.sum()
    qt_counts = (torch.cdist(qt, centers_t).argmin(dim=1)
                 .bincount(minlength=n_wells))
else:
    dists = torch.cdist(mala, centers_t)
    mala_lab = dists.argmin(dim=1)
    near = dists.min(dim=1).values < 2.5 * args.cluster_radius
    occ = torch.bincount(mala_lab[near], minlength=n_wells).float()
    occ = occ / occ.sum()

# ---------------- (3) kNN coverage (2D-benchmark convention) ----------------
def knn_coverage(samples, ref, k=5):
    dxx = torch.cdist(ref, ref)
    dxx.fill_diagonal_(float('inf'))
    nnd = dxx.topk(k, dim=1, largest=False).values[:, -1]
    dxy = torch.cdist(ref, samples)
    return (dxy < nnd.unsqueeze(1)).any(dim=1).float().mean().item()

cov_mala_of_qt = knn_coverage(mala, qt)     # MALA reaches QT wells?
cov_qt_of_mala = knn_coverage(qt, mala)     # QT reaches MALA regions?

# ---------------- report -----------------------------------------------------
keep = (occ >= 0.005) if args.cluster in ('mala', 'sym') else (qt_counts >= max(2, int(0.002 * args.n_pool)))
print(f"wells (QT census, radius {args.cluster_radius}): {n_wells} raw, "
      f"{int(keep.sum())} with >=0.2% of pool")
rows = []
for c in range(n_wells):
    if not keep[c]:
        continue
    rows.append((c, qt_counts[c].item(), occ[c].item(), theta0[c].item(),
                 proj[c].item(), Uc[c].item(), phic[c].item(), prc[c].item()))
    print(f"  well {c}: QT {qt_counts[c]:5d}  MALA occ {occ[c]:.3f}  "
          f"theta0 {theta0[c]:+.2f}  proj {proj[c]:+.2f}  U {Uc[c]:8.2f}  "
          f"misfit {phic[c]:8.2f}  prior {prc[c]:6.2f}")
print(f"coverage: MALA covers QT {cov_mala_of_qt:.3f}; QT covers MALA {cov_qt_of_mala:.3f}")

# barrier probe: linear path between the two most-occupied wells
top2 = occ.argsort(descending=True)[:2]
if len(top2) == 2 and occ[top2[1]] > 0:
    a, b = centers_t[top2[0]], centers_t[top2[1]]
    ts = torch.linspace(0, 1, 41, device=device).unsqueeze(1)
    path = (1 - ts) * a.unsqueeze(0) + ts * b.unsqueeze(0)
    Up = u.eval(path).clone()
    barrier = (Up.max() - max(Up[0], Up[-1])).item()
    print(f"barrier (linear path, top-2 wells): {barrier:.2f} kT "
          f"(occ {occ[top2[0]]:.3f} vs {occ[top2[1]]:.3f})")
else:
    barrier = float('nan')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
ax = axes[0]
m_th0 = mala[:, 0] * std_all[0]
m_proj = (mala * tr_nc.unsqueeze(0)).sum(-1) / (tr_nc.norm() + 1e-9)
ax.scatter(m_th0.cpu(), m_proj.cpu(), s=2, alpha=0.2, label='MALA')
ax.scatter(theta0[keep].cpu(), proj[keep].cpu(), s=80, marker='x', c='red', label='wells')
ax.set_xlabel(r'$\theta_0$'); ax.set_ylabel('proj on truth direction'); ax.legend(fontsize=8)
ax.set_title(f'{int(keep.sum())} wells')
ax = axes[1]
idx = [r[0] for r in rows]
ax.bar(range(len(idx)), [r[2] for r in rows], color='tab:purple', alpha=0.7)
ax.set_xticks(range(len(idx)), [str(i) for i in idx])
ax.set_xlabel('well'); ax.set_ylabel('MALA occupancy')
ax = axes[2]
ax.bar(range(len(idx)), [r[6] for r in rows], color='tab:red', alpha=0.7)
ax.set_xticks(range(len(idx)), [str(i) for i in idx])
ax.set_xlabel('well'); ax.set_ylabel('misfit at center')
plt.tight_layout()
plt.savefig(FIG / f'wells_m{args.m_low}.png', dpi=400, bbox_inches='tight',
            pad_inches=0.02)

with open(HERE / 'wells.md', 'a') as f:
    f.write(f"\n## Geometry {args.m_low}x{args.m_low} in {args.m_full}x{args.m_full} "
            f"(alpha={p.ALPHA}, tilt={p.EPS_TILT}, sigma_obs={p.SIGMA_OBS}, "
            f"A={p.PRIOR_AMP}, s={p.PRIOR_S})\n\n"
            f"{int(keep.sum())} wells; coverage MALA-of-QT {cov_mala_of_qt:.3f}, "
            f"QT-of-MALA {cov_qt_of_mala:.3f}\n\n"
            "| well | QT count | MALA occ | theta0 | proj | U | misfit | prior |\n"
            "|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:+.2f} | {r[4]:+.2f} | "
                f"{r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} |\n")
print(f"wrote wells.md, figures/wells_m{args.m_low}.png")
