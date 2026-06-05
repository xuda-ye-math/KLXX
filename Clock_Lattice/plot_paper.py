# pyright: reportArgumentType=false
"""Publication figures for the L=8 (D=64) clock-model subsection (sweep story).

fig_clock_target.pdf   — structure of the cold target, displayed on the
                         validation set of the X-regularized B=10k run:
                         (a) magnetization-plane scatter colored by sector,
                         (b) per-site angle marginal (validation vs pushforward),
                         (c) sector occupancy with +-2sigma whiskers.
fig_clock_training.pdf — the sweep comparison, B=1000 and B=10000 pairs:
                         (a) acceptance timeline at B=10k (both methods,
                             same ladder, same rejections, both reach t=1),
                         (b,c) accepted-rung validation ESS by stage,
                             balance vs bare KL, at B=1k and B=10k.

Data: data_L8_{balance,kl}_B{1k,10k}.pth (release data-L8-sweep).
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'mathtext.fontset': 'cm', 'font.family': 'serif',
})

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / 'figures'
P = 6
COL = {'balance': 'tab:red', 'kl': 'tab:blue'}
LBL = {'balance': r'KL$+$X$_{\mu}+$X$_{(\hat{\mu}+\bar{\nu})/2}$', 'kl': 'forward KL'}

D = {}
for meth in ['balance', 'kl']:
    for b in ['B1k', 'B10k', 'B100k']:
        D[meth, b] = torch.load(HERE / f'data_L8_{meth}_{b}.pth',
                                weights_only=False, map_location='cpu')

# ---------------------------------------------------------------- figure 1
bal = D['balance', 'B10k']
REB = torch.load(HERE / 'rebuild_L8_balance_B10k_N1000000.pth',
                 weights_only=False, map_location='cpu')
y = REB['y'].to(torch.float32)          # fresh staged-rebuild samples (N=1e6)
m = torch.exp(1j * y).mean(dim=1)
sector = torch.remainder(torch.round(torch.angle(m) * P / (2 * np.pi)),
                         P).to(torch.long)
colors = plt.get_cmap('tab10')(np.arange(P))
print(f"B10k balance: tv_valid={bal['tv_valid']:.4f} tv_push={bal['tv_push']:.4f} "
      f"abs_m_valid={bal['abs_m_valid']:.4f} abs_m_push={bal['abs_m_push']:.4f} "
      f"N_valid={y.shape[0]} N_push={bal['samples_push'].shape[0]}")

fig = plt.figure(figsize=(10.0, 3.1))
gs = fig.add_gridspec(1, 3)
axA = fig.add_subplot(gs[0])
gsb = gs[1].subgridspec(2, 1, hspace=0.0)   # (b): two glued histograms
axB1 = fig.add_subplot(gsb[0])
axB2 = fig.add_subplot(gsb[1], sharex=axB1)
axC = fig.add_subplot(gs[2])

# (a) magnetization plane
for k in range(P):
    sel = (sector == k).nonzero(as_tuple=True)[0][:1500]
    axA.scatter(m.real[sel], m.imag[sel], s=2, alpha=0.18,
                color=colors[k], rasterized=True)
th = np.linspace(0, 2 * np.pi, 256)
axA.plot(np.cos(th), np.sin(th), lw=0.6, color='gray')
for k in range(P):
    a = 2 * np.pi * k / P
    axA.plot([0, 1.05 * np.cos(a)], [0, 1.05 * np.sin(a)], lw=0.5, ls=':',
             color='gray')
    axA.annotate(f'$s={k}$', (0.78 * np.cos(a), 0.78 * np.sin(a)),
                 fontsize=7.5, ha='center', va='center', color='black',
                 path_effects=[pe.withStroke(linewidth=1.6, foreground='white')])
axA.set_xlim(-1.12, 1.12); axA.set_ylim(-1.12, 1.12)
axA.set_aspect('equal')
axA.set_xlabel(r'$\mathrm{Re}\, m(\theta)$')
axA.set_ylabel(r'$\mathrm{Im}\, m(\theta)$')
axA.text(0.02, 0.02, r'each point: one $\theta \in [-\pi,\pi)^{64}$',
         transform=axA.transAxes, fontsize=7)
axA.set_title(r'(a) magnetization plane')

# (b) per-site angle marginal: reweighted (top) / pushforward (bottom)
from matplotlib.ticker import MaxNLocator
shift = lambda a: np.remainder(a + np.pi / 6, 2 * np.pi) - np.pi / 6
yp_sites = bal['samples_push'].to(torch.float32).flatten().numpy()
edges = np.linspace(-np.pi / 6, 2 * np.pi - np.pi / 6, 181)  # 30 bins per block
for a, dat, alf in ((axB1, y.flatten().numpy(), None),       # colors as in (c)
                    (axB2, yp_sites, 0.45)):
    h, _ = np.histogram(shift(dat), bins=edges, density=True)
    for k in range(P):
        a.stairs(h[30 * k:30 * (k + 1)], edges[30 * k:30 * (k + 1) + 1],
                 fill=True, color=colors[k], alpha=alf)
axB1.text(0.98, 0.90, 'reweighted', transform=axB1.transAxes,
          fontsize=8, ha='right', va='top')
axB2.text(0.98, 0.90, 'pushforward', transform=axB2.transAxes,
          fontsize=8, ha='right', va='top')
axB1.set_xlim(-np.pi / 6, 2 * np.pi - np.pi / 6)
ytop = max(axB1.get_ylim()[1], axB2.get_ylim()[1])
for a in (axB1, axB2):
    a.set_ylim(0, ytop)
    a.set_ylabel('density')
    for k in range(5):                # six blocks, one per clock angle
        a.axvline(np.pi / 6 + k * np.pi / 3, ls='--', lw=0.6, color='gray')
    a.axhline(1 / (2 * np.pi), ls='--', lw=0.8, color='black')  # uniform density
secx = axB1.secondary_xaxis('top')    # block labels s=0..5 along the top
secx.set_xticks([k * np.pi / 3 for k in range(6)])
secx.set_xticklabels([f'$s={k}$' for k in range(6)])
secx.tick_params(length=0, pad=2, labelsize=7.5)
axB1.tick_params(labelbottom=False)            # glued: labels only below
axB1.yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
axB2.yaxis.set_major_locator(MaxNLocator(4, prune='upper'))
axB2.set_xticks([k * np.pi / 3 for k in range(6)])
axB2.set_xticklabels([r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$',
                      r'$4\pi/3$', r'$5\pi/3$'])
axB2.set_xlabel(r'site angle $\theta_j$')
axB1.set_title(r'(b) site angle marginal', pad=14)

# (c) sector occupancy
from matplotlib.patches import Patch
sec_reb = torch.remainder(torch.round(torch.angle(m) * P / (2 * np.pi)),
                          P).to(torch.long)
cv = torch.bincount(sec_reb, minlength=P).double().numpy(); cv /= cv.sum()
cp = np.asarray(bal['counts_push'], dtype=float); cp /= cp.sum()
x = np.arange(P); w = 0.4
axC.bar(x - w / 2, cv, w, color=[c for c in colors])
axC.bar(x + w / 2, cp, w, color=[c for c in colors], alpha=0.45)
for xi, v in zip(x, cv):
    axC.text(xi - w / 2, v + 0.005, f'{v:.3f}', ha='center', fontsize=6.5,
             path_effects=[pe.withStroke(linewidth=1.6, foreground='white')])
axC.axhline(1 / 6, ls='--', lw=0.8, color='black')     # uniform occupancy
axC.set_xlabel(r'sector $s$')
axC.set_ylabel('occupancy')
axC.set_ylim(0, max(0.30, cv.max() + 0.05))
axC.set_title('(c) sector occupancy')
axC.legend(handles=[Patch(facecolor='0.35', label='reweighted'),
                    Patch(facecolor='0.35', alpha=0.45,
                          label='pushforward')],
           loc='upper right')

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_clock_target.png', dpi=400, bbox_inches='tight',
            pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# per-step ESS curves of the FIRST THREE stages, one panel per stage;
# rows = batch sizes; total width matches fig_clock_training
fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.4), sharey=True)
for row, (b, bb) in enumerate([('B1k', '10^3'), ('B10k', '10^4'),
                               ('B100k', '10^5')]):
    for col in range(3):
        ax_ = axes[row, col]
        st_b = D['balance', b]['stages'][col]
        for meth in ['kl', 'balance']:
            h = np.asarray(D[meth, b]['stages'][col]['train_ess_hist'],
                           dtype=float)
            lw_row = [0.55, 0.8, 1.3][row]      # larger B: smoother curve
            ax_.plot(np.arange(len(h)), h, lw=lw_row, color=COL[meth],
                     alpha=0.85, label=LBL[meth], rasterized=True)
        ax_.set_xlim(0, len(h))
        ax_.set_ylim(0, 1.0)
        ax_.grid(True, color='0.85', lw=0.5, ls='--', zorder=0)
        ax_.set_axisbelow(True)
        ax_.set_title(f"stage {col+1} ($t_{{{col+1}}}={st_b['t']:.2f}$), $B={bb}$",
                      fontsize=8.5, pad=6)
        if col == 0:
            ax_.set_ylabel('per-step ESS')
        if row == 2:
            ax_.set_xlabel('gradient step')
for row in range(3):
    axes[row, 0].legend(loc='upper left', fontsize=7)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig_clock_esscurves.png', dpi=400, bbox_inches='tight',
            pad_inches=0.02)
plt.close(fig)
print('wrote', FIGDIR / 'fig_clock_target.png')
print('wrote', FIGDIR / 'fig_clock_esscurves.png')
