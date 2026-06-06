# pyright: reportArgumentType=false
"""Figures + tables for the 6x6 phi^4 benchmark.

Reads data.pth (training runs) and phi4_reference.pth (PT referee).
Writes figures/fig_background.png, figures/fig_methods.png, figures/fig_ess.png,
results_table.md, results_table.csv.  No data files are modified.
"""
from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'mathtext.fontset': 'cm', 'font.family': 'serif',
})

from zflows.utils import resample
from core import magnetization
from parameters import L, KAPPA, LAMBDA, H, STEPS

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

METHODS = ('KL', 'KL+X_mu', 'KL+X_mu+X_hat_mu', 'KL+X_mu+X_mix')
LABEL = {
    'KL':                'forward KL',
    'KL+X_mu':           r'forward KL+$\mathrm{X}_\mu$',
    'KL+X_mu+X_hat_mu':  r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$',
    'KL+X_mu+X_mix':     r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$',
}
COLOR = {
    'KL':                "#1F77B4A0",   # tab:blue
    'KL+X_mu':           "#2CA02CA0",   # tab:green
    'KL+X_mu+X_hat_mu':  "#D62728A0",   # tab:red
    'KL+X_mu+X_mix':     "#9467BDA0",   # tab:purple
}

data = torch.load(HERE / 'data.pth', weights_only=False, map_location='cpu')
ref = torch.load(HERE / 'phi4_reference.pth', weights_only=False, map_location='cpu')
runs = data['runs']
have = [m for m in METHODS if m in runs]

m_ref = ref['m_trace'].ravel().numpy()
p_plus_ref = float((m_ref > 0).mean())
dF_ref = float(np.log(max(p_plus_ref, 1e-9) / max(1 - p_plus_ref, 1e-9)))

BINS, LIM = 81, 1.6
def hist(m):
    h, e = np.histogram(m, bins=BINS, range=(-LIM, LIM), density=True)
    return h, 0.5 * (e[:-1] + e[1:])

def whist(m, w):
    h, e = np.histogram(m, bins=BINS, range=(-LIM, LIM), density=True, weights=w)
    return h, 0.5 * (e[:-1] + e[1:])

# ---------------- fig_background: potential, reference p(m), vacua heatmaps
fig = plt.figure(figsize=(10, 3.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1, 1])
ax = fig.add_subplot(gs[0])
ph = np.linspace(-1.8, 1.8, 300)
ax.plot(ph, ph**2 + LAMBDA * (ph**2 - 1)**2, color='black', label='single site')
ax.plot(ph, ph**2 + LAMBDA * (ph**2 - 1)**2 - 4 * KAPPA * ph**2 + H * ph,
        color='tab:red', ls='--', label='with aligned neighbors')
ax.set_xlabel(r'$\varphi_x$'); ax.set_ylabel('site energy')
ax.legend(); ax.set_title('(a) per-site double well')
ax = fig.add_subplot(gs[1])
h0, c0 = hist(m_ref)
ax.semilogy(c0, h0 + 1e-12, color='black')
ax.set_xlabel(r'magnetization $m$'); ax.set_ylabel(r'$p(m)$')
ax.set_title(f'(b) PT reference, $\\Delta F={dF_ref:.2f}$')
samp = ref['samples_t1']
msamp = samp.mean(dim=1)
for j, (sel, tag) in enumerate([(msamp.argmin(), r'$m<0$ vacuum'),
                                (msamp.argmax(), r'$m>0$ vacuum')]):
    ax = fig.add_subplot(gs[2 + j])
    im = ax.imshow(samp[sel].view(L, L).numpy(), cmap='coolwarm', vmin=-1.6, vmax=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(('(c) ' if j == 0 else '(d) ') + tag)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(FIG / 'fig_background.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------- fig_methods: per-method reweighted p(m) vs reference
n = len(have)
fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 2.9), squeeze=False)
rows = []
for j, meth in enumerate(have):
    r = runs[meth]
    mag = magnetization(r['samples']).numpy()
    w = r['weights'].numpy()
    ax = axes[0][j]
    hp, cp = hist(mag)
    ax.semilogy(cp, hp + 1e-12, color='0.6', lw=1.0, label='pushforward')
    hw, cw = whist(mag, w / w.sum())
    ax.semilogy(cw, hw + 1e-12, color=COLOR[meth], lw=1.6, label='reweighted')
    ax.semilogy(c0, h0 + 1e-12, color='black', ls=':', lw=1.2, label='PT reference')
    wn = w / w.sum()
    p_plus = float(wn[mag > 0].sum())
    dF = float(np.log(max(p_plus, 1e-9) / max(1 - p_plus, 1e-9)))
    ax.set_title(f"{LABEL[meth]}\nESS={r['final_ess']:.2f}, $p_+$={p_plus:.2f}")
    ax.set_xlabel(r'$m$')
    ax.set_ylim(1e-4, 30)
    if j == 0:
        ax.set_ylabel(r'$p(m)$'); ax.legend(loc='lower center', fontsize=6.5)
    rows.append(dict(method=meth, final_ess=round(r['final_ess'], 4),
                     p_plus=round(p_plus, 4), delta_F=round(dF, 3)))
plt.tight_layout()
plt.savefig(FIG / 'fig_methods.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------- fig_ess
fig, ax = plt.subplots(figsize=(5, 3.4))
for meth in have:
    ax.plot(runs[meth]['ess_history'], color=COLOR[meth], lw=0.7, label=LABEL[meth])
ax.set_xlabel('step'); ax.set_ylabel('training ESS')
ax.set_xlim(0, STEPS); ax.set_ylim(0, 1)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(FIG / 'fig_ess.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------- tables
with open(HERE / 'results_table.csv', 'w', newline='') as f:
    wcsv = csv.DictWriter(f, fieldnames=['method', 'final_ess', 'p_plus', 'delta_F'])
    wcsv.writeheader(); wcsv.writerows(rows)
with open(HERE / 'results_table.md', 'w') as f:
    f.write(f"# phi^4 6x6 results (kappa={KAPPA}, lambda={LAMBDA}, h={H})\n\n")
    f.write(f"PT reference: p(m>0) = {p_plus_ref:.3f}, Delta F = {dF_ref:.2f} kT, "
            f"barrier = {ref['barrier']:.2f} kT, roundtrips = {ref['roundtrips']}\n\n")
    f.write("| method | final ESS | p(m>0) reweighted | Delta F (kT) |\n|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['method']} | {r['final_ess']:.3f} | {r['p_plus']:.3f} | {r['delta_F']:.2f} |\n")
    f.write(f"| PT reference | --- | {p_plus_ref:.3f} | {dF_ref:.2f} |\n")
print("wrote figures/fig_background.png, fig_methods.png, fig_ess.png, results_table.md/.csv")
for r in rows:
    print(r)
