# pyright: reportArgumentType=false
"""ESS-ladder figure for the paper's Poisson subsection (replaces tab: poisson).

Per sigma_obs, one panel: the temperature axis 0 -> 1 with each accepted stage
drawn as an arc -- forward KL above (blue), the X-regularized loss below (red) --
the per-stage FINE validation ESS labelled on each arc apex, a cross at the
final bridge the forward KL run failed. Each panel title carries the staged-
sampler TV to the PT referee. Style ported from zflows-md/make_summary.py.

    ~/.envs/torch/bin/python plot_ladder.py   ->  figures/poisson_ladders.png
"""
import csv
import math
import statistics
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'mathtext.fontset': 'cm', 'font.family': 'serif'})
HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
C_KL, C_X = "#1F77B4", "#D62728"          # forward KL (blue, top); KL+X (red, bottom)
SIGMAS = ['0.01', '0.015', '0.02', '0.025']

# staged-sampler TV (canonical), mean over replays
tv = {}
rows = {}
for r in csv.DictReader(open(HERE / 'staged_tv.csv')):
    rows.setdefault(r['file'], []).append(float(r['tv']))
for f, vs in rows.items():
    tv[f] = statistics.mean(vs)


def stages_of(tag):
    f = HERE / f'data_{tag}.pth'
    if not f.exists():
        return None, True
    D = torch.load(f, weights_only=False, map_location='cpu')
    return [(s['t'], s['fine_ess']) for s in D['stages']], D['complete']


def draw(ax, st, complete, color, sign):
    if not st:
        return
    ts = [0.0] + [t for t, _ in st]
    for i, (t1, ess) in enumerate(st):
        t0 = ts[i]
        mid, r = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
        th = np.linspace(0, np.pi, 60)
        h = 0.18 + 1.0 * r
        ax.plot(mid + r * np.cos(th), sign * h * np.sin(th), color=color, lw=1.8, zorder=3)
        lift = 0.04 + (0.10 if r < 0.06 and i % 2 == 1 else 0.0)  # destagger tight arcs
        ax.text(mid, sign * (h + lift), f"{ess:.2f}", ha='center',
                va='bottom' if sign > 0 else 'top', color=color, fontsize=9, zorder=4)
    ax.plot(ts, [0] * len(ts), 'o', color='0.2', ms=3, zorder=5)
    if not complete:                       # cross at the unreached final bridge
        ax.plot(1.0, 0.0, 'x', color=color, ms=8, mew=2, zorder=6)


fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.6))
for ax, sig in zip(axes.ravel(), SIGMAS):
    sk, ck = stages_of(f'kl_o{sig}')
    sb, cb = stages_of(f'balance_o{sig}')
    ax.axhline(0, color='0.3', lw=1.0, zorder=1)
    draw(ax, sk, ck, C_KL, +1)
    draw(ax, sb, cb, C_X, -1)
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([])
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(r'$t$ (temperature)')
    ax.set_title(rf'$\sigma_{{\mathrm{{obs}}}} = {sig}$', fontsize=11)

# shared colour key as a clean figure legend (no axis-tick collision)
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=C_KL, lw=2, label='forward KL'),
           Line2D([0], [0], color=C_X, lw=2, label=r'KL$+\mathrm{X}_\mu+\mathrm{X}_{\mathrm{mix}}$')]
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / 'poisson_ladders.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
print('wrote figures/poisson_ladders.png')
