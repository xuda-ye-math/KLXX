# pyright: reportArgumentType=false
"""Two-method comparison figures (balance vs kl) for one lattice size:
training-ESS history side by side, sector occupancy side by side, ladder
overlay. Reads data_L{L}_balance.pth and data_L{L}_kl.pth.

Usage: python plot_comparison.py 4   (or 8)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {'balance': 'tab:blue', 'kl': 'tab:orange'}


def load(L, method):
    p = HERE / f'data_L{L}_{method}.pth'
    if not p.exists():
        return None
    return torch.load(p, weights_only=False, map_location='cpu')


def main(L: int):
    runs = {m: load(L, m) for m in ('balance', 'kl')}
    runs = {m: d for m, d in runs.items() if d is not None}
    if len(runs) < 2:
        print(f'need both data_L{L}_balance.pth and data_L{L}_kl.pth '
              f'(found: {list(runs)})')
        return
    P = next(iter(runs.values()))['config']['P']
    figdir = HERE / 'figures'
    figdir.mkdir(exist_ok=True)

    # ---- training ESS history, stacked panels ----
    fig, axes = plt.subplots(len(runs), 1, figsize=(7, 2.6 * len(runs)),
                             sharex=False)
    for ax, (m, d) in zip(np.atleast_1d(axes), runs.items()):
        off = 0
        for i, s in enumerate(d['stages']):
            h = np.asarray(s['train_ess_hist'], dtype=float)
            xs = np.arange(off, off + len(h))
            ax.plot(xs, h, lw=0.6, color=COLORS[m])
            ax.axvline(off, color='gray', lw=0.4, ls=':')
            ax.text(off + len(h) / 2, 1.04, f"$t_k$={s['t']:.2f}",
                    ha='center', fontsize=6)
            off += len(h)
        flag = '' if d['complete'] else '  [INCOMPLETE]'
        ax.set_ylabel('per-step ESS')
        ax.set_ylim(0, 1.12)
        ax.set_title(f'{m}{flag}', fontsize=9, loc='left')
    np.atleast_1d(axes)[-1].set_xlabel('gradient step (stages concatenated)')
    fig.suptitle(f'training ESS history, L={L} (balance vs kl)', fontsize=10)
    plt.tight_layout()
    plt.savefig(figdir / f'cmp_ess_steps_L{L}.png', dpi=300); plt.close(fig)

    # ---- sector occupancy side by side ----
    x = np.arange(P); w = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for j, (m, d) in enumerate(runs.items()):
        cp = np.asarray(d['counts_push'], dtype=float)
        flag = '' if d['complete'] else ' [INCOMPLETE]'
        ax.bar(x + (j - (len(runs) - 1) / 2) * w, cp / cp.sum(), w,
               label=f'{m}{flag}', color=COLORS[m])
    ax.axhline(1.0 / P, ls='--', lw=0.8, color='gray')
    ax.set_xlabel('clock sector $k$'); ax.set_ylabel('occupancy (pushforward)')
    ax.set_title(f'sector occupancy, $P={P}$, L={L}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / f'cmp_sectors_L{L}.png', dpi=300); plt.close(fig)

    # ---- ladder + validation ESS overlay ----
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for m, d in runs.items():
        ts = [0.0] + list(d['ladder'])
        val = [s['val_ess'] for s in d['stages']]
        axs[0].step(range(len(ts)), ts, where='post', marker='o',
                    label=m, color=COLORS[m])
        axs[1].plot(range(1, len(val) + 1), val, marker='s',
                    label=m, color=COLORS[m])
    axs[0].set_xlabel('stage $k$'); axs[0].set_ylabel('$t_k$')
    axs[0].set_title(f'adaptive ladder, L={L}'); axs[0].set_ylim(0, 1.05)
    axs[0].legend(fontsize=8)
    axs[1].axhline(0.3, ls='--', lw=0.8, color='gray')
    axs[1].set_xlabel('stage $k$'); axs[1].set_ylabel('validation ESS')
    axs[1].set_title('acceptance gate (floor 0.3)'); axs[1].set_ylim(0, 1.05)
    axs[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / f'cmp_ladder_L{L}.png', dpi=300); plt.close(fig)
    print(f'figures written to {figdir}/cmp_(ess_steps|sectors|ladder)_L{L}.png')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
