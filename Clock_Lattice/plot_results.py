# pyright: reportArgumentType=false
"""Figures for a Clock_Lattice run: ladder staircase + validation ESS,
sector occupancy bar chart, magnetization scatter. Reads data_<tag>.pth."""
import sys
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main(tag: str):
    d = torch.load(HERE / f"data_{tag}.pth", weights_only=False,
                   map_location='cpu')
    cfg = d['config']
    P = cfg['P']
    figdir = HERE / 'figures'
    figdir.mkdir(exist_ok=True)

    # ---- ladder staircase + per-stage validation ESS ----
    ts = [0.0] + list(d['ladder'])
    val_ess = [s['val_ess'] for s in d['stages']]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].step(range(len(ts)), ts, where='post', marker='o')
    ax[0].set_xlabel('stage $k$'); ax[0].set_ylabel('$t_k$')
    ax[0].set_title(f'adaptive ladder ({tag})'); ax[0].set_ylim(0, 1.05)
    ax[1].plot(range(1, len(val_ess) + 1), val_ess, marker='s', color='tab:red')
    ax[1].axhline(0.3, ls='--', lw=0.8, color='gray')
    ax[1].set_xlabel('stage $k$'); ax[1].set_ylabel('validation ESS')
    ax[1].set_title('acceptance gate (floor 0.3)'); ax[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figdir / f'ladder_{tag}.png', dpi=300); plt.close(fig)

    # ---- sector occupancy ----
    cp = np.asarray(d['counts_push'], dtype=float)
    cv = np.asarray(d['counts_valid'], dtype=float)
    x = np.arange(P); w = 0.4
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - w / 2, cp / cp.sum(), w, label='pushforward')
    ax.bar(x + w / 2, cv / cv.sum(), w, label='validation set')
    ax.axhline(1.0 / P, ls='--', lw=0.8, color='gray')
    ax.set_xlabel('clock sector $k$'); ax.set_ylabel('occupancy')
    ax.set_title(f'sector occupancy, $P={P}$ ({tag})')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / f'sectors_{tag}.png', dpi=300); plt.close(fig)

    # ---- per-step training ESS across stages (full recorded history) ----
    fig, ax = plt.subplots(figsize=(6, 3))
    off = 0
    for i, s in enumerate(d['stages']):
        h = np.asarray(s['train_ess_hist'], dtype=float)
        xs = np.arange(off, off + len(h))
        ax.plot(xs, h, lw=0.6, label=f"stage {i+1} ($t_k$={s['t']:.3f})")
        ax.axvline(off, color='gray', lw=0.4, ls=':')
        off += len(h)
    ax.set_xlabel('gradient step (stages concatenated)')
    ax.set_ylabel('per-step ESS')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'training ESS history ({tag})')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figdir / f'ess_steps_{tag}.png', dpi=300); plt.close(fig)

    # ---- magnetization scatter ----
    y = d['samples_push'][:5000]
    m = torch.exp(1j * y.to(torch.float32)).mean(dim=1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(m.real, m.imag, s=2, alpha=0.3)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), lw=0.5, color='gray')
    for k in range(P):
        a = 2 * np.pi * k / P
        ax.plot([0, np.cos(a)], [0, np.sin(a)], lw=0.5, ls=':', color='gray')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r'$\mathrm{Re}\,m$'); ax.set_ylabel(r'$\mathrm{Im}\,m$')
    ax.set_title(f'magnetization ({tag})')
    plt.tight_layout()
    plt.savefig(figdir / f'magnetization_{tag}.png', dpi=300); plt.close(fig)
    print(f"figures written to {figdir}/(ladder|sectors|magnetization)_{tag}.png")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'L4')
