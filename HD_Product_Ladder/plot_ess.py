"""ESS-trajectory figure for the AIS-ladder sweep, styled like HD_Product/ESS.png
but with one curve per ladder length M (instead of one per loss). Raw per-step ESS
(faint) plus a moving average (bold). Reads data_M{M}.pth for every available M."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

HERE = Path(__file__).resolve().parent
import parameters as P


def moving_average(a, w):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode='valid')


def main(mlist=None):
    mlist = mlist or [1, 2, 4, 8]   # retain only M = 1,2,4,8
    avail = [(M, HERE / f'data_M{M}.pth') for M in mlist if (HERE / f'data_M{M}.pth').exists()]
    if not avail:
        print("no data_M*.pth found"); return

    colors = cm.viridis(np.linspace(0.0, 0.85, len(avail)))
    d = steps = None
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    finals = {}
    for (M, path), col in zip(avail, colors):
        data = torch.load(path, weights_only=False)
        d = data['d']; steps = data['steps']
        # ess_history is a flat list, one direct mu/nu ESS per step (HD_Product style).
        ess = np.asarray(data['ess_history'], dtype=float)
        xs = np.arange(len(ess))
        finals[M] = data['final_ess']
        win = max(1, len(ess) // 50)
        ax.plot(xs, ess, color=col, linewidth=0.25, alpha=0.18)
        ma = moving_average(ess, win)
        ax.plot(xs[win - 1:win - 1 + len(ma)], ma, color=col, linewidth=1.0,
                label=f'$M={M}$')
    ax.set_xlabel('step')
    ax.set_ylabel('ESS')
    ax.set_xlim(0, steps)
    ax.set_ylim(0, None)
    ax.set_title(rf'ESS with AIS ladder length $M$ ($d={d}$)')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    out = HERE / 'ESS_ladder.png'
    plt.savefig(out, dpi=400)
    plt.close(fig)
    print(f"Saved {out}  (d={d}, steps={steps})")
    print("final ESS by M:", {M: round(v, 4) for M, v in finals.items()})


if __name__ == '__main__':
    arg = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else None
    main(arg)
