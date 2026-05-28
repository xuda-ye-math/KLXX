"""ESS-trajectory figure for one k (default k=7, d=128), styled like the paper's
Figure 3 (2D_Chessboard/ESS.png): the four losses' training-ESS curves overlaid.
Raw per-step ESS (faint) plus a moving average (bold) for readability."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

METHODS = ('KL', 'KL+X_mu', 'KL+X_mu+X_hat_mu', 'KL+X_mu+X_mix')
METHOD_LABEL = {
    'KL':                'forward KL',
    'KL+X_mu':           r'forward KL+$\mathrm{X}_\mu$',
    'KL+X_mu+X_hat_mu':  r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$',
    'KL+X_mu+X_mix':     r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$',
}
METHOD_COLOR = {
    'KL':                "#00008BC4",
    'KL+X_mu':           "#006400C4",
    'KL+X_mu+X_hat_mu':  "#8B0000C4",
    'KL+X_mu+X_mix':     "#4B0082C4",
}


def moving_average(a, w):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode='valid')


def main(k=7):
    data = torch.load(HERE / f'data_k{k}.pth', weights_only=False)
    d = data['d']
    steps = data['steps']
    win = max(1, steps // 50)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.6))
    for m in METHODS:
        ess = np.asarray(data['runs'][m]['ess_history'], dtype=float)
        x = np.arange(len(ess))
        ax.plot(x, ess, color=METHOD_COLOR[m], linewidth=0.4, alpha=0.25)
        ma = moving_average(ess, win)
        ax.plot(x[win - 1:win - 1 + len(ma)], ma, color=METHOD_COLOR[m],
                linewidth=1.4, label=METHOD_LABEL[m])
    ax.set_xlabel('step')
    ax.set_ylabel('ESS')
    ax.set_xlim(0, steps)
    ax.set_ylim(0, 1)
    ax.set_title(f'$d = {d}$')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    out = HERE / f'ESS_k{k}.png'
    plt.savefig(out, dpi=400)
    plt.close(fig)
    print(f"Saved {out}  (d={d}, steps={steps}, ma window={win})")
    for m in METHODS:
        print(f"  {m:<18} final_ess={data['runs'][m]['final_ess']:.4f}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
