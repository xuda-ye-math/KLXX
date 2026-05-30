"""ESS trajectory + mode-occupancy bar plots for Wave_2D. Renders ESS.png and modes.png."""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

METHODS = ('KL', 'KL+X_mu', 'KL+X_mu+X_hat_mu', 'KL+X_mu+X_mix')
LABEL = {
    'KL':                'forward KL',
    'KL+X_mu':           r'forward KL+$\mathrm{X}_\mu$',
    'KL+X_mu+X_hat_mu':  r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$',
    'KL+X_mu+X_mix':     r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$',
}
COLOR = {
    'KL':                "#00008B",
    'KL+X_mu':           "#006400",
    'KL+X_mu+X_hat_mu':  "#8B0000",
    'KL+X_mu+X_mix':     "#4B0082",
}


def moving_average(a, w):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode='valid')


def main():
    d = torch.load(HERE / 'data.pth', weights_only=False)
    cfg = d['config']
    steps = cfg.get('STEPS', None)

    # ESS trajectory
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6))
    for m in METHODS:
        ess = np.asarray(d['runs'][m]['ess_history'], dtype=float)
        x = np.arange(len(ess))
        w = max(1, len(ess) // 50)
        ax.plot(x, ess, color=COLOR[m], linewidth=0.4, alpha=0.25)
        ma = moving_average(ess, w)
        ax.plot(x[w - 1:w - 1 + len(ma)], ma, color=COLOR[m],
                linewidth=1.5, label=LABEL[m])
    ax.set_xlabel('step')
    ax.set_ylabel('training-target ESS')
    ax.set_xlim(0, steps or x[-1])
    ax.set_ylim(0, 1)
    ax.set_title('Darcy_2D — training ESS trajectory')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / 'ESS.png', dpi=400)
    plt.close(fig)

    # Mode occupancy bar chart (6 modes, 4 methods grouped)
    NMODES = 6
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6))
    bar_w = 0.18
    xs = np.arange(NMODES)
    for j, m in enumerate(METHODS):
        occ = np.asarray(d['runs'][m]['occupancy'], dtype=float)
        ax.bar(xs + (j - 1.5) * bar_w, occ, width=bar_w, color=COLOR[m], label=LABEL[m])
    ax.axhline(1.0 / NMODES, color='k', linestyle='--', linewidth=0.7, label='uniform 1/6')
    ax.set_xticks(xs)
    ax.set_xticklabels([r'$+\theta_*,\,k{=}{-}1$', r'$+\theta_*,\,k{=}0$', r'$+\theta_*,\,k{=}{+}1$',
                        r'$-\theta_*,\,k{=}{-}1$', r'$-\theta_*,\,k{=}0$', r'$-\theta_*,\,k{=}{+}1$'],
                       fontsize=8, rotation=20)
    ax.set_ylabel('posterior-sample share')
    ax.set_title('Darcy_2D — mode occupancy by method')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(HERE / 'modes.png', dpi=400)
    plt.close(fig)

    print(f"Saved ESS.png and modes.png in {HERE}")


if __name__ == '__main__':
    main()
