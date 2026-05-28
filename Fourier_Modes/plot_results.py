"""ESS-trajectory plot for the Fourier_Modes experiment, paper Figure 3 style:
faint raw per-step training-target ESS + bold per-method moving average. The
bare-KL fake-ESS pitfall (high ESS while covering 1/4 modes) is the headline
visible feature; the mode-coverage numbers themselves live in results_table.md.
"""
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
    'KL':                "#00008BA0",
    'KL+X_mu':           "#006400A0",
    'KL+X_mu+X_hat_mu':  "#8B0000A0",
    'KL+X_mu+X_mix':     "#4B0082A0",
}


def moving_average(a, w):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode='valid')


def main():
    data = torch.load(HERE / 'data.pth', weights_only=False)
    steps = data['config'].get('STEPS', None)

    # --- ESS trajectory -------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.6))
    longest = 0
    for m in METHODS:
        ess = np.asarray(data['runs'][m]['ess_history'], dtype=float)
        longest = max(longest, len(ess))
        x = np.arange(len(ess))
        w = max(1, len(ess) // 50)
        ax.plot(x, ess, color=COLOR[m], linewidth=0.4, alpha=0.25)
        ma = moving_average(ess, w)
        ax.plot(x[w - 1:w - 1 + len(ma)], ma, color=COLOR[m],
                linewidth=1.4, label=LABEL[m])
    ax.set_xlabel('step')
    ax.set_ylabel('training-grid ESS')
    ax.set_xlim(0, steps or longest)
    ax.set_ylim(0, 1)
    ax.set_title(r'training-target ESS ($d=9$)')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    out1 = HERE / 'ESS.png'
    plt.savefig(out1, dpi=400)
    plt.close(fig)
    print(f"Saved {out1}")


if __name__ == '__main__':
    main()
