"""Figures for the Sensor_Array experiment.

  ESS.png      -- training-target ESS trajectory (faint raw + bold moving average);
                  the fake-ESS pitfall: bare KL / KL+X_mu ride high while covering
                  only 1-2 of the 6 modes.
  samples.png  -- 1x4 panels, one per loss, scatter of the pushforward samples in
                  the (theta_1, theta_2) plane with the 6 permutation mode centers
                  marked; shows directly which losses collapse and which recover
                  all 3! = 6 exchangeable modes.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'mathtext.fontset': 'cm', 'font.family': 'serif',
})

from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import importance_weights, resample, langevin

import core
import parameters as P

HERE = Path(__file__).resolve().parent

METHODS = ('KL', 'KL+X_mu', 'KL+X_mu+X_hat_mu', 'KL+X_mu+X_mix')
LABEL = {
    'KL':                'forward KL',
    'KL+X_mu':           r'forward KL+$\mathrm{X}_\mu$',
    'KL+X_mu+X_hat_mu':  r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$',
    'KL+X_mu+X_mix':     r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$',
}
COLOR = {
    'KL':                "#1F77B4A0",
    'KL+X_mu':           "#2CA02CA0",
    'KL+X_mu+X_hat_mu':  "#D62728A0",
    'KL+X_mu+X_mix':     "#9467BDA0",
}


def moving_average(a, w):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode='valid')


def compute_resample(state_dict, u0, u1, n=20000):
    """Rebuild the flow, then run the one-step IS pipeline: pushforward ->
    importance resample -> short Langevin, exactly as in training."""
    flow = NSF(a=[-P.NSF_LIM] * P.N_SRC, b=[P.NSF_LIM] * P.N_SRC,
               bins=P.BINS, transforms=P.TRANSFORMS, hidden_features=P.HIDDEN)
    flow.load_state_dict({k: v for k, v in state_dict.items()})
    with torch.no_grad():
        x = u0.samples(n)
        G = flow.t()
        y, _ = G.inv.call_and_ladj(x)
        w = importance_weights(x, u0, u1, G.inv)
    y_res = resample(y, w)
    y_res = langevin(y_res, u1, step=P.IS_MC_STEP, iters=P.IS_MC_ITERS)
    return y_res.cpu()


def main():
    data = torch.load(HERE / 'data.pth', weights_only=False)
    steps = data['config'].get('STEPS', None)
    nmodes = data['nmodes']
    centers = data['centers'].numpy()                    # [6, 3]

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
    ax.set_ylabel('training-target ESS')
    ax.set_xlim(0, steps or longest)
    ax.set_ylim(0, 1)
    ax.set_title(r'training-target ESS ($d=3$, $3!=6$ modes)')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    out1 = HERE / 'ESS.png'
    plt.savefig(out1, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out1}")

    # --- mode-coverage scatter (theta_1, theta_2 projection) ------------------
    lim = 8.0
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True)
    for ax, m in zip(axes, METHODS):
        y = data['runs'][m]['samples'].numpy()           # [N, 3]
        ax.scatter(centers[:, 0], centers[:, 1], s=90, facecolors='none',
                   edgecolors='lightgray', linewidths=1.3, zorder=5)
        ax.scatter(y[:, 0], y[:, 1], s=0.4, alpha=0.12, color=COLOR[m], zorder=10)
        mf = data['runs'][m]['modes_found']
        ess = data['runs'][m]['final_ess']
        ax.set_title(f"{LABEL[m]}\nESS = $\\mathbf{{{ess:.2f}}}$, "
                     f"modes = $\\mathbf{{{mf}}}/{nmodes}$")
        ax.set_xlabel(r'$\theta_1$')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
    axes[0].set_ylabel(r'$\theta_2$')
    plt.tight_layout()
    out2 = HERE / 'samples.png'
    plt.savefig(out2, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out2}")

    # --- resample-pipeline scatter --------------------------------------------
    if 'state_dict' not in data['runs'][METHODS[0]]:
        print("No state_dict in data.pth; skipping resample.png (re-run train.py).")
        return
    sensors = core.sensor_positions(P.N_SENSORS, P.SENSOR_LIM)
    obs = core.make_data(P.THETA_STAR, sensors, P.ELL, P.SIGMA_OBS, seed=P.DATA_SEED)
    u0 = Gaussian(mean=[0.0] * P.N_SRC, variance=[P.SIGMA_PRIOR ** 2] * P.N_SRC)
    u1 = core.SensorArrayPosterior(sensors, obs, P.ELL, P.SIGMA_OBS, P.SIGMA_PRIOR)
    u1.enable_grad(); u1.enable_eval()

    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True)
    for ax, m in zip(axes, METHODS):
        yr = compute_resample(data['runs'][m]['state_dict'], u0, u1).numpy()
        ax.scatter(centers[:, 0], centers[:, 1], s=90, facecolors='none',
                   edgecolors='lightgray', linewidths=1.3, zorder=5)
        ax.scatter(yr[:, 0], yr[:, 1], s=0.4, alpha=0.12, color=COLOR[m], zorder=10)
        ax.set_title(f"{LABEL[m]}\n" + r"resample$(y, w)$")
        ax.set_xlabel(r'$\theta_1$')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
    axes[0].set_ylabel(r'$\theta_2$')
    plt.tight_layout()
    out3 = HERE / 'resample.png'
    plt.savefig(out3, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out3}")


if __name__ == '__main__':
    main()
