# pyright: reportArgumentType=false
"""Publication figures for the L=8 (D=64) clock-model subsection of the paper.

fig_clock_target.pdf   — structure/difficulty of the cold target (validation
                         set of the X-regularized run, val ESS 0.67):
                         (a) magnetization-plane scatter colored by sector,
                         (b) per-site angle marginal with the 6 clock angles,
                         (c) sector occupancy vs the uniform 1/6 line.
fig_clock_training.pdf — (a) per-attempt timeline of the acceptance loop
                         (every attempt = one full 1000-step training),
                         (b) in-training direct ESS, balance vs forward KL.

Sources: data_L8_balance.pth, data_L8_kl.pth; the 12 stage-4 forward-KL
attempts are not stored in the .pth (the stage was never accepted) and are
transcribed verbatim from train_status.log (run 2026-06-04 00:40-04:21).
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'mathtext.fontset': 'cm', 'font.family': 'serif',
})

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / 'figures'
P = 6

bal = torch.load(HERE / 'data_L8_balance.pth', weights_only=False,
                 map_location='cpu')
kl = torch.load(HERE / 'data_L8_kl.pth', weights_only=False,
                map_location='cpu')

# forward-KL stage-4 attempts (t_k, val ESS) — verbatim from train_status.log;
# the stage was never accepted, so these are absent from data_L8_kl.pth.
KL_S4_T = [0.7421, 0.6927, 0.6581, 0.6338, 0.6169, 0.6050,
           0.5967, 0.5909, 0.5868, 0.5840, 0.5820, 0.5806]
KL_S4_ESS = [0.112, 0.105, 0.103, 0.097, 0.091, 0.090,
             0.099, 0.096, 0.061, 0.104, 0.097, 0.084]

# ---------------------------------------------------------------- figure 1
y = bal['samples_valid'].to(torch.float32)          # validation set ~ mu
m = torch.exp(1j * y).mean(dim=1)
sector = torch.remainder(torch.round(torch.angle(m) * P / (2 * np.pi)),
                         P).to(torch.long)
colors = plt.get_cmap('tab10')(np.arange(P))

fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.1))

# (a) magnetization plane
for k in range(P):
    sel = sector == k
    ax[0].scatter(m.real[sel][:1500], m.imag[sel][:1500], s=2, alpha=0.35,
                  color=colors[k], rasterized=True)
th = np.linspace(0, 2 * np.pi, 256)
ax[0].plot(np.cos(th), np.sin(th), lw=0.6, color='gray')
for k in range(P):
    a = 2 * np.pi * k / P
    ax[0].plot([0, 1.05 * np.cos(a)], [0, 1.05 * np.sin(a)], lw=0.5, ls=':',
               color='gray')
    ax[0].annotate(f'$s={k}$', (0.78 * np.cos(a), 0.78 * np.sin(a)),
                   fontsize=7, ha='center', va='center', color='black')
ax[0].set_xlim(-1.12, 1.12); ax[0].set_ylim(-1.12, 1.12)
ax[0].set_aspect('equal')
ax[0].set_xlabel(r'$\mathrm{Re}\, m(\theta)$')
ax[0].set_ylabel(r'$\mathrm{Im}\, m(\theta)$')
ax[0].text(0.02, 0.02, r'each point: one $\theta \in [-\pi,\pi)^{64}$',
           transform=ax[0].transAxes, fontsize=7)
ax[0].set_title(r'(a) magnetization plane, $p=6$ sectors')

# (b) per-site angle marginal: validation set (fill) vs iid pushforward (line)
ax[1].hist(y.flatten().numpy(), bins=181, range=(-np.pi, np.pi),
           density=True, histtype='stepfilled', color='tab:blue', alpha=0.45,
           edgecolor='tab:blue', lw=0.8, label='validation set')
yp_sites = bal['samples_push'].to(torch.float32).flatten().numpy()
ax[1].hist(yp_sites, bins=181, range=(-np.pi, np.pi), density=True,
           histtype='step', color='0.2', lw=0.9, label='pushforward')
for k in range(-P // 2, P // 2 + 1):
    ax[1].axvline(2 * np.pi * k / P, ls='--', lw=0.6, color='gray')
ax[1].axhline(1 / (2 * np.pi), ls=':', lw=0.9, color='black',
              label=r'source $\mathrm{Unif}[-\pi,\pi)$')
ax[1].set_xlim(-np.pi, np.pi)
ax[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax[1].set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax[1].set_xlabel(r'site angle $\theta_j$')
ax[1].set_ylabel('density histogram')
ax[1].set_title(r'(b) site marginal: locking at $2\pi k/6$')
ax[1].legend(loc='lower right', framealpha=0.9)

# (c) sector occupancy
cv = np.asarray(bal['counts_valid'], dtype=float); cv /= cv.sum()
cp = np.asarray(bal['counts_push'], dtype=float); cp /= cp.sum()
x = np.arange(P); w = 0.4
from matplotlib.patches import Patch
ax[2].bar(x - w / 2, cv, w, color=[c for c in colors])
ax[2].bar(x + w / 2, cp, w, color=[c for c in colors], alpha=0.45)
# +-2 sigma multinomial whiskers on the iid pushforward bars: the sampling
# error is tiny, so the pushforward's deviations from 1/6 are genuine flow
# bias, while the larger validation-set imbalance is unrestored resampling
# fluctuation (see the closing paragraph of Section 5.4).
n_push = np.asarray(bal['counts_push'], dtype=float).sum()
sig = np.sqrt(cp * (1.0 - cp) / n_push)
ax[2].errorbar(x + w / 2, cp, yerr=2.0 * sig, fmt='none', ecolor='black',
               elinewidth=0.9, capsize=2.5, capthick=0.9)
for xi, v in zip(x, cv):
    ax[2].text(xi - w / 2, v + 0.005, f'{v:.3f}', ha='center',
               fontsize=6.5)
ax[2].axhline(1.0 / P, ls='--', lw=0.8, color='black')
ax[2].text(P - 0.45, 1.0 / P + 0.004, r'$1/6$', fontsize=8)
ax[2].set_xlabel(r'sector $s$')
ax[2].set_ylabel('occupancy')
ax[2].set_ylim(0, 0.30)
ax[2].set_title('(c) sector occupancy (6/6 covered)')
ax[2].legend(handles=[Patch(facecolor='0.35', label='validation set'),
                      Patch(facecolor='0.35', alpha=0.45,
                            label=r'pushforward ($\pm 2\sigma$)')],
             loc='upper right')

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_clock_target.pdf', bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig = plt.figure(figsize=(10.0, 3.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], hspace=0.12,
                      wspace=0.22)
axL = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axK = fig.add_subplot(gs[1, 1], sharex=axB, sharey=axB)

# (a) acceptance-loop timeline: every marker = one full training of a stage
def attempts_series(d):
    out = []
    for s in d['stages']:
        for a in s['attempts']:
            out.append((a['t_k'], a['val_ess'], a['accepted']))
    return out

for (xoff, series, label, color) in [
        (0, attempts_series(bal), 'balance', 'tab:blue'),
        (0, attempts_series(kl) + [(t, e, False) for t, e in
                                   zip(KL_S4_T, KL_S4_ESS)],
         'forward KL', 'tab:orange')]:
    xs = np.arange(1, len(series) + 1)
    ts = [s[0] for s in series]
    acc = np.array([s[2] for s in series])
    ax_ = axL
    ax_.plot(xs, ts, lw=0.8, color=color, alpha=0.6)
    ax_.scatter(xs[acc], np.array(ts)[acc], marker='o', s=34, color=color,
                label=f'{label}: accepted', zorder=3)
    if (~acc).any():
        ax_.scatter(xs[~acc], np.array(ts)[~acc], marker='x', s=30,
                    color=color, label=f'{label}: rejected', zorder=3)
axL.axhline(1.0, ls=':', lw=0.8, color='gray')
axL.annotate('ladder complete ($t=1$)', (1.2, 1.012), fontsize=8,
             color='gray')
axL.annotate('12 rejected attempts,\nstuck at $t=0.577$', (10.3, 0.50),
             fontsize=8, color='tab:orange', ha='center')
axL.set_xlabel('training attempt (each = one full stage training)')
axL.set_ylabel('attempted level $t_k$')
axL.set_ylim(0.15, 1.08)
axL.set_title('(a) acceptance loop: attempted levels')
axL.legend(loc='lower right', fontsize=7)

# (b) in-training direct ESS, balance (top) and forward KL (bottom)
for ax_, d, name, color in [(axB, bal, 'balance', 'tab:blue'),
                            (axK, kl, 'forward KL (incomplete)',
                             'tab:orange')]:
    off = 0
    for s in d['stages']:
        h = np.asarray(s['train_ess_hist'], dtype=float)
        ax_.plot(np.arange(off, off + len(h)), h, lw=0.55, color=color,
                 rasterized=True)
        ax_.axvline(off, color='gray', lw=0.4, ls=':')
        ax_.text(off + len(h) / 2, 0.9, f"$t_k={s['t']:.2f}$", ha='center',
                 fontsize=7)
        off += len(h)
    ax_.set_ylabel('direct ESS')
    ax_.set_ylim(0, 1.05)
    ax_.set_title(f'({"b" if ax_ is axB else "c"}) per-step ESS, {name}',
                  fontsize=9, pad=2)
axK.annotate(r'$0.52 \to 0.30$', xy=(2350, 0.36), xytext=(2520, 0.62),
             fontsize=8, color='tab:orange',
             arrowprops=dict(arrowstyle='->', color='tab:orange', lw=0.8))
plt.setp(axB.get_xticklabels(), visible=False)
axK.set_xlabel('gradient step (accepted stages, concatenated)')

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_clock_training.pdf', bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print('wrote', FIGDIR / 'fig_clock_target.pdf')
print('wrote', FIGDIR / 'fig_clock_training.pdf')
