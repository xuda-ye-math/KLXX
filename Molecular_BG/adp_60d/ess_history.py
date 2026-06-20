#!/usr/bin/env python
"""ADP KL+X+X (klxx_delta_sharpen) per-stage per-step ESS history, 2x6 grid (11 stages + a summary panel).

Each stage panel: the per-step direct ESS during that stage's flow training (train_ess_hist, blue). The flow
ENTERS as the identity map (run_boltzmann flow.zeros()), so step-0 ESS = the no-flow identity baseline
(grey dotted). The best ESS reached (red dashed) is the gate-snapshot the stage accepts. The improvement
  Delta = best - identity
quantifies how much the trained flow beats the identity map at that stage; Delta ~ 0 means the flow never
improved on identity (the identity/SMC step is effectively what is accepted). The 12th panel summarises
Delta across stages. Reads only data_klxx_delta_sharpen.pth (no GPU). Saves ess_history.png.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "font.family": "serif"})
REPO = "/mnt/projects/zflows-md"; folder = os.path.join(REPO, "adp_60d")
d = torch.load(os.path.join(folder, "data_klxx_delta_sharpen.pth"), weights_only=False)
st = d["stages"]; K = len(st)
print(f"klxx stages: {K}")

from matplotlib.lines import Line2D
BLUE, RED = "#1F77B4", "#D62728"                              # ESS curve (red) ; identity map (blue dashed)
METHOD = r"BG: forward KL$+\mathrm{X}_\mu+\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ ($\delta$-reweighted)"
fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.5)); axes = axes.ravel()   # 3x4 for paper: larger panels, bigger fonts
imps = []
for k, s in enumerate(st):
    ax = axes[k]; h = np.asarray(s["train_ess_hist"], float)
    ident, best = float(h[0]), float(h.max()); imp = best - ident; imps.append(imp)
    val, sharp = s.get("val_ess"), s.get("sharpen_ess")                  # validation + sharpening (accepted) ESS
    ax.plot(np.arange(len(h)), h, color=RED, lw=0.8)                      # per-step ESS curve (red)
    ax.axhline(ident, color=BLUE, ls="--", lw=1.4, zorder=1)             # identity-map level (t=0), blue dashed
    ax.plot([0], [ident], marker="o", ms=6.5, mfc="black", mec="black", clip_on=False, zorder=5)  # identity point at t=0
    ax.set_title(rf"stage {k+1}  ($t={s['t']:.2f}$)", fontsize=12)        # simple: stage + t only (method -> legend)
    ax.set_xlim(0, 500); ax.set_xticks([0, 250, 500]); ax.margins(y=0.10)  # x exactly [0,500] with ticks; ylim flexible
    ax.tick_params(labelsize=10)
    if k % 4 == 0: ax.set_ylabel("ESS", fontsize=12)
    ax.text(0.5, -0.15, f"final ESS: {val:.2f} ({sharp:.2f})", transform=ax.transAxes,
            ha="center", va="top", fontsize=11)                          # validation (sharpening), just below each panel
# summary panel (the 12th cell): improvement (best - identity) per stage
ax = axes[11]
ax.bar(range(1, K + 1), imps, color="0.5", alpha=0.85, width=0.72)
ax.set_title(r"improvement: best $-$ identity", fontsize=12)
ax.set_xlabel("stage", fontsize=12)                                       # no y-label (it overlapped the neighbour panel)
ax.set_xticks(range(1, K + 1)); ax.tick_params(labelsize=9); ax.set_ylim(0, None)
handles = [Line2D([0], [0], color=RED, lw=2.0, label=METHOD),
           Line2D([0], [0], color=BLUE, marker="o", mfc="black", mec="black", ls="--", lw=1.6, ms=7,
                  label=r"identity map ($t=0$)")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.015))
fig.suptitle(r"alanine dipeptide ($d=60$) — per-stage ESS history", y=1.0, fontsize=15)  # lowercase; method in bottom legend
fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.11, hspace=0.42, wspace=0.24)
out = os.path.join(folder, "ess_history.png")
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote", out, "| per-stage Delta:", [round(x, 3) for x in imps])
