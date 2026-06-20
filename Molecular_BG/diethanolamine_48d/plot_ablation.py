#!/usr/bin/env python
"""glycerol per-stage validation-ESS ladder -- the paper ladder.png style (make_summary.plot_ladder),
extended to the three ablation methods. Two panels: raw (left) and sharpen (right). Single t-axis per
panel with the shared ladder nodes; forward KL arcs ABOVE (blue), KL+X_mu+X_{(mu^+nu^)/2} BELOW (red),
and its delta-reweighted variant BELOW the red as a taller arc set (purple). Each accepted stage is an
arc whose apex is labelled with that stage's validation ESS; a cross marks an unreached final bridge.
The headline factor F is tabular (results_table.md). Saves glycerol_36d/ladder.png at 400 dpi.
Usage: python plot_ablation.py [glycerol_36d]
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

C_KL, C_X, C_XD = "#1F77B4", "#9467BD", "#D62728"          # forward KL (blue) ; KL+X unweighted (purple) ; KL+X delta = best (red)
XNAME = r"KL$+\mathrm{X}_\mu+\mathrm{X}_{(\hat\mu+\bar\nu)/2}$"
# per panel (schedule): (forward-KL tag, X tag, X-delta tag)
PANELS = [
    ("raw pushforward", ("kl_raw",     "klxx_raw",     "klxx_delta_raw")),
    ("sharpening",      ("kl_sharpen", "klxx_sharpen", "klxx_delta_sharpen")),
]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "mathtext.fontset": "cm", "font.family": "serif"})


def load(folder, tag):
    p = os.path.join(folder, f"data_{tag}.pth")
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def draw(ax, dd, color, sign, hboost=0.0):
    """plot_ladder arc convention: arcs on the t-axis (sign +1 above / -1 below), height ~ step width,
    per-stage validation ESS on the apex; hboost deepens a set (purple, below the red)."""
    if not dd:
        return
    stages = dd.get("stages", [])
    ts = [0.0] + [s["t"] for s in stages]
    for i, s in enumerate(stages):
        t0, t1 = ts[i], ts[i + 1]
        mid, r = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
        th = np.linspace(0, np.pi, 60)
        h = 0.18 + 1.0 * r + hboost                       # arc height grows with the step width
        ax.plot(mid + r * np.cos(th), sign * h * np.sin(th), color=color, lw=2.0, zorder=3)
        small = r < 0.07 and i % 2 == 1                    # destagger tight (small-t) labels
        lift = 0.04 + ((0.04 if sign > 0 else 0.10) if small else 0.0)  # blue limited by title; red/purple have room below
        se = s.get("sharpen_ess")                          # sharpen runs: "val (sharpen)"; raw: just "val"
        lbl = f"{s['val_ess']:.2f}" + (f" ({se:.2f})" if se is not None else "")
        fs = 8.0 if r < 0.07 else 10.0                     # larger ESS numbers; small-t labels a bit smaller
        ax.text(mid, sign * (h + lift), lbl, ha="center",
                va="bottom" if sign > 0 else "top", color=color, fontsize=fs, zorder=4)
    ax.plot(ts, [0] * len(ts), "o", color="0.2", ms=4, zorder=5)
    if not dd.get("complete", True):                      # cross at an unreached final bridge
        ax.plot(1.0, 0.0, "x", color=color, ms=11, mew=2.5, zorder=6)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "glycerol_36d"
    folder = arg if os.path.isabs(arg) else os.path.join(REPO, arg)
    base = os.path.basename(folder.rstrip("/")); name, dim = base.rsplit("_", 1)
    dim = dim[:-1] if dim.endswith("d") else dim
    import shutil; shutil.copy(os.path.abspath(__file__), os.path.join(folder, os.path.basename(__file__)))  # snapshot into the molecule folder (like run.py)

    cache = {}                                            # load each method once
    def L(t):
        if t not in cache:
            cache[t] = load(folder, t)
        return cache[t]
    pops = [(sched, tags) for sched, tags in PANELS if any(L(t) for t in tags)]  # only schedules with a run
    n = max(1, len(pops))
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.9 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (sched, (t_kl, t_x, t_xd)) in zip(axes, pops):
        ax.axhline(0, color="0.3", lw=1.0, zorder=1)
        draw(ax, L(t_kl), C_KL, +1)                       # forward KL above (blue)
        draw(ax, L(t_x),  C_X,  -1)                       # KL+X below (red)
        draw(ax, L(t_xd), C_XD, -1, hboost=0.38 if L(t_x) else 0.0)  # purple below red; if no red, sit at the normal depth
        ax.set_title(f"({sched})", pad=6)                 # breathing room title -> arcs
        ax.set_xlim(-0.03, 1.05); ax.set_ylim(-1.22, 0.66)  # headroom above blue and below purple
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([])
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel(r"$t$ (temperature)")             # x-axis on the bottom row only

    has_kl = any(L(t) for t in ("kl_raw", "kl_sharpen"))          # legend: only families that actually ran
    has_x  = any(L(t) for t in ("klxx_raw", "klxx_sharpen"))
    has_xd = any(L(t) for t in ("klxx_delta_raw", "klxx_delta_sharpen"))
    handles = []
    if has_kl: handles.append(Line2D([0], [0], color=C_KL, lw=2.5, label="forward KL"))
    if has_x:  handles.append(Line2D([0], [0], color=C_X, lw=2.5, label=XNAME))
    if has_xd: handles.append(Line2D([0], [0], color=C_XD, lw=2.5, label=XNAME + r" ($\delta$-reweighted)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.965), h_pad=3.0)
    fig.suptitle(rf"{name} ($d={dim}$) — per-stage ESS", y=0.985, fontsize=13)
    out = os.path.join(folder, "ladder.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  (400 dpi)", flush=True)


if __name__ == "__main__":
    main()
