#!/usr/bin/env python
"""Single-enantiomer alanine-dipeptide Ramachandran, folded from the symmetric
(L+D) data cached in ramachandran_data.npz by ramachandran.py.

The vacuum, achiral Amber force field samples both enantiomers, so the raw
Ramachandran is centrally symmetric under (phi,psi)->(-phi,-psi). ~96% of the
density sits in the C7eq enantiomer pair -- L at (phi<0,psi>0), D at
(phi>0,psi<0) -- which are cleanly split by the symmetry-invariant diagonal
psi=phi (the minor alpha_L/C7ax basins carry only ~4%). We therefore fold the
"wrong-side" half onto the chosen enantiomer via (phi,psi)->(-phi,-psi); each
folded plot keeps the FULL sample count (N=100000), the same amount of data as
the original two-form figure, and is directly comparable to the single-
enantiomer Ramachandran plots in the literature.

Re-plots instantly from the npz (numpy/matplotlib/scipy only, no SMC replay).
Usage: python ramachandran_single.py
"""
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "font.family": "serif"})
LOG = os.path.join(HERE, "ramachandran_single.log"); open(LOG, "w").close()
def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")

BINS, SIGMA = 64, 1.1                                          # match ramachandran.py defaults
def wrap180(x): return ((np.asarray(x, float) + 180.0) % 360.0) - 180.0
edges = np.linspace(-180, 180, BINS + 1); centers = 0.5 * (edges[:-1] + edges[1:])
def Hmap(phi, psi):
    h, _, _ = np.histogram2d(wrap180(phi), wrap180(psi), bins=[edges, edges], density=True)
    return gaussian_filter(h.T, SIGMA, mode="wrap")           # periodic smooth -> FAB-like landscape

def fold(phi, psi, form):
    """Fold the symmetric (L+D) samples onto ONE enantiomer via (phi,psi)->(-phi,-psi).
    L keeps the psi>phi half (dominant C7eq at phi<0,psi>0); R keeps psi<phi (the mirror, = D)."""
    phi, psi = wrap180(phi), wrap180(psi)
    m = (psi < phi) if form == "L" else (psi > phi)           # samples on the wrong side -> reflect
    pf, sf = phi.copy(), psi.copy()
    pf[m], sf[m] = -phi[m], -psi[m]
    return pf, sf

log("START ramachandran_single: loading ramachandran_data.npz")
z = np.load(os.path.join(HERE, "ramachandran_data.npz"))
ref = (z["ref_phi"], z["ref_psi"]); bg = (z["bg_phi"], z["bg_psi"])
log(f"  loaded N={len(ref[0])} samples each (ref, bg)")

for form in ("L", "R"):
    HR = Hmap(*fold(*ref, form)); HB = Hmap(*fold(*bg, form))
    vmax = max(HR.max(), HB.max()); vmin = vmax * 1e-4
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
    for ax, (lab, Hm) in zip(axes, [("annealed SMC reference", HR), ("Boltzmann generator (BG)", HB)]):
        im = ax.imshow(np.clip(Hm, vmin, None), origin="lower", extent=(-180, 180, -180, 180),
                       cmap="viridis", norm=norm, aspect="equal", interpolation="bilinear")
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=10)
        ax.set_xlabel(r"$\phi$ (deg)", fontsize=11.5); ax.set_title(lab, fontsize=11.5, pad=7)
        pj, fj = np.unravel_index(int(np.argmax(Hm)), Hm.shape)  # the single folded basin maximum
        px, py = float(centers[fj]), float(centers[pj])
        ax.plot(px, py, "o", color="red", ms=8, markeredgecolor="white", markeredgewidth=1.4, zorder=6)
        ax.annotate(form, (px, py), textcoords="offset points", xytext=(9, 7), ha="center",
                    color="white", fontsize=13, fontweight="bold", zorder=7,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="black")])
    axes[0].set_ylabel(r"$\psi$ (deg)", fontsize=11.5)
    axes[1].tick_params(labelleft=False)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.035, shrink=0.86, aspect=26)
    cb.set_label("density (log scale)", fontsize=10, labelpad=10); cb.ax.tick_params(labelsize=9)
    name = {"L": "L", "R": "R\\,(D)"}[form]
    fig.suptitle(rf"alanine dipeptide ($d=60$) — Ramachandran, {name}-enantiomer only (folded, $N=10^5$)",
                 y=0.96, fontsize=12.5)
    fig.text(0.5, -0.02, r"both enantiomers folded onto one via the $(\phi,\psi)\to(-\phi,-\psi)$ symmetry",
             ha="center", fontsize=9.5)
    out = os.path.join(HERE, f"ramachandran_{form}.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    log(f"  DONE {form}-enantiomer -> {os.path.basename(out)}")
log("END ramachandran_single")
