#!/usr/bin/env python
"""Final ADP Ramachandran figure: loads the cached MD data (md_implicit.npz, ~35 ns amber96
implicit, L-alanine) and renders it (raw + smoothed, FAB Fig-19 log-viridis style) next to the
FAB Fig-19 reference (fab_fig19_test.png). Loads cached data only -- no MD rerun. Writes a NEW
file ramachandran_final.png (does not overwrite/delete anything). Run: python make_final_figure.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
TK = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
TL = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]

z = np.load(os.path.join(HERE, "md_implicit.npz"))
phi, psi = z["phi"], z["psi"]; tns = float(z["t_ns"]) if "t_ns" in z.files else 0.0
e = np.linspace(-np.pi, np.pi, 101)
H, _, _ = np.histogram2d(np.radians(phi), np.radians(psi), bins=[e, e], density=True)
H = H.T
Hs = gaussian_filter(H, 1.0, mode="wrap")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (Hplot, lab) in zip(axes[:2], [(H, "raw"), (Hs, "smoothed σ=1")]):
    vmax = Hplot.max(); im = ax.imshow(Hplot, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                                       cmap="viridis", norm=LogNorm(vmin=vmax*1e-4, vmax=vmax), aspect="equal")
    ax.set_xticks(TK); ax.set_xticklabels(TL); ax.set_yticks(TK); ax.set_yticklabels(TL)
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\psi$")
    ax.set_title(f"amber96 implicit MD ({lab})\nt={tns:.0f} ns  N={len(phi)}  frac($\\phi$>0)={(phi>0).mean():.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax = axes[2]; ax.imshow(plt.imread(os.path.join(HERE, "fab_fig19_test.png"))); ax.axis("off")
ax.set_title("FAB Fig. 19 (reference test data)")
fig.suptitle("ADP Ramachandran — local amber96 implicit MD (L-alanine) vs FAB Fig. 19  "
             "[αL mode at φ>0 under-sampled by design; compare flat φ<0 regions]", fontsize=11)
fig.tight_layout()
out = os.path.join(HERE, "ramachandran_final.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out}  (t={tns:.0f}ns N={len(phi)} frac_pos={(phi>0).mean():.4f})", flush=True)
