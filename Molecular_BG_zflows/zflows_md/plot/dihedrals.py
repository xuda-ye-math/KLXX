#!/usr/bin/env python
"""Per-molecule multimodal dihedral marginals: MD reference vs forward KL vs KL+X_mu+X_mix.
Stored in each molecule folder as dihedrals.png. CPU-only.

Molecules with saved bg_dihedrals/ref_dihedrals (alkanes) plot directly. Hetero molecules
need an inference pass (reconstruct flow -> generate -> dihedrals) handled separately.
"""
import torch
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except ImportError:
    HAVE_KDE = False

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Molecular_BG")
plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm", "font.family": "serif"})  # ../Log* style
C_MD, C_KL, C_KLXX = "0.5", "#1F77B4", "#D62728"   # MD grey ; forward KL blue ; KL+X red


def density(data, grid):
    """Periodic (wrapped) density on [-180,180] degrees."""
    data = np.asarray(data, float).ravel()
    data = data[np.isfinite(data)]
    if HAVE_KDE and data.size > 5:
        tiled = np.concatenate([data - 360, data, data + 360])
        return gaussian_kde(tiled, bw_method=0.10)(grid) * 3.0
    h, e = np.histogram(data, bins=48, range=(-180, 180), density=True)
    c = 0.5 * (e[:-1] + e[1:])
    return np.interp(grid, c, h)


def plot_marginals(folder, name, d, bg_klxx, bg_kl, ref, kind="backbone dihedral"):
    """bg_klxx/bg_kl/ref: arrays (N, ndih) of angles in degrees."""
    bg_klxx, bg_kl, ref = np.asarray(bg_klxx), np.asarray(bg_kl), np.asarray(ref)
    ndih = bg_klxx.shape[1]
    nrow = max(1, int(np.floor(np.sqrt(ndih))))        # squarest grid that fits N panels
    ncol = int(np.ceil(ndih / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.35 * nrow), squeeze=False)
    grid = np.linspace(-180, 180, 400)
    for i in range(ndih):
        ax = axes[i // ncol][i % ncol]
        ax.fill_between(grid, density(ref[:, i], grid), color=C_MD, alpha=0.40, lw=0, zorder=1)
        ax.plot(grid, density(bg_klxx[:, i], grid), color=C_KLXX, lw=2.4, alpha=0.9, zorder=2)
        ax.plot(grid, density(bg_kl[:, i], grid), color=C_KL, lw=1.9, ls=(0, (5, 5)), zorder=3)
        ax.set_title(rf"$\phi_{{{i + 1}}}$")
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, 0, 180])
        ax.set_yticks([])
        if i % ncol == 0:
            ax.set_ylabel("density", labelpad=6)
    empty = [axes[j // ncol][j % ncol] for j in range(ndih, nrow * ncol)]
    for ax in empty:
        ax.axis("off")
    handles = [Patch(fc=C_MD, alpha=0.40, label="MD reference"),
               Line2D([0], [0], color=C_KLXX, lw=2.4, label=r"KL$+X_\mu+X_{\mathrm{mix}}$"),
               Line2D([0], [0], color=C_KL, lw=1.8, ls=(0, (6, 3)), label="forward KL")]
    fig.suptitle(rf"{name} ($d={d}$) — {kind} marginals", y=0.995)
    if empty:                                          # legend in the first empty slot
        empty[0].legend(handles=handles, loc="center", frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:                                              # full grid -> bottom-center legend
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = os.path.join(folder, "dihedrals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({ndih} dihedrals)")


def fab_marginals(folder, name, d, bg_klxx, bg_kl, ref, labels, kind="dihedral"):
    """FAB-style (arXiv 2208.01893) marginals: one row per torsion, normal | log columns.
    MD ground truth = thick black; forward KL + KL+X overlaid; phi axis in pi units; the log
    column exposes where the methods diverge in the minor modes / tails."""
    to_rad = lambda x: np.deg2rad(np.asarray(x, float))
    bg_klxx, bg_kl, ref = to_rad(bg_klxx), to_rad(bg_kl), to_rad(ref)
    n = ref.shape[1]
    grid = np.linspace(-np.pi, np.pi, 500)
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tlab = [r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"]

    def kde(a):
        a = a[np.isfinite(a)]
        if HAVE_KDE and a.size > 5:
            t = np.concatenate([a - 2 * np.pi, a, a + 2 * np.pi])
            return gaussian_kde(t, bw_method=0.05)(grid) * 3.0
        h, e = np.histogram(a, bins=60, range=(-np.pi, np.pi), density=True)
        return np.interp(grid, 0.5 * (e[:-1] + e[1:]), h)

    fig, axes = plt.subplots(n, 2, figsize=(8.6, 2.5 * n), squeeze=False)
    for i in range(n):
        dmd, dkl, dkx = kde(ref[:, i]), kde(bg_kl[:, i]), kde(bg_klxx[:, i])
        for j, ax in enumerate(axes[i]):
            ax.plot(grid, dmd, color="k", lw=2.4, label="MD (ground truth)", zorder=1)
            ax.plot(grid, dkl, color=C_KL, lw=1.8, label="forward KL", zorder=2)
            ax.plot(grid, dkx, color=C_KLXX, lw=1.8, label=r"KL$+X_\mu+X_{\mathrm{mix}}$", zorder=3)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tlab)
            if j == 1:
                ax.set_yscale("log")
                ax.set_ylim(1e-4, None)
            if i == 0:
                ax.set_title("normal scale" if j == 0 else "log scale (minor modes / tails)")
            if i == n - 1:
                ax.set_xlabel(r"dihedral $\phi$ (rad)")
            if j == 0:
                ax.set_ylabel(f"{labels[i]}\nprob. density")
    axes[0][1].legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.suptitle(rf"{name} ($d={d}$) — multimodal {kind} marginals", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(folder, "dihedrals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({n} torsions x normal/log)", flush=True)


def plot_direct(name, d):
    folder = os.path.join(ROOT, f"{name}_{d}d")
    klxx = torch.load(os.path.join(folder, "data_klxx.pth"), weights_only=False)
    kl = torch.load(os.path.join(folder, "data_kl.pth"), weights_only=False)
    if klxx.get("bg_dihedrals") is None:
        print(f"{name}: no saved dihedrals (needs inference)"); return
    ref = klxx["ref_dihedrals"]
    longref = os.path.join(folder, "ref_dihedrals_40ns.npy")   # converged reference (ref_md.py)
    if os.path.exists(longref):
        ref = np.load(longref)
        print(f"  using converged reference {os.path.basename(longref)} (N={len(ref)})")
    plot_marginals(folder, name, d, klxx["bg_dihedrals"], kl["bg_dihedrals"], ref)


if __name__ == "__main__":
    print("KDE available:", HAVE_KDE)
    plot_direct("dodecane", 30)   # the kept alkane with saved dihedrals
