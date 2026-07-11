"""Paper Figure 7 (fig_clock_target) on the L=8 klxx B=25k run — structure
of the cold clock measure at d = 64, displayed on 10^6 reweighted samples:

    (a) magnetization-plane scatter colored by sector, 1500 points/sector,
    (b) per-site angle marginal: reweighted (top) vs pushforward (bottom),
        dashed well boundaries, black line at the uniform density 1/(2pi),
    (c) sector occupancy of reweighted and pushforward samples, black line
        at the uniform occupancy 1/6.

The reweighted set is one fresh staged rebuild of N = 10^6 source draws
through the trained ladder (map-reweight-resample per stage, the staged
sampler of occupancy_bias_B25k.py); it is generated ONCE on the GPU and
saved to rebuild_klxx_B25k_N1000000.npz — reruns re-render the figure
from the saved data with no GPU work. The pushforward samples (maps alone,
no reweighting) come from data_klxx_B25k.npz.

Run from the repo root:
    PYTHONPATH=/mnt/projects/jflows ~/.envs/jax/bin/python \
        Codes/Lattice_Clock/plot_clock_target.py
Writes fig_clock_target.png (and the rebuild npz on first run) next to
this file; progress in plot_clock_target_status.log.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

P = 6
N_REBUILD = 1000000
REBUILD_SEED = 777
REBUILD = HERE / f"rebuild_klxx_B25k_N{N_REBUILD}.npz"
STATUS = HERE / "plot_clock_target_status.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as fh:
        fh.write(line + "\n")


def get_rebuild() -> np.ndarray:
    """The 10^6-sample staged rebuild (saved once, reloaded thereafter)."""
    if REBUILD.exists():
        log(f"rebuild found: {REBUILD.name} (no GPU work)")
        return np.load(REBUILD)["y"]
    log(f"rebuild missing — running the staged sampler at N={N_REBUILD} ...")
    from occupancy_bias_B25k import load_run, staged_sample
    ladder, flows = load_run("klxx")
    t0 = time.time()
    y = np.asarray(staged_sample(N_REBUILD, 1, seed=REBUILD_SEED,
                                 ladder=ladder, flows=flows), dtype=np.float32)
    np.savez_compressed(REBUILD, y=y, seed=REBUILD_SEED,
                        ladder=np.asarray(ladder))
    log(f"rebuild done in {time.time() - t0:.0f}s — saved {REBUILD.name}")
    return y


def main() -> None:
    open(STATUS, "a").close()
    log("START fig_clock_target (klxx B=25k)")
    data = np.load(HERE / "data_klxx_B25k.npz")
    y = get_rebuild().astype(np.float32)              # [1e6, 64] reweighted
    yp = data["samples_push"].astype(np.float32)      # pushforward (maps alone)

    m = np.exp(1j * y).mean(axis=1)
    sector = np.round(np.angle(m) * P / (2 * np.pi)).astype(np.int64) % P
    log(f"rebuild: N={y.shape[0]}  mean|m|={np.abs(m).mean():.4f}  "
        f"push: N={yp.shape[0]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "mathtext.fontset": "cm", "font.family": "serif",
    })
    colors = plt.get_cmap("tab10")(np.arange(P))

    fig = plt.figure(figsize=(10.0, 3.1))
    gs = fig.add_gridspec(1, 3)
    axA = fig.add_subplot(gs[0])
    gsb = gs[1].subgridspec(2, 1, hspace=0.0)   # (b): two glued histograms
    axB1 = fig.add_subplot(gsb[0])
    axB2 = fig.add_subplot(gsb[1], sharex=axB1)
    axC = fig.add_subplot(gs[2])

    # (a) magnetization plane
    for k in range(P):
        sel = np.nonzero(sector == k)[0][:1500]
        axA.scatter(m.real[sel], m.imag[sel], s=2, alpha=0.18,
                    color=colors[k], rasterized=True)
    th = np.linspace(0, 2 * np.pi, 256)
    axA.plot(np.cos(th), np.sin(th), lw=0.6, color="gray")
    for k in range(P):
        a = 2 * np.pi * k / P
        axA.plot([0, 1.05 * np.cos(a)], [0, 1.05 * np.sin(a)], lw=0.5, ls=":",
                 color="gray")
        axA.annotate(f"$s={k}$", (0.78 * np.cos(a), 0.78 * np.sin(a)),
                     fontsize=7.5, ha="center", va="center", color="black",
                     path_effects=[pe.withStroke(linewidth=1.6,
                                                 foreground="white")])
    axA.set_xlim(-1.12, 1.12); axA.set_ylim(-1.12, 1.12)
    axA.set_aspect("equal")
    axA.set_xlabel(r"$\mathrm{Re}\, m(\theta)$")
    axA.set_ylabel(r"$\mathrm{Im}\, m(\theta)$")
    axA.text(0.02, 0.02, r"each point: one $\theta \in [-\pi,\pi)^{64}$",
             transform=axA.transAxes, fontsize=7)
    axA.set_title(r"(a) magnetization plane")

    # (b) per-site angle marginal: reweighted (top) / pushforward (bottom)
    shift = lambda a: np.remainder(a + np.pi / 6, 2 * np.pi) - np.pi / 6
    edges = np.linspace(-np.pi / 6, 2 * np.pi - np.pi / 6, 181)  # 30 bins/block
    for a, dat, alf in ((axB1, y.ravel(), None),       # colors as in (c)
                        (axB2, yp.ravel(), 0.45)):
        h, _ = np.histogram(shift(dat), bins=edges, density=True)
        for k in range(P):
            a.stairs(h[30 * k:30 * (k + 1)], edges[30 * k:30 * (k + 1) + 1],
                     fill=True, color=colors[k], alpha=alf)
    axB1.text(0.98, 0.90, "reweighted", transform=axB1.transAxes,
              fontsize=8, ha="right", va="top")
    axB2.text(0.98, 0.90, "pushforward", transform=axB2.transAxes,
              fontsize=8, ha="right", va="top")
    axB1.set_xlim(-np.pi / 6, 2 * np.pi - np.pi / 6)
    ytop = max(axB1.get_ylim()[1], axB2.get_ylim()[1])
    for a in (axB1, axB2):
        a.set_ylim(0, ytop)
        a.set_ylabel("density")
        for k in range(5):                # six blocks, one per clock angle
            a.axvline(np.pi / 6 + k * np.pi / 3, ls="--", lw=0.6, color="gray")
        a.axhline(1 / (2 * np.pi), ls="--", lw=0.8, color="black")
    secx = axB1.secondary_xaxis("top")    # block labels s=0..5 along the top
    secx.set_xticks([k * np.pi / 3 for k in range(6)])
    secx.set_xticklabels([f"$s={k}$" for k in range(6)])
    secx.tick_params(length=0, pad=2, labelsize=7.5)
    axB1.tick_params(labelbottom=False)            # glued: labels only below
    axB1.yaxis.set_major_locator(MaxNLocator(4, prune="lower"))
    axB2.yaxis.set_major_locator(MaxNLocator(4, prune="upper"))
    axB2.set_xticks([k * np.pi / 3 for k in range(6)])
    axB2.set_xticklabels([r"$0$", r"$\pi/3$", r"$2\pi/3$", r"$\pi$",
                          r"$4\pi/3$", r"$5\pi/3$"])
    axB2.set_xlabel(r"site angle $\theta_j$")
    axB1.set_title(r"(b) site angle marginal", pad=14)

    # (c) sector occupancy
    cv = np.bincount(sector, minlength=P).astype(float); cv /= cv.sum()
    cp = np.asarray(data["counts_push"], dtype=float); cp /= cp.sum()
    x = np.arange(P); w = 0.4
    axC.bar(x - w / 2, cv, w, color=[c for c in colors])
    axC.bar(x + w / 2, cp, w, color=[c for c in colors], alpha=0.45)
    for xi, v in zip(x, cv):
        axC.text(xi - w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.5,
                 path_effects=[pe.withStroke(linewidth=1.6,
                                             foreground="white")])
    axC.axhline(1 / 6, ls="--", lw=0.8, color="black")     # uniform occupancy
    axC.set_xlabel(r"sector $s$")
    axC.set_ylabel("occupancy")
    axC.set_ylim(0, max(0.30, cv.max() + 0.05))
    axC.set_title("(c) sector occupancy")
    axC.legend(handles=[Patch(facecolor="0.35", label="reweighted"),
                        Patch(facecolor="0.35", alpha=0.45,
                              label="pushforward")],
               loc="upper right")

    plt.tight_layout()
    out = HERE / "fig_clock_target.png"
    plt.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"DONE — saved {out}  occupancy(reweighted)={np.round(cv, 4).tolist()} "
        f"occupancy(push)={np.round(cp, 4).tolist()}")


if __name__ == "__main__":
    main()
