#!/usr/bin/env python
"""Render the two-panel clock-sector occupancy-bias figure.

The left panel compares training batch sizes at fixed ``N=640000``.  The
right panel shows particle-count scaling at fixed ``B=2000``.  Both panels
are rendered from the merged artifact archives, with no sampler or GPU work.

The two panels share one configuration, ``B=2000`` at ``N=640000``, and each
archive holds a four-repetition sample of it.  Both drivers use the same four
seeds, the same stage flows, and the same chunk of 80000 rows for the MALA
rejuvenation, so the per-chunk keys ``fold_in(key_k, 2000 + chunk)`` coincide
and the two samples are the same computation.  The shared point is taken once,
from the batch-size archive.

Run from the repository root:
    python Codes_New/Lattice_Clock/plot_occupancy_bias.py

Writes ``Codes_New/Lattice_Clock/results/occupancy_bias.png``.
"""

import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
BATCH_DATA = ARTIFACTS / "occupancy_bias_N640000" / "data.npz"
SCALING_DATA = ARTIFACTS / "occupancy_bias_B2000" / "data.npz"
OUTPUT = RESULTS / "occupancy_bias.png"

METHODS = ("klx", "klxx")
LABEL = {
    "klx": r"KL+$\mathrm{X}_\pi$",
    "klxx": r"KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$",
}
COLOR = {"klx": "tab:blue", "klxx": "tab:red"}
EXPECTED_N = 640000
EXPECTED_B_VALUES = (2000, 1000, 500, 250)   # what the archive holds
PLOT_B_VALUES = (2000, 1000, 500)           # B = 250 is too noisy to plot
EXPECTED_BASE_SIZE = 10000
MAX_REPORT_K = 6
# The panels meet at B=2000, N=640000. Both archives measure it; the scaling
# panel takes its value from the batch-size archive so the two agree.
SHARED_B = 2000
SHARED_K = 6


def log(message: str) -> None:
    """Print a timestamped rendering update."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def summarize(values: np.ndarray) -> tuple[float, float]:
    """Return the mean and standard error of a one-dimensional sample."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"expected at least two values, found shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("bias sample contains a nonfinite value")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def load_batch_summary(
) -> tuple[list[int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load four-seed bias summaries at fixed sample size."""
    with np.load(BATCH_DATA, allow_pickle=False) as data:
        particle_count = int(data["N"])
        batch_sizes = [int(value) for value in data["B_values"]]
        seeds = np.asarray(data["seeds"], dtype=np.int64)
        if particle_count != EXPECTED_N:
            raise ValueError(
                f"{BATCH_DATA}: N={particle_count}, expected {EXPECTED_N}"
            )
        if tuple(batch_sizes) != EXPECTED_B_VALUES:
            raise ValueError(
                f"{BATCH_DATA}: B_values={batch_sizes}, "
                f"expected {EXPECTED_B_VALUES}"
            )
        if seeds.shape != (4,):
            raise ValueError(f"{BATCH_DATA}: seed shape {seeds.shape}, expected (4,)")

        plotted = [b for b in batch_sizes if b in PLOT_B_VALUES]
        if tuple(plotted) != PLOT_B_VALUES:
            raise ValueError(
                f"{BATCH_DATA}: cannot plot {PLOT_B_VALUES}, archive has {batch_sizes}"
            )
        means: dict[str, np.ndarray] = {}
        sems: dict[str, np.ndarray] = {}
        for method in METHODS:
            summaries = [
                summarize(data[f"bias_{method}_B{batch_size}"])
                for batch_size in plotted
            ]
            means[method] = np.asarray([value[0] for value in summaries])
            sems[method] = np.asarray([value[1] for value in summaries])
    return plotted, means, sems


def load_scaling_summary(
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load particle-count scaling summaries at fixed batch size.

    The shared point at k=SHARED_K is read from the batch-size archive; every
    other point comes from this archive.  See the module docstring.
    """
    if EXPECTED_BASE_SIZE * 2**SHARED_K != EXPECTED_N:
        raise ValueError(
            f"shared point k={SHARED_K} is N={EXPECTED_BASE_SIZE * 2**SHARED_K}, "
            f"but the batch-size run is at N={EXPECTED_N}"
        )
    with np.load(BATCH_DATA, allow_pickle=False) as batch:
        shared = {
            method: np.asarray(
                batch[f"bias_{method}_B{SHARED_B}"], dtype=np.float64
            )
            for method in METHODS
        }

    with np.load(SCALING_DATA, allow_pickle=False) as data:
        base_size = int(data["BASE_SZIE"])
        ks = [int(value) for value in data["ks"] if int(value) <= MAX_REPORT_K]
        if base_size != EXPECTED_BASE_SIZE:
            raise ValueError(
                f"{SCALING_DATA}: base size {base_size}, "
                f"expected {EXPECTED_BASE_SIZE}"
            )
        if ks != list(range(MAX_REPORT_K + 1)):
            raise ValueError(
                f"{SCALING_DATA}: reported k values {ks}, "
                f"expected {list(range(MAX_REPORT_K + 1))}"
            )

        particle_counts = np.asarray(
            [base_size * 2**k for k in ks], dtype=np.float64
        )
        means: dict[str, np.ndarray] = {}
        sems: dict[str, np.ndarray] = {}
        for method in METHODS:
            own = np.asarray(data[f"bias_{method}_k{SHARED_K}"], dtype=np.float64)
            if own.shape == shared[method].shape:
                log(f"shared point {method}: scaling archive vs batch-size archive, "
                    f"max |difference| over seeds = {float(np.max(np.abs(own - shared[method]))):.3e} "
                    f"(the batch-size values are plotted in both panels)")
            else:
                log(f"shared point {method}: seed counts differ "
                    f"({own.shape} vs {shared[method].shape}); batch-size values plotted")
            summaries = [
                summarize(
                    shared[method] if k == SHARED_K
                    else data[f"bias_{method}_k{k}"]
                )
                for k in ks
            ]
            means[method] = np.asarray([value[0] for value in summaries])
            sems[method] = np.asarray([value[1] for value in summaries])
            if np.any(means[method] <= 0.0):
                raise ValueError(f"{SCALING_DATA}: {method} has nonpositive bias")
    return particle_counts, means, sems


def main() -> None:
    """Render the combined occupancy-bias figure."""
    batch_sizes, batch_means, batch_sems = load_batch_summary()
    particle_counts, scaling_means, scaling_sems = load_scaling_summary()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.fontset": "cm",
            "font.family": "serif",
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 2.8),
        gridspec_kw={"width_ratios": (0.94, 1.06)},
    )

    ax = axes[0]
    positions = np.arange(len(batch_sizes), dtype=np.float64)
    width = 0.36
    for index, method in enumerate(METHODS):
        offset = (index - 0.5) * width
        ax.bar(
            positions + offset,
            batch_means[method],
            width=width,
            yerr=2.0 * batch_sems[method],
            capsize=2.5,
            color=COLOR[method],
            edgecolor="white",
            linewidth=0.5,
            label=LABEL[method],
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
            zorder=3,
        )
        log(
            f"{method} by B: "
            f"means={['%.5f' % value for value in batch_means[method]]}, "
            f"sems={['%.5f' % value for value in batch_sems[method]]}"
        )
    ax.set_title("(a) Batch-size dependence", loc="left")
    ax.xaxis.set_major_locator(FixedLocator(positions))
    ax.xaxis.set_major_formatter(
        FixedFormatter([str(batch_size) for batch_size in batch_sizes])
    )
    ax.xaxis.set_minor_locator(NullLocator())
    batch_ticks = np.arange(0.0, 0.0351, 0.005)
    ax.yaxis.set_major_locator(FixedLocator(batch_ticks))
    ax.yaxis.set_major_formatter(
        FixedFormatter(
            ["0" if value == 0.0 else f"{value:.3f}" for value in batch_ticks]
        )
    )
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.0, 0.02)
    ax.set_xlabel(r"training batch size $B$")
    ax.set_ylabel("occupancy bias")
    ax.grid(axis="y", alpha=0.2, linewidth=0.6, zorder=0)
    ax.legend(loc="upper left", framealpha=0.9)

    ax = axes[1]
    for method in METHODS:
        ax.errorbar(
            particle_counts,
            scaling_means[method],
            yerr=2.0 * scaling_sems[method],
            marker="o",
            markersize=3.8,
            linewidth=1.2,
            capsize=2.5,
            color=COLOR[method],
            label=LABEL[method],
            zorder=3,
        )
        slope = np.polyfit(
            np.log(particle_counts), np.log(scaling_means[method]), 1
        )[0]
        log(
            f"{method} by N: "
            f"means={['%.5f' % value for value in scaling_means[method]]}, "
            f"slope={slope:.3f}"
        )
    reference = scaling_means["klxx"][0] * (
        particle_counts / particle_counts[0]
    ) ** -0.5
    ax.plot(
        particle_counts,
        reference,
        linestyle="--",
        linewidth=1.0,
        color="gray",
        label=r"$N^{-1/2}$ reference",
        zorder=2,
    )
    ax.set_title(r"(b) Sample-size scaling ($B=2000$)", loc="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(particle_counts))
    ax.xaxis.set_major_formatter(
        FixedFormatter(
            [
                r"$10^4$",
                r"$2\!\times\!10^4$",
                r"$4\!\times\!10^4$",
                r"$8\!\times\!10^4$",
                r"$1.6\!\times\!10^5$",
                r"$3.2\!\times\!10^5$",
                r"$6.4\!\times\!10^5$",
            ]
        )
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([2e-3, 5e-3, 1e-2, 2e-2, 4e-2]))
    ax.yaxis.set_major_formatter(
        FixedFormatter(
            [
                r"$2\!\times\!10^{-3}$",
                r"$5\!\times\!10^{-3}$",
                r"$10^{-2}$",
                r"$2\!\times\!10^{-2}$",
                r"$4\!\times\!10^{-2}$",
            ]
        )
    )
    ax.yaxis.set_minor_locator(NullLocator())
    plt.setp(
        ax.get_xticklabels(),
        rotation=25,
        horizontalalignment="right",
        rotation_mode="anchor",
        fontsize=7,
    )
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel("occupancy bias")
    ax.grid(alpha=0.2, linewidth=0.6, zorder=0)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7.4)

    fig.tight_layout(w_pad=1.1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"DONE — saved {OUTPUT}")


if __name__ == "__main__":
    main()
