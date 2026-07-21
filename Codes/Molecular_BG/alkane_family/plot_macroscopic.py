#!/usr/bin/env python
"""Compare KLXX macroscopic observables with OpenMM benchmarks."""

import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "macroscopic.json"
OPENMM = ROOT / "openmm_reference" / "reference.json"
OPENMM_TORSION = ROOT / "openmm_reference" / "torsion_reference.json"
OUTPUT = ROOT / "results" / "macroscopic.png"


plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})


def values(records, name):
    return np.asarray([record[name] for record in records], dtype=float)


def seed_range(records, name):
    means = np.asarray([
        [seed[name] for seed in record["seeds"]]
        for record in records
    ])
    center = values(records, name)
    return np.vstack((center - means.min(axis=1), means.max(axis=1) - center))


def openmm_records(dimensions):
    direct = json.loads(OPENMM.read_text())["molecules"]
    torsional = json.loads(OPENMM_TORSION.read_text())["molecules"]
    return [
        (torsional if dimension >= 36 else direct)[str(dimension)]
        for dimension in dimensions
    ]


def stacked_bars(ax, x, records, width, hatch=None):
    trans = values(records, "trans_fraction")
    gauche_plus = values(records, "gauche_plus_fraction")
    gauche_minus = values(records, "gauche_minus_fraction")
    options = {
        "width": width,
        "edgecolor": "#333333" if hatch else "white",
        "linewidth": 0.65,
        "hatch": hatch,
    }
    ax.bar(x, trans, color="#0072B2", **options)
    ax.bar(x, gauche_plus, bottom=trans, color="#E69F00", **options)
    ax.bar(
        x,
        gauche_minus,
        bottom=trans + gauche_plus,
        color="#009E73",
        **options,
    )


def main():
    data = json.loads(DATA.read_text())
    dimensions = sorted(int(key) for key in data)
    klxx = [data[str(dimension)] for dimension in dimensions]
    raw = openmm_records(dimensions)
    carbon_count = np.asarray(dimensions) // 9
    klxx_x = carbon_count - 0.035
    openmm_x = carbon_count + 0.035

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.25),
        gridspec_kw={"width_ratios": (1.05, 1.22, 1.15)},
    )

    ax = axes[0]
    klxx_energy = values(klxx, "physical_energy_mean_kj_mol")
    raw_energy = values(raw, "physical_energy_mean_kj_mol")
    ax.plot(
        klxx_x,
        klxx_energy,
        color="#0072B2",
        marker="o",
        markersize=4.8,
        linewidth=1.5,
        label="KLXX",
        zorder=3,
    )
    ax.errorbar(
        openmm_x,
        raw_energy,
        yerr=seed_range(raw, "physical_energy_mean_kj_mol"),
        color="#333333",
        marker="D",
        markerfacecolor="white",
        markersize=4.2,
        linestyle="--",
        linewidth=1.25,
        capsize=2.5,
        label="OpenMM",
        zorder=2,
    )
    ax.set_title("(a) Physical energy", loc="left")
    ax.set_xlabel(r"carbon count $n_{\mathrm{C}}$")
    ax.set_ylabel(r"$U$ (kJ mol$^{-1}$)")
    ax.set_xticks(carbon_count)
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="upper left", framealpha=0.9)

    ax = axes[1]
    observables = (
        ("carbon_radius_mean_nm", r"$R_{g,\mathrm{C}}$", "#009E73", "o"),
        ("carbon_end_to_end_mean_nm", r"$R_{ee,\mathrm{C}}$", "#D55E00", "s"),
    )
    for name, label, color, marker in observables:
        ax.plot(
            klxx_x,
            values(klxx, name),
            color=color,
            marker=marker,
            markersize=4.7,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
        ax.errorbar(
            openmm_x,
            values(raw, name),
            yerr=seed_range(raw, name),
            color=color,
            marker=marker,
            markerfacecolor="white",
            markersize=4.2,
            linestyle="--",
            linewidth=1.15,
            capsize=2.2,
            zorder=2,
        )
    ax.set_title("(b) Carbon-skeleton size", loc="left")
    ax.set_xlabel(r"carbon count $n_{\mathrm{C}}$")
    ax.set_ylabel("distance (nm)")
    ax.set_xticks(carbon_count)
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, linewidth=0.6)
    shape_legend = ax.legend(loc="upper left", framealpha=0.9)
    ax.add_artist(shape_legend)
    ax.legend(
        handles=(
            Line2D([], [], color="#555555", marker="o", label="KLXX"),
            Line2D(
                [],
                [],
                color="#555555",
                marker="o",
                markerfacecolor="white",
                linestyle="--",
                label="OpenMM",
            ),
        ),
        loc="lower right",
        framealpha=0.9,
    )

    ax = axes[2]
    torsional_klxx = [record for record in klxx if record["backbone_torsion_count"]]
    torsional_raw = [record for record in raw if record["backbone_torsion_count"]]
    torsional_count = np.asarray([
        record["dimension"] // 9 for record in torsional_klxx
    ])
    offset = 0.18
    stacked_bars(ax, torsional_count - offset, torsional_klxx, 0.32)
    stacked_bars(ax, torsional_count + offset, torsional_raw, 0.32, "////")
    ax.set_title("(c) Backbone rotamers", loc="left")
    ax.set_xlabel(r"carbon count $n_{\mathrm{C}}$")
    ax.set_ylabel("rotamer population")
    ax.set_xticks(torsional_count)
    ax.set_xlim(3.45, 6.55)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    rotamer_legend = ax.legend(
        handles=(
            Patch(facecolor="#0072B2", label="trans"),
            Patch(facecolor="#E69F00", label=r"gauche$^+$"),
            Patch(facecolor="#009E73", label=r"gauche$^-$"),
        ),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
    )
    ax.add_artist(rotamer_legend)
    ax.legend(
        handles=(
            Patch(facecolor="0.75", edgecolor="#333333", label="KLXX"),
            Patch(
                facecolor="white",
                edgecolor="#333333",
                hatch="////",
                label="OpenMM",
            ),
        ),
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0,
        frameon=False,
    )

    fig.tight_layout(w_pad=1.25)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    relative_energy = np.abs(klxx_energy - raw_energy) / np.abs(raw_energy)
    print(OUTPUT)
    print(f"maximum relative energy difference: {relative_energy.max():.3%}")


if __name__ == "__main__":
    main()
