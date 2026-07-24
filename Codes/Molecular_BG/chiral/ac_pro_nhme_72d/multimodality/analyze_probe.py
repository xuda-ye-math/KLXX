#!/usr/bin/env python
"""Analyze and plot the saved raw OpenMM Ac-Pro-NHMe multimodality probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent


def dihedral(positions: np.ndarray, atoms) -> np.ndarray:
    a, b, c, d = (positions[..., int(index), :] for index in atoms)
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    return np.arctan2(
        np.sum(np.cross(b1, v) * w, axis=-1),
        np.sum(v * w, axis=-1),
    )


def pucker(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ring = positions[..., [6, 16, 13, 10, 7], :]
    centered = ring - ring.mean(axis=-2, keepdims=True)
    shifted = np.roll(centered, -1, axis=-2)
    normal = np.sum(np.cross(centered, shifted), axis=-2)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    heights = np.sum(centered * normal[..., None, :], axis=-1)
    index = np.arange(5, dtype=float)
    scale = math.sqrt(2.0 / 5.0)
    cosine = scale * np.sum(heights * np.cos(4.0 * math.pi * index / 5.0), axis=-1)
    sine = -scale * np.sum(heights * np.sin(4.0 * math.pi * index / 5.0), axis=-1)
    return np.hypot(cosine, sine), np.arctan2(sine, cosine)


def transitions(labels: np.ndarray) -> int:
    return int(np.count_nonzero(labels[1:] != labels[:-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "data" / "openmm_probe.npz")
    parser.add_argument("--burnin", type=int, default=1000)
    args = parser.parse_args()
    source = args.input.resolve()
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        positions = np.asarray(data["positions_nm"])
        energies = np.asarray(data["energies_kj_mol"])
        acceptance = np.asarray(data["swap_acceptance"])
        walker = np.asarray(data["walker_index"])
        stereo_margin = np.asarray(data["stereo_min_margin_nm3"])
    if args.burnin < 0 or args.burnin >= len(positions):
        parser.error(f"burnin must lie in [0, {len(positions) - 1}]")

    cold = positions[args.burnin :, 0]
    hot = positions[args.burnin :, -1]
    omega = dihedral(cold, [1, 4, 6, 16])
    ring_chi1 = dihedral(cold, [6, 16, 13, 10])
    amplitude, phase = pucker(cold)
    omega_hot = dihedral(hot, [1, 4, 6, 16])
    ring_chi1_hot = dihedral(hot, [6, 16, 13, 10])

    cis = np.abs(omega) < math.pi / 2.0
    pucker_plus = ring_chi1 >= 0.0
    joint_labels = 2 * cis.astype(int) + pucker_plus.astype(int)
    joint_names = (
        "trans / pucker -",
        "trans / pucker +",
        "cis / pucker -",
        "cis / pucker +",
    )
    counts = np.bincount(joint_labels, minlength=4)
    occupancy = counts / counts.sum()
    omega_hot_cis = np.abs(omega_hot) < math.pi / 2.0
    pucker_hot_plus = ring_chi1_hot >= 0.0

    bins = 72
    histogram, omega_edges, chi_edges = np.histogram2d(
        omega,
        ring_chi1,
        bins=bins,
        range=((-math.pi, math.pi), (-math.pi, math.pi)),
    )
    smooth = gaussian_filter(histogram, sigma=1.0, mode="wrap")
    positive = smooth > 0.0
    free_energy = np.full_like(smooth, np.nan, dtype=float)
    free_energy[positive] = -np.log(smooth[positive] / np.max(smooth))
    free_energy[free_energy > 8.0] = np.nan

    plt.rcParams.update({"font.size": 12})
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    image = axes[0].imshow(
        free_energy.T,
        origin="lower",
        extent=(-math.pi, math.pi, -math.pi, math.pi),
        aspect="auto",
        cmap="viridis_r",
        vmin=0.0,
        vmax=8.0,
    )
    axes[0].axvline(-math.pi / 2.0, color="white", lw=0.8, ls="--")
    axes[0].axvline(math.pi / 2.0, color="white", lw=0.8, ls="--")
    axes[0].axhline(0.0, color="white", lw=0.8, ls=":")
    axes[0].set_xlabel(r"ACE--PRO $\omega$ (rad)")
    axes[0].set_ylabel(r"ring $\chi_1$ (rad)")
    axes[0].set_title("joint free energy")
    figure.colorbar(image, ax=axes[0], label=r"$\Delta F/k_\mathrm{B}T$")

    omega_grid = np.linspace(-math.pi, math.pi, bins + 1)
    omega_density, _ = np.histogram(omega, bins=omega_grid, density=True)
    omega_centers = 0.5 * (omega_grid[:-1] + omega_grid[1:])
    axes[1].plot(omega_centers, omega_density, color="#d62728", lw=2.0)
    axes[1].axvspan(-math.pi / 2.0, math.pi / 2.0, color="#999999", alpha=0.15)
    axes[1].text(0.0, axes[1].get_ylim()[1] * 0.92, "cis", ha="center", va="top")
    axes[1].text(-2.45, axes[1].get_ylim()[1] * 0.92, "trans", ha="center", va="top")
    axes[1].text(2.45, axes[1].get_ylim()[1] * 0.92, "trans", ha="center", va="top")
    axes[1].set_xlabel(r"ACE--PRO $\omega$ (rad)")
    axes[1].set_ylabel("density")
    axes[1].set_title("cis/trans marginal")

    chi_density, _ = np.histogram(ring_chi1, bins=omega_grid, density=True)
    axes[2].plot(omega_centers, chi_density, color="#1f77b4", lw=2.0)
    axes[2].axvline(0.0, color="#555555", lw=0.8, ls="--")
    axes[2].set_xlabel(r"ring $\chi_1$ (rad)")
    axes[2].set_ylabel("density")
    axes[2].set_title("ring-pucker marginal")

    figure.suptitle(
        "N-acetyl-L-proline N-methylamide: raw OpenMM cis/trans and ring puckering",
        y=1.01,
    )
    figure.tight_layout()
    results = HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    figure_path = results / "openmm_multimodality.png"
    if figure_path.exists():
        raise FileExistsError(f"refusing to overwrite existing figure: {figure_path}")
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    metrics = {
        "schema_version": 1,
        "source": str(source.relative_to(HERE)),
        "burnin_rounds": args.burnin,
        "cold_samples": int(len(cold)),
        "temperatures_kelvin": metadata["temperatures_kelvin"],
        "swap_acceptance": acceptance.tolist(),
        "stereo_min_margin_nm3": stereo_margin.tolist(),
        "cis_occupancy_300K": float(cis.mean()),
        "trans_occupancy_300K": float((~cis).mean()),
        "pucker_plus_occupancy_300K": float(pucker_plus.mean()),
        "pucker_minus_occupancy_300K": float((~pucker_plus).mean()),
        "joint_state_occupancy_300K": {
            name: float(value) for name, value in zip(joint_names, occupancy, strict=True)
        },
        "occupied_joint_states_above_1pct": int(np.count_nonzero(occupancy >= 0.01)),
        "cold_cis_trans_changes": transitions(cis),
        "cold_pucker_sign_changes": transitions(pucker_plus),
        "hot_cis_occupancy": float(omega_hot_cis.mean()),
        "hot_cis_trans_changes": transitions(omega_hot_cis),
        "hot_pucker_sign_changes": transitions(pucker_hot_plus),
        "pucker_amplitude_nm_mean": float(amplitude.mean()),
        "pucker_amplitude_nm_min": float(amplitude.min()),
        "pucker_amplitude_nm_max": float(amplitude.max()),
        "pucker_phase_range_rad": [float(phase.min()), float(phase.max())],
        "energy_300K_kj_mol_range": [
            float(energies[args.burnin :, 0].min()),
            float(energies[args.burnin :, 0].max()),
        ],
        "walker_shape": list(walker.shape),
        "figure": str(figure_path.relative_to(HERE)),
    }
    metrics_path = results / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite existing metrics: {metrics_path}")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    report_path = HERE / "REPORT.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    occupied = metrics["occupied_joint_states_above_1pct"]
    report = [
        "# Ac-Pro-NHMe raw OpenMM multimodality probe",
        "",
        "The saved raw-potential replica-exchange trajectory uses "
        f"{metadata['replicas']} temperatures from 300 K to "
        f"{metadata['temperatures_kelvin'][-1]:g} K, "
        f"{metadata['rounds']} rounds, and {metadata['steps_per_round']} OpenMM "
        "steps per round. The first "
        f"{args.burnin} rounds are excluded below.",
        "",
        f"The 300 K slot contains {metrics['cold_samples']} analyzed samples and "
        f"occupies {occupied} cis/trans--pucker joint states above 1%. "
        f"Cis/trans occupancies are {metrics['cis_occupancy_300K']:.4f}/"
        f"{metrics['trans_occupancy_300K']:.4f}; pucker +/- occupancies are "
        f"{metrics['pucker_plus_occupancy_300K']:.4f}/"
        f"{metrics['pucker_minus_occupancy_300K']:.4f}.",
        "",
        "| joint state | 300 K occupancy |",
        "|---|---:|",
    ]
    for name, value in metrics["joint_state_occupancy_300K"].items():
        report.append(f"| {name} | {value:.6f} |")
    report.extend(
        [
            "",
            f"The cold slot records {metrics['cold_cis_trans_changes']} cis/trans "
            f"changes and {metrics['cold_pucker_sign_changes']} pucker-sign changes "
            "after burn-in. These counts include configurations entering through "
            "replica swaps and therefore establish visited support, not kinetic "
            "transition rates. The trajectory preserved the fixed L-proline center "
            f"with minimum signed-volume margin {stereo_margin[0]:.3e} nm^3.",
            "",
            "This enhanced-sampling probe demonstrates nontrivial conformational "
            "support when both cis/trans sectors and both pucker signs are occupied. "
            "Its finite length does not certify fully converged 300 K basin weights.",
            "",
            "![Raw OpenMM multimodality](results/openmm_multimodality.png)",
            "",
        ]
    )
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
