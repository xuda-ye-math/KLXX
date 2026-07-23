#!/usr/bin/env python
"""Analyze target fidelity and regularization ESS for the achiral study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm

from jflows_md.openmm import OpenMM_Potential
from jflows_md.system import Molecular_Bundle

from run import (
    ROOT,
    artifact_complete,
    artifact_path,
    bundle_hashes,
    bundle_path,
    load_config,
    regularization,
    run_metadata,
    sha256,
)


RESULTS = ROOT / "results"
KB_KJ_MOL_K = 0.00831446261815324


def load_run(config: dict, phase: str, molecule: str, seed: int, rg_param) -> dict:
    path = artifact_path(phase, molecule, seed, rg_param)
    expected = run_metadata(config, phase, molecule, seed, rg_param)
    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    if not artifact_complete(path, expected, bundle.n_atoms):
        raise ValueError(f"missing, stale, or invalid run artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        burnin = expected["burnin_rounds"]
        return {
            "metadata": expected,
            "positions": np.asarray(data["positions_nm"])[burnin:, 0],
            "all_positions": np.asarray(data["positions_nm"])[burnin:],
            "energies": np.asarray(data["energies_kj_mol"])[burnin:, 0],
            "acceptance": np.asarray(data["swap_acceptance"]),
        }


def dihedral(positions: np.ndarray, atoms) -> np.ndarray:
    a, b, c, d = (positions[:, int(index)] for index in atoms)
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v) * w, axis=1), np.sum(v * w, axis=1))


def torsions(positions: np.ndarray, definitions) -> np.ndarray:
    return np.stack([dihedral(positions, atoms) for atoms in definitions], axis=1)


def signed_volume(positions: np.ndarray, atoms) -> np.ndarray:
    center, first, second, third = map(int, atoms)
    a = positions[:, first] - positions[:, center]
    b = positions[:, second] - positions[:, center]
    c = positions[:, third] - positions[:, center]
    return np.sum(a * np.cross(b, c), axis=1)


def histogram(values: np.ndarray, bins: int) -> np.ndarray:
    counts = np.histogram(values, bins=bins, range=(-math.pi, math.pi))[0] + 0.5
    return counts / counts.sum()


def js_bits(first: np.ndarray, second: np.ndarray) -> float:
    middle = 0.5 * (first + second)
    return float(
        0.5
        * (np.sum(first * np.log2(first / middle)) + np.sum(second * np.log2(second / middle)))
    )


def states(phi: np.ndarray) -> np.ndarray:
    return np.where(np.abs(phi) >= 2 * math.pi / 3, 0, np.where(phi >= 0, 1, 2))


def occupancies(phi: np.ndarray) -> np.ndarray:
    state = states(phi)
    result = np.empty((phi.shape[1], 3))
    for index in range(phi.shape[1]):
        counts = np.bincount(state[:, index], minlength=3) + 0.5
        result[index] = counts / counts.sum()
    return result


def joint_distribution(phi: np.ndarray) -> np.ndarray:
    state = states(phi)
    codes = np.sum(state * 3 ** np.arange(state.shape[1]), axis=1)
    counts = np.bincount(codes, minlength=3 ** state.shape[1]) + 0.5
    return counts / counts.sum()


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(first - second)))


def importance_ess(log_weights: np.ndarray) -> float:
    values = np.asarray(log_weights, dtype=float)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return float(weights.sum() ** 2 / (len(weights) * np.sum(weights**2)))


def autocorrelation_time(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    if len(values) < 4 or not np.any(values):
        return 1.0
    size = len(values)
    fft = np.fft.rfft(values, n=2 * size)
    correlation = np.fft.irfft(fft * np.conjugate(fft))[:size]
    correlation /= np.arange(size, 0, -1)
    correlation /= correlation[0]
    total = 0.0
    for index in range(1, size - 1, 2):
        pair = correlation[index] + correlation[index + 1]
        if pair <= 0:
            break
        total += pair
    return max(1.0, float(1.0 + 2.0 * total))


def split_rhat(chains) -> float:
    half = min(len(chain) for chain in chains) // 2
    split = np.stack([piece for chain in chains for piece in (chain[:half], chain[-half:])])
    within = np.mean(np.var(split, axis=1, ddof=1))
    if within == 0:
        return 1.0 if np.ptp(np.mean(split, axis=1)) == 0 else 1e9
    between = half * np.var(np.mean(split, axis=1), ddof=1)
    variance = (half - 1) * within / half + between / half
    return float(math.sqrt(variance / within))


def circular_blocks(values: np.ndarray, block_length: int, rng) -> np.ndarray:
    values = np.asarray(values)
    count = math.ceil(len(values) / block_length)
    starts = rng.integers(0, len(values), size=count)
    indices = (starts[:, None] + np.arange(block_length)) % len(values)
    return values[indices.ravel()[: len(values)]]


def bootstrap_ess(chains, replicates: int, seed: int) -> tuple[float, float, int]:
    block = max(1, math.ceil(2 * max(autocorrelation_time(chain) for chain in chains)))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        sample = np.concatenate([circular_blocks(chain, block, rng) for chain in chains])
        estimates[index] = importance_ess(sample)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high), block


def mixing_metrics(runs, phi_chains, volume_chains, thresholds) -> dict:
    observable_groups = [[run["energies"] for run in runs]]
    markov_ess = []
    half_marginal_tv = []
    half_joint_tv = []
    for chain_index, phi in enumerate(phi_chains):
        half = len(phi) // 2
        half_joint_tv.append(total_variation(joint_distribution(phi[:half]), joint_distribution(phi[half:])))
        first_occ = occupancies(phi[:half])
        second_occ = occupancies(phi[half:])
        half_marginal_tv.extend(0.5 * np.sum(np.abs(first_occ - second_occ), axis=1))
        for torsion_index in range(phi.shape[1]):
            for function in (np.sin, np.cos):
                trace = function(phi[:, torsion_index])
                markov_ess.append(len(trace) / autocorrelation_time(trace))
            state = states(phi[:, [torsion_index]])[:, 0]
            for value in range(3):
                observable_groups.append(
                    [(states(other[:, [torsion_index]])[:, 0] == value).astype(float) for other in phi_chains]
                )
        if volume_chains is not None:
            indicator = (volume_chains[chain_index] > 0).astype(float)
            markov_ess.append(len(indicator) / autocorrelation_time(indicator))
            half_marginal_tv.append(abs(indicator[:half].mean() - indicator[half:].mean()))
    for torsion_index in range(phi_chains[0].shape[1]):
        for function in (np.sin, np.cos):
            observable_groups.append([function(phi[:, torsion_index]) for phi in phi_chains])
    if volume_chains is not None:
        observable_groups.append([(values > 0).astype(float) for values in volume_chains])
    rhats = [split_rhat(group) for group in observable_groups]
    result = {
        "min_markov_ess": float(min(markov_ess)),
        "max_split_rhat": float(max(rhats)),
        "max_half_marginal_tv": float(max(half_marginal_tv, default=0.0)),
        "max_half_joint_tv": float(max(half_joint_tv, default=0.0)),
        "min_swap_acceptance": float(min(np.min(run["acceptance"]) for run in runs)),
    }
    result["pass"] = bool(
        result["min_markov_ess"] >= thresholds["min_markov_ess"]
        and result["max_split_rhat"] < thresholds["max_split_rhat"]
        and result["max_half_marginal_tv"] < thresholds["max_half_marginal_tv"]
        and result["min_swap_acceptance"] >= thresholds["min_swap_acceptance"]
    )
    return result


def stress_pair(bundle) -> tuple[int, int]:
    adjacency = [set() for _ in range(bundle.n_atoms)]
    for first, second in bundle.system["bonds"]:
        adjacency[first].add(second)
        adjacency[second].add(first)
    graph = np.full((bundle.n_atoms, bundle.n_atoms), bundle.n_atoms + 1, dtype=int)
    for source in range(bundle.n_atoms):
        graph[source, source] = 0
        frontier = [source]
        depth = 0
        while frontier:
            depth += 1
            frontier = sorted(
                {neighbor for atom in frontier for neighbor in adjacency[atom] if graph[source, neighbor] > depth}
            )
            for atom in frontier:
                graph[source, atom] = depth
    pairs = [
        (first, second)
        for first in range(bundle.n_atoms)
        for second in range(first + 1, bundle.n_atoms)
        if graph[first, second] > 1
    ]
    return min(pairs, key=lambda pair: (-graph[pair], pair))


def stress_metrics(physical, regularized) -> dict:
    pair = stress_pair(physical.bundle)
    reference = physical.reference_positions_nm
    direction = reference[pair[1]] - reference[pair[0]]
    direction /= np.linalg.norm(direction)
    distances = np.asarray((0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04))
    frames = np.repeat(reference[None], len(distances), axis=0)
    frames[:, pair[1]] = frames[:, pair[0]] + distances[:, None] * direction
    raw = np.max(np.linalg.norm(physical.forces(frames, platform="CUDA"), axis=2), axis=1)
    reg = np.max(np.linalg.norm(regularized.forces(frames, platform="CUDA"), axis=2), axis=1)
    mask = distances <= 0.10
    return {
        "pair": list(pair),
        "distance_nm": distances.tolist(),
        "raw_force_kj_mol_nm": raw.tolist(),
        "regularized_force_kj_mol_nm": reg.tolist(),
        "median_log10_attenuation_below_0p10_nm": float(np.median(np.log10(raw[mask] / reg[mask]))),
    }


def force_quantiles(potential, frames: np.ndarray, count: int) -> list[float]:
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    force = potential.forces(frames[indices], platform="CUDA")
    maxima = np.max(np.linalg.norm(force, axis=2), axis=1)
    return np.quantile(maxima, (0.5, 0.95, 0.99, 1.0)).tolist()


def floor_fraction(bundle, positions: np.ndarray, floor_nm: float) -> tuple[float, float]:
    spec = bundle.system
    pairs = []
    for family in ("pair", "exception"):
        for index, charge, epsilon in zip(
            spec[f"{family}_idx"],
            spec[f"{family}_chargeprod_e2"],
            spec[f"{family}_epsilon_kj_mol"],
            strict=True,
        ):
            if abs(charge) > 0.0 or epsilon > 0.0:
                pairs.append(index)
    pair_idx = np.asarray(pairs, dtype=int)
    flat = positions.reshape((-1, positions.shape[-2], 3))
    distance = np.linalg.norm(flat[:, pair_idx[:, 0]] - flat[:, pair_idx[:, 1]], axis=-1)
    return float(np.mean(distance < floor_nm)), float(np.mean(np.any(distance < floor_nm, axis=1)))


def structural_metrics(raw_phi, reg_phi, bins: int) -> dict:
    raw_pool = np.concatenate(raw_phi)
    reg_pool = np.concatenate(reg_phi)
    raw_js = []
    candidate_js = []
    raw_occ = []
    candidate_occ = []
    occ_raw_pool = occupancies(raw_pool)
    occ_reg_pool = occupancies(reg_pool)
    for index in range(raw_pool.shape[1]):
        raw_js.append(js_bits(histogram(raw_phi[0][:, index], bins), histogram(raw_phi[1][:, index], bins)))
        candidate_js.append(js_bits(histogram(raw_pool[:, index], bins), histogram(reg_pool[:, index], bins)))
        raw_occ.append(float(np.max(np.abs(occupancies(raw_phi[0])[[index]] - occupancies(raw_phi[1])[[index]]))))
        candidate_occ.append(float(np.max(np.abs(occ_raw_pool[index] - occ_reg_pool[index]))))
    raw_joint = total_variation(joint_distribution(raw_phi[0]), joint_distribution(raw_phi[1]))
    candidate_joint = total_variation(joint_distribution(raw_pool), joint_distribution(reg_pool))
    js_pass = all(value <= max(0.03, 2 * base) for value, base in zip(candidate_js, raw_js, strict=True))
    occupancy_pass = all(value <= max(0.05, 2 * base) for value, base in zip(candidate_occ, raw_occ, strict=True))
    joint_pass = candidate_joint <= max(0.10, 2 * raw_joint)
    return {
        "candidate_js_bits": candidate_js,
        "raw_baseline_js_bits": raw_js,
        "candidate_occupancy_error": candidate_occ,
        "raw_baseline_occupancy_error": raw_occ,
        "joint_state_tv": candidate_joint,
        "raw_joint_state_tv": raw_joint,
        "max_js_bits": float(max(candidate_js)),
        "raw_max_js_bits": float(max(raw_js)),
        "max_occupancy_error": float(max(candidate_occ)),
        "raw_max_occupancy_error": float(max(raw_occ)),
        "js_pass": bool(js_pass),
        "occupancy_pass": bool(occupancy_pass),
        "joint_pass": bool(joint_pass),
    }


def molecule_metrics(config: dict, phase: str, molecule: str) -> tuple[dict, np.ndarray, np.ndarray]:
    info = config["molecules"][molecule]
    rg_param = regularization(config)
    raw_runs = [load_run(config, phase, molecule, seed, None) for seed in config["seeds"]]
    reg_runs = [load_run(config, phase, molecule, seed, rg_param) for seed in config["seeds"]]
    raw_phi = [torsions(run["positions"], info["torsions"]) for run in raw_runs]
    reg_phi = [torsions(run["positions"], info["torsions"]) for run in reg_runs]
    volume_atoms = info.get("signed_volume_atoms")
    raw_volume = None if volume_atoms is None else [signed_volume(run["positions"], volume_atoms) for run in raw_runs]
    reg_volume = None if volume_atoms is None else [signed_volume(run["positions"], volume_atoms) for run in reg_runs]
    raw_pool = np.concatenate([run["positions"] for run in raw_runs])
    reg_pool = np.concatenate([run["positions"] for run in reg_runs])
    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    regularized = physical.regularized(rg_param)
    raw_target_weights = []
    reverse_weights = []
    reg_delta = []
    energy_errors = []
    for run in reg_runs:
        raw_energy = physical.physical_energy(run["positions"], platform="CUDA")
        reg_energy = regularized.regularized_energy(run["positions"], platform="CUDA")
        raw_target_weights.append(-physical.beta * (raw_energy - reg_energy))
        reg_delta.append(raw_energy - reg_energy)
        energy_errors.append(float(np.max(np.abs(reg_energy - run["energies"]))))
    for run in raw_runs:
        raw_energy = physical.physical_energy(run["positions"], platform="CUDA")
        reg_energy = regularized.regularized_energy(run["positions"], platform="CUDA")
        reverse_weights.append(physical.beta * (raw_energy - reg_energy))
        energy_errors.append(float(np.max(np.abs(raw_energy - run["energies"]))))
    bootstrap = config["bootstrap_replicates"]
    raw_low, raw_high, raw_block = bootstrap_ess(raw_target_weights, bootstrap, 8100 + info["dimension"])
    rev_low, rev_high, rev_block = bootstrap_ess(reverse_weights, bootstrap, 9100 + info["dimension"])
    raw_ress = importance_ess(np.concatenate(raw_target_weights))
    reverse_ress = importance_ess(np.concatenate(reverse_weights))
    structural = structural_metrics(raw_phi, reg_phi, config["histogram_bins"])
    volume_metrics = None
    volume_pass = True
    if raw_volume is not None:
        raw_positive = [float(np.mean(values > 0)) for values in raw_volume]
        reg_positive = [float(np.mean(values > 0)) for values in reg_volume]
        baseline = abs(raw_positive[0] - raw_positive[1])
        shift = abs(np.mean(raw_positive) - np.mean(reg_positive))
        volume_pass = shift <= max(0.05, 2 * baseline)
        volume_metrics = {
            "label": info["signed_volume_label"],
            "atoms": volume_atoms,
            "raw_positive_fraction_per_seed": raw_positive,
            "regularized_positive_fraction_per_seed": reg_positive,
            "pooled_positive_fraction_shift": float(shift),
            "raw_seed_difference": float(baseline),
            "pass": bool(volume_pass),
        }
    raw_mixing = mixing_metrics(raw_runs, raw_phi, raw_volume, config["mixing"])
    reg_mixing = mixing_metrics(reg_runs, reg_phi, reg_volume, config["mixing"])
    delta = np.concatenate(reg_delta)
    floor_300 = floor_fraction(physical.bundle, reg_pool, rg_param[1])
    floor_all = floor_fraction(
        physical.bundle,
        np.concatenate([run["all_positions"] for run in reg_runs]),
        rg_param[1],
    )
    stress = stress_metrics(physical, regularized)
    fidelity_pass = bool(
        raw_low >= config["importance_ess_min"]
        and rev_low >= config["importance_ess_min"]
        and structural["js_pass"]
        and structural["occupancy_pass"]
        and structural["joint_pass"]
        and volume_pass
        and max(energy_errors) <= config["energy_crosscheck_atol_kj_mol"]
    )
    properness_ratio = 2 * rg_param[0] / (KB_KJ_MOL_K * max(config["temperatures_kelvin"]))
    row = {
        "molecule": molecule,
        "display_name": info["display_name"],
        "dimension": info["dimension"],
        "phase": phase,
        "e_kj_mol": rg_param[0],
        "r_nm": rg_param[1],
        "tail_exponent_at_400K": properness_ratio,
        "tail_proper": bool(properness_ratio > info["dimension"]),
        "raw_target_ress": raw_ress,
        "raw_target_ress_ci_low": raw_low,
        "raw_target_ress_ci_high": raw_high,
        "raw_target_ress_block_length": raw_block,
        "reverse_ress": reverse_ress,
        "reverse_ress_ci_low": rev_low,
        "reverse_ress_ci_high": rev_high,
        "reverse_ress_block_length": rev_block,
        "per_seed_raw_target_ress": [importance_ess(values) for values in raw_target_weights],
        "per_seed_reverse_ress": [importance_ess(values) for values in reverse_weights],
        **structural,
        "signed_volume": volume_metrics,
        "raw_mixing": raw_mixing,
        "regularized_mixing": reg_mixing,
        "energy_difference_quantiles_kj_mol": np.quantile(delta, (0, 0.5, 0.95, 0.99, 1)).tolist(),
        "materially_modified_fraction": float(np.mean(np.abs(delta) > 0.1)),
        "energy_crosscheck_max_abs_kj_mol": float(max(energy_errors)),
        "floor_interaction_fraction_300K": floor_300[0],
        "floor_frame_fraction_300K": floor_300[1],
        "floor_interaction_fraction_all_replicas": floor_all[0],
        "floor_frame_fraction_all_replicas": floor_all[1],
        "raw_force_quantiles_kj_mol_nm": force_quantiles(physical, raw_pool, config["force_frames"]),
        "regularized_force_quantiles_kj_mol_nm": force_quantiles(regularized, reg_pool, config["force_frames"]),
        "stress": stress,
        "fidelity_pass": fidelity_pass,
        "mixing_pass": bool(raw_mixing["pass"] and reg_mixing["pass"]),
    }
    row["pass"] = bool(row["tail_proper"] and row["fidelity_pass"] and row["mixing_pass"])
    return row, np.concatenate(raw_phi), np.concatenate(reg_phi)


def write_csv(rows) -> None:
    fields = (
        "molecule", "display_name", "dimension", "phase", "e_kj_mol", "r_nm", "tail_proper",
        "raw_target_ress", "raw_target_ress_ci_low", "raw_target_ress_ci_high",
        "reverse_ress", "reverse_ress_ci_low", "reverse_ress_ci_high", "max_js_bits",
        "max_occupancy_error", "joint_state_tv", "raw_joint_state_tv",
        "materially_modified_fraction", "floor_frame_fraction_300K",
        "energy_crosscheck_max_abs_kj_mol", "fidelity_pass", "mixing_pass", "pass",
    )
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_ress(rows, config) -> None:
    names = [row["display_name"] for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    for offset, prefix, label, color in (
        (-width / 2, "raw_target_ress", "regularized to raw", "#2864a8"),
        (width / 2, "reverse_ress", "raw to regularized", "#d17a22"),
    ):
        value = np.asarray([row[prefix] for row in rows])
        low = np.asarray([row[f"{prefix}_ci_low"] for row in rows])
        high = np.asarray([row[f"{prefix}_ci_high"] for row in rows])
        axis.bar(x + offset, value, width, color=color, alpha=0.86, label=label)
        axis.errorbar(x + offset, value, yerr=np.vstack((value - low, high - value)), fmt="none", ecolor="black", capsize=3)
    axis.axhline(config["importance_ess_min"], color="black", linestyle=":", linewidth=1)
    axis.set_xticks(x, names)
    axis.set_ylabel("normalized regularization ESS")
    axis.set_ylim(0, 1.03)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "regularization_ess.png", dpi=220)
    plt.close(fig)


def plot_torsions(phi, config) -> None:
    columns = max(len(info["torsions"]) for info in config["molecules"].values())
    fig, axes = plt.subplots(len(phi), columns, figsize=(3.1 * columns, 2.65 * len(phi)), squeeze=False)
    edges = np.linspace(-180, 180, config["histogram_bins"] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for row_index, (molecule, (raw, reg)) in enumerate(phi.items()):
        info = config["molecules"][molecule]
        for column in range(columns):
            axis = axes[row_index, column]
            if column >= raw.shape[1]:
                axis.axis("off")
                continue
            axis.plot(centers, histogram(raw[:, column], config["histogram_bins"]), label="raw")
            axis.plot(centers, histogram(reg[:, column], config["histogram_bins"]), label="regularized")
            axis.set_title(f"{info['display_name']}: {info['torsion_labels'][column]}", fontsize=9)
            axis.set_xlim(-180, 180)
            axis.set_xticks((-180, -90, 0, 90, 180))
            axis.grid(alpha=0.2)
            if row_index == len(phi) - 1:
                axis.set_xlabel("dihedral (degrees)")
            if column == 0:
                axis.set_ylabel("probability per bin")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "torsion_comparison.png", dpi=220)
    plt.close(fig)


def plot_stress(rows) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(10.2, 3.25), sharey=True)
    for axis, row in zip(axes, rows, strict=True):
        stress = row["stress"]
        axis.semilogy(stress["distance_nm"], stress["raw_force_kj_mol_nm"], "o-", label="raw")
        axis.semilogy(stress["distance_nm"], stress["regularized_force_kj_mol_nm"], "o-", label="regularized")
        axis.set_title(row["display_name"])
        axis.set_xlabel("compressed distance (nm)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("maximum force (kJ mol$^{-1}$ nm$^{-1}$)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "stress_forces.png", dpi=220)
    plt.close(fig)


def write_report(rows, config, phase: str) -> None:
    passed = all(row["pass"] for row in rows)
    fidelity = all(row["fidelity_pass"] for row in rows)
    e, r = regularization(config)
    if passed:
        conclusion = (
            f"The evidence supports **`rg_param = ({e:.1f}, {r:.2f})` as a reasonable common choice** "
            "for NMA, glycerol, and neutral diethanolamine under the tested molecular models. All frozen "
            "tail, overlap, structural-fidelity, energy, and mixing gates passed."
        )
    elif fidelity:
        failed = ", ".join(row["display_name"] for row in rows if not row["pass"])
        conclusion = (
            f"The target-fidelity evidence supports `rg_param = ({e:.1f}, {r:.2f})`, but the full "
            f"three-molecule recommendation remains unresolved because frozen mixing checks failed for: {failed}."
        )
    else:
        failed = ", ".join(row["display_name"] for row in rows if not row["fidelity_pass"])
        conclusion = (
            f"Do **not** treat `rg_param = ({e:.1f}, {r:.2f})` as validated for all three molecules: "
            f"one or more frozen fidelity checks failed for {failed}."
        )
    lines = [
        "# Regularization report for three achiral molecules",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"This report uses the `{phase}` artifacts. It covers only NMA, glycerol, and neutral "
        "diethanolamine; it does not cover cyclohexane or repeat the alkane-family study. No normalizing "
        "flow was trained, so the result concerns target fidelity and regularity rather than end-to-end "
        "Boltzmann-generator training.",
        "",
        "## Numerical summary",
        "",
        "| molecule | d | raw-target RESS [95% CI] | reverse RESS [95% CI] | max JS (bits) | joint TV | changed frames | mixing | result |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['dimension']} | {row['raw_target_ress']:.6f} "
            f"[{row['raw_target_ress_ci_low']:.6f}, {row['raw_target_ress_ci_high']:.6f}] | "
            f"{row['reverse_ress']:.6f} [{row['reverse_ress_ci_low']:.6f}, {row['reverse_ress_ci_high']:.6f}] | "
            f"{row['max_js_bits']:.6f} | {row['joint_state_tv']:.6f} | "
            f"{row['materially_modified_fraction']:.3e} | {'pass' if row['mixing_pass'] else 'fail'} | "
            f"{'pass' if row['pass'] else 'unresolved/fail'} |"
        )
    lines.extend([
        "",
        "Raw-target RESS reweights regularized frames to the raw target; reverse RESS reweights raw "
        "frames to the regularized target. Both are normalized importance-sampling ESS values in `[0,1]`, "
        "not trajectory ESS values. Confidence intervals use circular blocks selected from the measured "
        "weight autocorrelation.",
        "",
        "![Directional regularization ESS](results/regularization_ess.png)",
        "",
        "![Raw and regularized torsion distributions](results/torsion_comparison.png)",
        "",
        "## Tail and regularity checks",
        "",
        f"At 400 K, `2e/(k_B T) = {rows[0]['tail_exponent_at_400K']:.3f}`. This exceeds the largest "
        "tested internal dimension, 48, so the regularized tail is proper across the full replica ladder.",
        "",
        "| molecule | 300 K frames entering floor | all-replica frames entering floor | stress attenuation (log10) | energy cross-check (kJ/mol) |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['floor_frame_fraction_300K']:.3e} | "
            f"{row['floor_frame_fraction_all_replicas']:.3e} | "
            f"{row['stress']['median_log10_attenuation_below_0p10_nm']:.3f} | "
            f"{row['energy_crosscheck_max_abs_kj_mol']:.3e} |"
        )
    lines.extend([
        "",
        "The stress panel deliberately compresses one graph-distant pair and is evidence that the "
        "regularizer suppresses singular short-range forces. It is not an equilibrium observable.",
        "",
        "![Short-distance force stress panel](results/stress_forces.png)",
        "",
        "## Conformational and mixing checks",
        "",
        "The configured heavy-atom torsions include the NMA amide torsion, three glycerol torsions, "
        "and four diethanolamine backbone torsions. The achiral signed-volume diagnostics monitor both "
        "central-carbon pseudochirality in glycerol and pyramidal nitrogen inversion in diethanolamine; "
        "they do not restrict either sign.",
        "",
        "| molecule | raw min Markov ESS | regularized min Markov ESS | raw max R-hat | regularized max R-hat | raw/reg min swap | signed-volume shift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        volume = row["signed_volume"]
        shift = "n/a" if volume is None else f"{volume['pooled_positive_fraction_shift']:.5f}"
        lines.append(
            f"| {row['display_name']} | {row['raw_mixing']['min_markov_ess']:.2f} | "
            f"{row['regularized_mixing']['min_markov_ess']:.2f} | "
            f"{row['raw_mixing']['max_split_rhat']:.5f} | {row['regularized_mixing']['max_split_rhat']:.5f} | "
            f"{row['raw_mixing']['min_swap_acceptance']:.3f}/{row['regularized_mixing']['min_swap_acceptance']:.3f} | {shift} |"
        )
    lines.extend([
        "",
        "Exact per-seed RESS, bootstrap block lengths, marginal baselines, half-window diagnostics, "
        "energy-difference quantiles, sampled-force quantiles, and stress arrays are in "
        "`results/metrics.json`. The frozen protocol is `.aris/EXPERIMENT_PLAN.md`; bundle and software "
        "hashes are in `results/provenance.json` and in every trajectory artifact.",
    ])
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(config: dict, phase: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    phi = {}
    for molecule in config["molecules"]:
        print(f"ANALYZE {molecule} phase={phase}", flush=True)
        row, raw_phi, reg_phi = molecule_metrics(config, phase, molecule)
        rows.append(row)
        phi[molecule] = (raw_phi, reg_phi)
        print(
            f"RESULT {molecule} raw-target-RESS={row['raw_target_ress']:.6f} "
            f"reverse-RESS={row['reverse_ress']:.6f} fidelity={row['fidelity_pass']} "
            f"mixing={row['mixing_pass']} pass={row['pass']}",
            flush=True,
        )
    (RESULTS / "metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(rows)
    provenance = {
        "config_sha256": sha256(ROOT / "config.json"),
        "jflows_md_commit": config["jflows_md_commit"],
        "openmm_version": mm.version.full_version,
        "phase": phase,
        "bundles": {name: bundle_hashes(config, name) for name in config["molecules"]},
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_ress(rows, config)
    plot_torsions(phi, config)
    plot_stress(rows)
    write_report(rows, config, phase)
    print(f"three-molecule pass: {all(row['pass'] for row in rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("production", "extended"))
    args = parser.parse_args()
    config = load_config()
    platform = mm.Platform.getPlatformByName(config["platform"])
    platform.setPropertyDefaultValue("Precision", config["precision"])
    analyze(config, args.phase)


if __name__ == "__main__":
    main()
