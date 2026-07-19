#!/usr/bin/env python
"""Analyze OpenMM ensembles, select `(e, r)`, and write the final report."""

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

from run import bundle_path, complete, label, load_config, metadata


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_run(config, phase, molecule, seed, rg_param):
    path = ROOT / "data" / phase / f"{molecule}__{label(rg_param)}__seed{seed}.npz"
    rounds = config["extension_rounds"] if phase.endswith("_extended") else config["rounds"]
    expected = metadata(
        config,
        phase,
        molecule,
        seed,
        rg_param,
        rounds,
        config["steps_per_round"],
        config["burnin_rounds"],
    )
    n_atoms = Molecular_Bundle.load(bundle_path(config, molecule)).n_atoms
    if not complete(path, expected, n_atoms, len(config["temperatures_kelvin"])):
        raise ValueError(f"missing, stale, or invalid run artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        record = json.loads(str(data["metadata"]))
        burnin = record["burnin_rounds"]
        return {
            "metadata": record,
            "positions": np.asarray(data["positions_nm"])[burnin:, 0],
            "energies": np.asarray(data["energies_kj_mol"])[burnin:, 0],
            "acceptance": np.asarray(data["swap_acceptance"]),
        }


def dihedrals(positions, carbon_count):
    values = []
    for first in range(carbon_count - 3):
        a, b, c, d = (positions[:, first + i] for i in range(4))
        b0 = -(b - a)
        b1 = c - b
        b2 = d - c
        b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
        v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
        w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
        values.append(
            np.arctan2(
                np.sum(np.cross(b1, v) * w, axis=1),
                np.sum(v * w, axis=1),
            )
        )
    if not values:
        return np.empty((len(positions), 0))
    return np.stack(values, axis=1)


def histogram(values, bins):
    counts = np.histogram(values, bins=bins, range=(-math.pi, math.pi))[0] + 0.5
    return counts / counts.sum()


def js_bits(first, second):
    middle = 0.5 * (first + second)
    return 0.5 * (
        np.sum(first * np.log2(first / middle))
        + np.sum(second * np.log2(second / middle))
    )


def rotamers(phi):
    return np.where(np.abs(phi) >= 2 * math.pi / 3, 0, np.where(phi >= 0, 1, 2))


def occupancies(phi):
    state = rotamers(phi)
    result = np.empty((phi.shape[1], 3))
    for torsion in range(phi.shape[1]):
        counts = np.bincount(state[:, torsion], minlength=3) + 0.5
        result[torsion] = counts / counts.sum()
    return result


def joint_distribution(phi):
    states = rotamers(phi)
    codes = np.sum(states * 3 ** np.arange(states.shape[1]), axis=1)
    counts = np.bincount(codes, minlength=3 ** states.shape[1]) + 0.5
    return counts / counts.sum()


def total_variation(first, second):
    return 0.5 * np.sum(np.abs(first - second))


def importance_ess(log_weights):
    shifted = np.asarray(log_weights) - np.max(log_weights)
    weights = np.exp(shifted)
    return float(weights.sum() ** 2 / (len(weights) * np.sum(weights**2)))


def autocorrelation_time(values):
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    if not np.any(values):
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
    return max(1.0, 1.0 + 2.0 * total)


def split_rhat(chains):
    half = min(len(chain) for chain in chains) // 2
    split = np.stack([part for chain in chains for part in (chain[:half], chain[-half:])])
    within = np.mean(np.var(split, axis=1, ddof=1))
    if within == 0:
        return 1.0 if np.ptp(np.mean(split, axis=1)) == 0 else 1e9
    between = half * np.var(np.mean(split, axis=1), ddof=1)
    variance = (half - 1) * within / half + between / half
    return float(math.sqrt(variance / within))


def mixing(runs, carbon_count):
    torsion_chains = [dihedrals(run["positions"], carbon_count) for run in runs]
    observables = [[run["energies"] for run in runs]]
    neff = []
    for torsion in range(max(0, carbon_count - 3)):
        for function in (np.sin, np.cos):
            chains = [function(phi[:, torsion]) for phi in torsion_chains]
            observables.append(chains)
            neff.extend(len(chain) / autocorrelation_time(chain) for chain in chains)
        state = [rotamers(phi[:, [torsion]])[:, 0] for phi in torsion_chains]
        for value in range(3):
            observables.append([(chain == value).astype(float) for chain in state])
    rhats = [split_rhat(chains) for chains in observables]
    half_tv = []
    for phi in torsion_chains:
        if phi.shape[1]:
            half = len(phi) // 2
            half_tv.append(
                total_variation(joint_distribution(phi[:half]), joint_distribution(phi[half:]))
            )
    swap_min = min(float(np.min(run["acceptance"])) for run in runs)
    min_neff = min(neff) if neff else None
    result = {
        "min_markov_ess": None if min_neff is None else float(min_neff),
        "max_split_rhat": float(max(rhats)),
        "max_half_rotamer_tv": float(max(half_tv, default=0.0)),
        "min_swap_acceptance": swap_min,
    }
    result["pass"] = (
        (result["min_markov_ess"] is None or result["min_markov_ess"] >= 100)
        and result["max_split_rhat"] < 1.05
        and result["max_half_rotamer_tv"] < 0.10
        and result["min_swap_acceptance"] >= 0.10
    )
    return result, torsion_chains


def circular_blocks(values, block_length, rng):
    values = np.asarray(values)
    blocks = math.ceil(len(values) / block_length)
    starts = rng.integers(0, len(values), size=blocks)
    indices = (starts[:, None] + np.arange(block_length)) % len(values)
    return values[indices.ravel()[: len(values)]]


def bootstrap_ess(log_weight_chains, block_length, replicates, seed):
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        sample = np.concatenate(
            [circular_blocks(chain, block_length, rng) for chain in log_weight_chains]
        )
        estimates[index] = importance_ess(sample)
    return np.quantile(estimates, (0.025, 0.975))


def bootstrap_dihedrals(raw_chains, candidate_chains, bins, block_length, replicates, seed):
    if raw_chains[0].shape[1] == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    max_js = np.empty(replicates)
    joint_tv = np.empty(replicates)
    for index in range(replicates):
        raw = np.concatenate(
            [circular_blocks(chain, block_length, rng) for chain in raw_chains]
        )
        candidate = np.concatenate(
            [circular_blocks(chain, block_length, rng) for chain in candidate_chains]
        )
        max_js[index] = max(
            js_bits(histogram(raw[:, i], bins), histogram(candidate[:, i], bins))
            for i in range(raw.shape[1])
        )
        joint_tv[index] = total_variation(
            joint_distribution(raw), joint_distribution(candidate)
        )
    return float(np.quantile(max_js, 0.975)), float(np.quantile(joint_tv, 0.975))


def stress_pair(bundle):
    bonds = [tuple(pair) for pair in bundle.bundle.system["bonds"]]
    adjacency = [set() for _ in range(bundle.n_atoms)]
    for first, second in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    distances = np.full((bundle.n_atoms, bundle.n_atoms), bundle.n_atoms + 1, dtype=int)
    for source in range(bundle.n_atoms):
        distances[source, source] = 0
        frontier = [source]
        for depth in range(1, bundle.n_atoms + 1):
            frontier = sorted({neighbor for atom in frontier for neighbor in adjacency[atom] if distances[source, neighbor] > depth})
            for atom in frontier:
                distances[source, atom] = depth
            if not frontier:
                break
    pairs = [
        (first, second)
        for first in range(bundle.n_atoms)
        for second in range(first + 1, bundle.n_atoms)
        if distances[first, second] > 1
    ]
    return min(pairs, key=lambda pair: (-distances[pair], pair))


def stress_metrics(physical, regularized):
    pair = stress_pair(physical)
    reference = physical.reference_positions_nm
    direction = reference[pair[1]] - reference[pair[0]]
    direction /= np.linalg.norm(direction)
    distances = np.asarray((0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04))
    frames = np.repeat(reference[None], len(distances), axis=0)
    frames[:, pair[1]] = frames[:, pair[0]] + distances[:, None] * direction
    raw_force = physical.forces(frames, platform="CUDA")
    reg_force = regularized.forces(frames, platform="CUDA")
    raw_norm = np.max(np.linalg.norm(raw_force, axis=2), axis=1)
    reg_norm = np.max(np.linalg.norm(reg_force, axis=2), axis=1)
    mask = distances <= 0.10
    attenuation = float(np.median(np.log10(raw_norm[mask] / reg_norm[mask])))
    return pair, distances, raw_norm, reg_norm, attenuation


def sampled_force_quantiles(potential, frames, count):
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    forces = potential.forces(frames[indices], platform="CUDA")
    norms = np.max(np.linalg.norm(forces, axis=2), axis=1)
    return np.quantile(norms, (0.5, 0.95, 0.99, 1.0))


def artifacts_complete(config, phase, molecule, rg_param):
    rounds = config["extension_rounds"] if phase.endswith("_extended") else config["rounds"]
    n_atoms = Molecular_Bundle.load(bundle_path(config, molecule)).n_atoms
    replicas = len(config["temperatures_kelvin"])
    for seed in config["seeds"]:
        for parameter in (None, rg_param):
            expected = metadata(
                config,
                phase,
                molecule,
                seed,
                parameter,
                rounds,
                config["steps_per_round"],
                config["burnin_rounds"],
            )
            path = ROOT / "data" / phase / f"{molecule}__{label(parameter)}__seed{seed}.npz"
            if not complete(path, expected, n_atoms, replicas):
                return False
    return True


def candidate_metrics(config, phase, molecule, rg_param):
    extended = f"{phase}_extended"
    if artifacts_complete(config, extended, molecule, rg_param):
        phase = extended
    seeds = config["seeds"]
    raw_runs = [load_run(config, phase, molecule, seed, None) for seed in seeds]
    candidate_runs = [load_run(config, phase, molecule, seed, rg_param) for seed in seeds]
    carbon_count = config["molecules"][molecule]["carbon_count"]
    raw_mix, raw_phi = mixing(raw_runs, carbon_count)
    candidate_mix, candidate_phi = mixing(candidate_runs, carbon_count)
    raw_pool = np.concatenate([run["positions"] for run in raw_runs])
    candidate_pool = np.concatenate([run["positions"] for run in candidate_runs])
    raw_phi_pool = np.concatenate(raw_phi)
    candidate_phi_pool = np.concatenate(candidate_phi)

    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    regularized = physical.regularized(rg_param)
    beta = physical.beta
    log_weight_chains = []
    crosscheck = []
    delta_pool = []
    for run in candidate_runs:
        raw_energy = physical.physical_energy(run["positions"], platform="CUDA")
        reg_energy = regularized.regularized_energy(run["positions"], platform="CUDA")
        log_weight_chains.append(-beta * (raw_energy - reg_energy))
        delta_pool.append(raw_energy - reg_energy)
        crosscheck.append(float(np.max(np.abs(reg_energy - run["energies"]))))
    raw_log_weight_chains = []
    for run in raw_runs:
        raw_energy = physical.physical_energy(run["positions"], platform="CUDA")
        reg_energy = regularized.regularized_energy(run["positions"], platform="CUDA")
        raw_log_weight_chains.append(beta * (raw_energy - reg_energy))

    block_length = max(
        1,
        math.ceil(2 * max(autocorrelation_time(chain) for chain in log_weight_chains)),
    )
    ess = importance_ess(np.concatenate(log_weight_chains))
    ess_low, ess_high = bootstrap_ess(
        log_weight_chains,
        block_length,
        config["bootstrap_replicates"],
        5300 + int(rg_param[0] * 10 + rg_param[1] * 100),
    )
    reverse_ess = importance_ess(np.concatenate(raw_log_weight_chains))

    per_seed = []
    for index, seed in enumerate(seeds):
        seed_js = [
            js_bits(
                histogram(raw_phi[index][:, torsion], config["histogram_bins"]),
                histogram(candidate_phi[index][:, torsion], config["histogram_bins"]),
            )
            for torsion in range(raw_phi[index].shape[1])
        ]
        seed_occ = []
        if raw_phi[index].shape[1]:
            raw_occupancy = occupancies(raw_phi[index])
            candidate_occupancy = occupancies(candidate_phi[index])
            seed_occ = np.max(np.abs(raw_occupancy - candidate_occupancy), axis=1).tolist()
            seed_joint = total_variation(
                joint_distribution(raw_phi[index]), joint_distribution(candidate_phi[index])
            )
        else:
            seed_joint = 0.0
        per_seed.append(
            {
                "seed": seed,
                "importance_ess": importance_ess(log_weight_chains[index]),
                "max_js_bits": float(max(seed_js, default=0.0)),
                "max_occupancy_error": float(max(seed_occ, default=0.0)),
                "joint_rotamer_tv": float(seed_joint),
            }
        )

    raw_baseline_js = []
    raw_baseline_occ = []
    candidate_js = []
    candidate_occ = []
    for torsion in range(raw_phi_pool.shape[1]):
        raw_baseline_js.append(
            js_bits(
                histogram(raw_phi[0][:, torsion], config["histogram_bins"]),
                histogram(raw_phi[1][:, torsion], config["histogram_bins"]),
            )
        )
        raw_baseline_occ.append(
            float(
                np.max(
                    np.abs(
                        occupancies(raw_phi[0]) [torsion]
                        - occupancies(raw_phi[1])[torsion]
                    )
                )
            )
        )
        candidate_js.append(
            js_bits(
                histogram(raw_phi_pool[:, torsion], config["histogram_bins"]),
                histogram(candidate_phi_pool[:, torsion], config["histogram_bins"]),
            )
        )
        candidate_occ.append(
            float(
                np.max(
                    np.abs(
                        occupancies(raw_phi_pool)[torsion]
                        - occupancies(candidate_phi_pool)[torsion]
                    )
                )
            )
        )
    raw_joint_tv = (
        total_variation(joint_distribution(raw_phi[0]), joint_distribution(raw_phi[1]))
        if raw_phi_pool.shape[1]
        else 0.0
    )
    candidate_joint_tv = (
        total_variation(joint_distribution(raw_phi_pool), joint_distribution(candidate_phi_pool))
        if raw_phi_pool.shape[1]
        else 0.0
    )
    dihedral_block = max(
        1,
        math.ceil(
            2
            * max(
                [autocorrelation_time(np.cos(chain[:, i])) for chain in raw_phi + candidate_phi for i in range(chain.shape[1])]
                or [1.0]
            )
        ),
    )
    max_js_high, joint_tv_high = bootstrap_dihedrals(
        raw_phi,
        candidate_phi,
        config["histogram_bins"],
        dihedral_block,
        config["bootstrap_replicates"],
        6400 + int(rg_param[0] * 10 + rg_param[1] * 100),
    )
    pair, stress_distance, stress_raw, stress_reg, attenuation = stress_metrics(
        physical, regularized
    )
    raw_force = sampled_force_quantiles(physical, raw_pool, config["force_frames"])
    reg_force = sampled_force_quantiles(regularized, candidate_pool, config["force_frames"])
    delta = np.concatenate(delta_pool)

    js_pass = all(
        value <= max(0.03, 2 * baseline)
        for value, baseline in zip(candidate_js, raw_baseline_js, strict=True)
    )
    occupancy_pass = all(
        value <= max(0.05, 2 * baseline)
        for value, baseline in zip(candidate_occ, raw_baseline_occ, strict=True)
    )
    joint_pass = candidate_joint_tv <= max(0.10, 2 * raw_joint_tv)
    eligible = (
        raw_mix["pass"]
        and candidate_mix["pass"]
        and ess_low >= config["importance_ess_min"]
        and js_pass
        and occupancy_pass
        and joint_pass
        and max(crosscheck) <= config["energy_crosscheck_atol_kj_mol"]
    )
    return {
        "molecule": molecule,
        "artifact_phase": phase,
        "dimension": config["molecules"][molecule]["dimension"],
        "e_kj_mol": rg_param[0],
        "r_nm": rg_param[1],
        "eligible": bool(eligible),
        "importance_ess": ess,
        "importance_ess_ci_low": float(ess_low),
        "importance_ess_ci_high": float(ess_high),
        "reverse_importance_ess": reverse_ess,
        "max_js_bits": float(max(candidate_js, default=0.0)),
        "max_js_ci_high": max_js_high,
        "max_occupancy_error": float(max(candidate_occ, default=0.0)),
        "joint_rotamer_tv": float(candidate_joint_tv),
        "joint_rotamer_tv_ci_high": joint_tv_high,
        "raw_max_js_bits": float(max(raw_baseline_js, default=0.0)),
        "raw_max_occupancy_error": float(max(raw_baseline_occ, default=0.0)),
        "raw_joint_rotamer_tv": float(raw_joint_tv),
        "stress_attenuation_log10": attenuation,
        "stress_pair": list(pair),
        "stress_distance_nm": stress_distance.tolist(),
        "stress_raw_force": stress_raw.tolist(),
        "stress_regularized_force": stress_reg.tolist(),
        "raw_force_quantiles": raw_force.tolist(),
        "regularized_force_quantiles": reg_force.tolist(),
        "energy_difference_quantiles": np.quantile(delta, (0, 0.5, 0.95, 0.99, 1)).tolist(),
        "materially_modified_fraction": float(np.mean(np.abs(delta) > 0.1)),
        "energy_crosscheck_max_abs_kj_mol": float(max(crosscheck)),
        "raw_mixing": raw_mix,
        "candidate_mixing": candidate_mix,
        "candidate_js_bits": candidate_js,
        "candidate_occupancy_error": candidate_occ,
        "raw_baseline_js_bits": raw_baseline_js,
        "raw_baseline_occupancy_error": raw_baseline_occ,
        "block_length": block_length,
        "dihedral_block_length": dihedral_block,
        "per_seed": per_seed,
    }, raw_phi_pool, candidate_phi_pool


def write_csv(path, rows):
    fields = [
        "molecule", "dimension", "e_kj_mol", "r_nm", "eligible",
        "importance_ess", "importance_ess_ci_low", "importance_ess_ci_high",
        "reverse_importance_ess", "max_js_bits", "max_js_ci_high",
        "max_occupancy_error", "joint_rotamer_tv", "joint_rotamer_tv_ci_high",
        "raw_max_js_bits", "raw_max_occupancy_error", "raw_joint_rotamer_tv",
        "stress_attenuation_log10", "materially_modified_fraction",
        "energy_crosscheck_max_abs_kj_mol",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_pilot(rows, phi_by_candidate, config):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for e in sorted({row["e_kj_mol"] for row in rows}):
        subset = sorted((row for row in rows if row["e_kj_mol"] == e), key=lambda row: row["r_nm"])
        r = [row["r_nm"] for row in subset]
        axes[0].plot(r, [row["importance_ess_ci_low"] for row in subset], "o-", label=f"e={e:g}")
        axes[1].plot(r, [row["max_js_bits"] for row in subset], "o-")
        axes[2].plot(r, [row["stress_attenuation_log10"] for row in subset], "o-")
    axes[0].axhline(config["importance_ess_min"], color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("raw-target importance ESS (95% lower)")
    axes[1].set_ylabel("maximum dihedral JS (bits)")
    axes[2].set_ylabel(r"stress attenuation $\log_{10}(F_{raw}/F_{rg})$")
    for axis in axes:
        axis.set_xlabel("r (nm)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "pilot_summary.png", dpi=220)
    plt.close(fig)

    selected = next((row for row in rows if row.get("selected")), None)
    if selected is None:
        return
    key = (selected["e_kj_mol"], selected["r_nm"])
    raw_phi, candidate_phi = phi_by_candidate[key]
    fig, axes = plt.subplots(1, raw_phi.shape[1], figsize=(4 * raw_phi.shape[1], 3.2), squeeze=False)
    edges = np.linspace(-180, 180, config["histogram_bins"] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for index, axis in enumerate(axes[0]):
        axis.plot(centers, histogram(raw_phi[:, index], config["histogram_bins"]), label="raw")
        axis.plot(centers, histogram(candidate_phi[:, index], config["histogram_bins"]), label=f"e={key[0]:g}, r={key[1]:g}")
        axis.set_xlabel(f"C{index + 1}-C{index + 2}-C{index + 3}-C{index + 4} (deg)")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("probability per bin")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "pilot_dihedrals.png", dpi=220)
    plt.close(fig)


def pilot(config, e_values=None, r_values=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    phi = {}
    e_values = config["candidate_e_kj_mol"] if e_values is None else e_values
    r_values = config["candidate_r_nm"] if r_values is None else r_values
    for e in e_values:
        for r in r_values:
            print(f"ANALYZE hexane e={e:g} r={r:g}", flush=True)
            row, raw_phi, candidate_phi = candidate_metrics(config, "pilot", "hexane", (e, r))
            rows.append(row)
            phi[(e, r)] = (raw_phi, candidate_phi)
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["stress_attenuation_log10"],
            -row["importance_ess_ci_low"],
            row["max_js_bits"],
            row["e_kj_mol"],
            -row["r_nm"],
        )
    )
    selected = eligible[0] if eligible else None
    if selected is not None:
        selected["selected"] = True
    write_csv(RESULTS / "pilot_metrics.csv", rows)
    (RESULTS / "pilot_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    selection = {
        "status": "candidate_selected" if selected else "no_eligible_candidate",
        "selected": None if selected is None else {"e_kj_mol": selected["e_kj_mol"], "r_nm": selected["r_nm"]},
        "eligible_count": len(eligible),
        "ranking": [
            {"e_kj_mol": row["e_kj_mol"], "r_nm": row["r_nm"]}
            for row in eligible
        ],
    }
    (RESULTS / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    plot_pilot(rows, phi, config)
    print(json.dumps(selection, indent=2))


def plot_verification(rows, phi, rg_param, config):
    torsional = [name for name in ("butane", "pentane", "hexane") if name in phi]
    fig, axes = plt.subplots(len(torsional), 3, figsize=(11, 3.0 * len(torsional)), squeeze=False)
    edges = np.linspace(-180, 180, config["histogram_bins"] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for row_index, molecule in enumerate(torsional):
        raw, candidate = phi[molecule]
        for torsion in range(3):
            axis = axes[row_index, torsion]
            if torsion >= raw.shape[1]:
                axis.axis("off")
                continue
            axis.plot(centers, histogram(raw[:, torsion], config["histogram_bins"]), label="raw")
            axis.plot(centers, histogram(candidate[:, torsion], config["histogram_bins"]), label="regularized")
            axis.set_title(f"{molecule}, torsion {torsion + 1}")
            axis.set_xlabel("dihedral (deg)")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.supylabel("probability per bin")
    fig.tight_layout()
    fig.savefig(RESULTS / "family_dihedrals.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    dimensions = [row["dimension"] for row in rows]
    axis.plot(dimensions, [row["importance_ess"] for row in rows], "o-", label="importance ESS")
    axis.plot(dimensions, [row["importance_ess_ci_low"] for row in rows], "s--", label="95% lower bound")
    axis.axhline(config["importance_ess_min"], color="black", linestyle=":", linewidth=1)
    axis.set_xlabel("internal dimension")
    axis.set_ylabel("raw-target importance ESS")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "family_importance_ess.png", dpi=220)
    plt.close(fig)


def report(rows, rg_param):
    passed = all(row["eligible"] for row in rows)
    e, r = rg_param
    lines = [
        "# Alkane-family regularization report",
        "",
        "## Recommendation",
        "",
    ]
    if passed:
        lines.append(
            f"Use **`e = {e:g} kJ/mol` and `r = {r:g} nm`** as the common regularized target "
            "for methane through n-hexane (54D). The pair passed the preregistered native-OpenMM "
            "distribution-fidelity and regularity-proxy gates on all six molecules."
        )
    else:
        failed = ", ".join(row["molecule"] for row in rows if not row["eligible"])
        lines.append(
            f"The tested pair `e = {e:g} kJ/mol`, `r = {r:g} nm` is **not yet a family-wide "
            f"recommendation** because the preregistered gates failed for: {failed}."
        )
    lines.extend(
        [
            "",
            "The evidence uses only `jflows_md.openmm`: raw and regularized OpenMM systems, native "
            "Langevin seeding, native replica exchange, and native energy/force evaluations. NumPy "
            "is used only for analysis. No flow was trained, so the study establishes target fidelity "
            "and force attenuation relevant to BG rather than directly demonstrating trainability.",
            "",
            "## Why smaller `e` values were excluded",
            "",
            "The logarithmic high-energy tail is normalizable in internal dimension `d` only when "
            "`2e/(k_B T) > d`. At 54D this requires `e > 67.35 kJ/mol` at 300 K and "
            "`e > 89.80 kJ/mol` at the 400 K top replica. Therefore `e=25` and `e=50` are not "
            "valid family-wide targets for this protocol.",
            "",
            "## Family-wide numerical summary",
            "",
            "| molecule | d | ESS | 95% lower | max JS (bits) | joint TV | stress log10 attenuation | pass |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['molecule']} | {row['dimension']} | {row['importance_ess']:.4f} | "
            f"{row['importance_ess_ci_low']:.4f} | {row['max_js_bits']:.4f} | "
            f"{row['joint_rotamer_tv']:.4f} | {row['stress_attenuation_log10']:.3f} | "
            f"{'yes' if row['eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "![Raw and regularized backbone-dihedral distributions](results/family_dihedrals.png)",
            "",
            "![Raw-target importance ESS over dimension](results/family_importance_ess.png)",
            "",
            "## Interpretation",
            "",
            "The dihedral panels are the primary mode-coverage evidence. Importance ESS measures how "
            "well regularized samples reweight back to the raw singular target; it is not a Markov-chain "
            "ESS. The stress attenuation is measured on deliberately compressed nonbonded pairs and is "
            "reported only as a regularity diagnostic, not as an equilibrium observable.",
            "",
            "Full per-seed mixing diagnostics, reverse-overlap ESS, force quantiles, energy differences, "
            "and exact stress-panel arrays are stored in `results/verification_metrics.json`. The frozen "
            "design is in `.aris/EXPERIMENT_PLAN.md` and exact simulation settings are in `config.json`.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


def verification(config, rg_param):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    phi = {}
    for molecule in config["molecules"]:
        print(f"ANALYZE {molecule} e={rg_param[0]:g} r={rg_param[1]:g}", flush=True)
        row, raw_phi, candidate_phi = candidate_metrics(
            config, "verification", molecule, rg_param
        )
        rows.append(row)
        if raw_phi.shape[1]:
            phi[molecule] = (raw_phi, candidate_phi)
    write_csv(RESULTS / "verification_metrics.csv", rows)
    (RESULTS / "verification_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    plot_verification(rows, phi, rg_param, config)
    report(rows, rg_param)
    print(f"family pass: {all(row['eligible'] for row in rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pilot", "verification"))
    parser.add_argument("--e", type=float)
    parser.add_argument("--r", type=float)
    parser.add_argument("--pilot-r", type=float, action="append")
    args = parser.parse_args()
    config = load_config()
    platform = mm.Platform.getPlatformByName(config["platform"])
    platform.setPropertyDefaultValue("Precision", config["precision"])
    if args.phase == "pilot":
        pilot(
            config,
            None if args.e is None else [args.e],
            args.pilot_r,
        )
    else:
        if args.e is None or args.r is None:
            parser.error("verification requires --e and --r")
        verification(config, (args.e, args.r))


if __name__ == "__main__":
    main()
