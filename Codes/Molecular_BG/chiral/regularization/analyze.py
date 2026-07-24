#!/usr/bin/env python
"""Analyze, select, and report chiral molecular regularization candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm

from jflows_md.openmm import OpenMM_Potential
from jflows_md.system import Molecular_Bundle

from run import (
    CONFIG_PATH,
    ROOT,
    artifact_complete,
    artifact_path,
    bundle_hashes,
    bundle_path,
    candidate_eligible,
    candidate_order,
    checked_stereocenters,
    condition_label,
    load_config,
    run_metadata,
    sha256,
    start_path,
    stereo_margins,
    tail_exponent,
    verified_jflows_md_source,
)


RESULTS = ROOT / "results"
ANALYZE_PATH = Path(__file__).resolve()


def metric_path(phase: str, molecule: str, rg_param) -> Path:
    return RESULTS / phase / f"{molecule}__{condition_label(rg_param)}.json"


def atomic_write_text(path: Path, content: str, *, replace_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"staging path already exists: {temporary}")
    if path.exists() and not replace_existing:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def metric_fingerprint(metric_without_plots: dict) -> str:
    payload = json.dumps(
        metric_without_plots,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_plot_paths(phase: str, molecule: str, rg_param, fingerprint: str) -> dict:
    stem = f"{molecule}__{condition_label(rg_param)}__{fingerprint[:16]}"
    return {
        kind: str((RESULTS / phase / f"{stem}_{kind}.png").relative_to(ROOT))
        for kind in ("ress", "torsions", "stress")
    }


def load_run(config: dict, phase: str, molecule: str, seed: int, rg_param) -> dict:
    expected = run_metadata(config, phase, molecule, seed, rg_param)
    path = artifact_path(phase, molecule, seed, rg_param)
    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    centers = checked_stereocenters(config, molecule, bundle)
    if not artifact_complete(
        path,
        expected,
        bundle,
        centers,
        config["stereo_min_margin_nm3"],
    ):
        raise ValueError(f"missing, stale, or invalid run artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        burnin = expected["burnin_rounds"]
        return {
            "metadata": expected,
            "positions": np.asarray(data["positions_nm"])[burnin:, 0],
            "all_positions": np.asarray(data["positions_nm"])[burnin:],
            "energies": np.asarray(data["energies_kj_mol"])[burnin:, 0],
            "all_energies": np.asarray(data["energies_kj_mol"])[burnin:],
            "acceptance": np.asarray(data["swap_acceptance"]),
            "walker_index": np.asarray(data["walker_index"])[burnin:],
            "stereo_min_margin_nm3": np.asarray(data["stereo_min_margin_nm3"]),
            "path": str(path.relative_to(ROOT)),
        }


def dihedral(positions: np.ndarray, atoms) -> np.ndarray:
    a, b, c, d = (positions[:, int(index)] for index in atoms)
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    return np.arctan2(
        np.sum(np.cross(b1, v) * w, axis=1),
        np.sum(v * w, axis=1),
    )


def torsions(positions: np.ndarray, definitions) -> np.ndarray:
    return np.stack([dihedral(positions, atoms) for atoms in definitions], axis=1)


def distances(positions: np.ndarray, definitions) -> np.ndarray:
    if not definitions:
        return np.empty((len(positions), 0), dtype=float)
    return np.stack(
        [
            np.linalg.norm(positions[:, int(first)] - positions[:, int(second)], axis=1)
            for first, second in definitions
        ],
        axis=1,
    )


def histogram(values: np.ndarray, bins: int) -> np.ndarray:
    counts = np.histogram(values, bins=bins, range=(-math.pi, math.pi))[0] + 0.5
    return counts / counts.sum()


def js_bits(first: np.ndarray, second: np.ndarray) -> float:
    middle = 0.5 * (first + second)
    return float(
        0.5
        * (
            np.sum(first * np.log2(first / middle))
            + np.sum(second * np.log2(second / middle))
        )
    )


def states(phi: np.ndarray) -> np.ndarray:
    return np.where(
        np.abs(phi) >= 2 * math.pi / 3,
        0,
        np.where(phi >= 0, 1, 2),
    )


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


def split_rhat(chains) -> float:
    half = min(len(chain) for chain in chains) // 2
    if half < 2:
        return float("inf")
    split = np.stack([piece for chain in chains for piece in (chain[:half], chain[-half:])])
    within = np.mean(np.var(split, axis=1, ddof=1))
    if within == 0:
        return 1.0 if np.ptp(np.mean(split, axis=1)) == 0 else 1e9
    between = half * np.var(np.mean(split, axis=1), ddof=1)
    variance = (half - 1) * within / half + between / half
    return float(math.sqrt(variance / within))


def round_trip_count(walker_index: np.ndarray) -> int:
    replicas = walker_index.shape[1]
    slot_by_walker = np.argsort(walker_index, axis=1)
    total = 0
    for walker in range(replicas):
        origin = None
        reached_other = False
        for slot in slot_by_walker[:, walker]:
            if slot not in (0, replicas - 1):
                continue
            endpoint = int(slot)
            if origin is None:
                origin = endpoint
            elif endpoint != origin:
                reached_other = True
            elif reached_other:
                total += 1
                reached_other = False
    return total


def mixing_metrics(runs, phi_chains, thresholds) -> dict:
    observable_groups = [[run["energies"] for run in runs]]
    markov_ess = []
    half_marginal_tv = []
    half_joint_tv = []
    for phi in phi_chains:
        half = len(phi) // 2
        half_joint_tv.append(
            total_variation(joint_distribution(phi[:half]), joint_distribution(phi[half:]))
        )
        first_occ = occupancies(phi[:half])
        second_occ = occupancies(phi[half:])
        half_marginal_tv.extend(0.5 * np.sum(np.abs(first_occ - second_occ), axis=1))
        for torsion_index in range(phi.shape[1]):
            for function in (np.sin, np.cos):
                trace = function(phi[:, torsion_index])
                markov_ess.append(len(trace) / autocorrelation_time(trace))
            state = states(phi[:, [torsion_index]])[:, 0]
            for value in range(3):
                indicator_group = [
                    (states(other[:, [torsion_index]])[:, 0] == value).astype(float)
                    for other in phi_chains
                ]
                observable_groups.append(indicator_group)
    for torsion_index in range(phi_chains[0].shape[1]):
        for function in (np.sin, np.cos):
            observable_groups.append([function(phi[:, torsion_index]) for phi in phi_chains])
    rhats = [split_rhat(group) for group in observable_groups]
    round_trips = [round_trip_count(run["walker_index"]) for run in runs]
    result = {
        "min_markov_ess": float(min(markov_ess)),
        "max_split_rhat": float(max(rhats)),
        "max_half_marginal_tv": float(max(half_marginal_tv, default=0.0)),
        "max_half_joint_tv": float(max(half_joint_tv, default=0.0)),
        "min_swap_acceptance": float(min(np.min(run["acceptance"]) for run in runs)),
        "round_trips_per_seed": round_trips,
    }
    result["pass"] = bool(
        result["min_markov_ess"] >= thresholds["min_markov_ess"]
        and result["max_split_rhat"] < thresholds["max_split_rhat"]
        and result["max_half_marginal_tv"] < thresholds["max_half_marginal_tv"]
        and result["min_swap_acceptance"] >= thresholds["min_swap_acceptance"]
        and min(round_trips) >= thresholds["min_round_trips_per_seed"]
    )
    return result


def structural_metrics(raw_phi, candidate_phi, config: dict) -> dict:
    raw_pool = np.concatenate(raw_phi)
    candidate_pool = np.concatenate(candidate_phi)
    raw_occ_pool = occupancies(raw_pool)
    candidate_occ_pool = occupancies(candidate_pool)
    candidate_js = []
    candidate_occ = []
    for index in range(raw_pool.shape[1]):
        candidate_js.append(
            js_bits(
                histogram(raw_pool[:, index], config["histogram_bins"]),
                histogram(candidate_pool[:, index], config["histogram_bins"]),
            )
        )
        candidate_occ.append(
            float(np.max(np.abs(raw_occ_pool[index] - candidate_occ_pool[index])))
        )
    candidate_joint = total_variation(
        joint_distribution(raw_pool),
        joint_distribution(candidate_pool),
    )

    raw_js = []
    raw_occ = []
    raw_joint = None
    structural_pass = None
    if len(raw_phi) >= 2:
        first_occ = occupancies(raw_phi[0])
        second_occ = occupancies(raw_phi[1])
        for index in range(raw_pool.shape[1]):
            raw_js.append(
                js_bits(
                    histogram(raw_phi[0][:, index], config["histogram_bins"]),
                    histogram(raw_phi[1][:, index], config["histogram_bins"]),
                )
            )
            raw_occ.append(float(np.max(np.abs(first_occ[index] - second_occ[index]))))
        raw_joint = total_variation(
            joint_distribution(raw_phi[0]),
            joint_distribution(raw_phi[1]),
        )
        gate = config["structural"]
        structural_pass = bool(
            all(
                value <= max(gate["max_js_bits_floor"], gate["raw_baseline_multiplier"] * base)
                for value, base in zip(candidate_js, raw_js, strict=True)
            )
            and all(
                value
                <= max(
                    gate["max_occupancy_error_floor"],
                    gate["raw_baseline_multiplier"] * base,
                )
                for value, base in zip(candidate_occ, raw_occ, strict=True)
            )
            and candidate_joint
            <= max(
                gate["max_joint_state_tv_floor"],
                gate["raw_baseline_multiplier"] * raw_joint,
            )
        )
    return {
        "candidate_js_bits": candidate_js,
        "max_js_bits": float(max(candidate_js)),
        "candidate_occupancy_error": candidate_occ,
        "max_occupancy_error": float(max(candidate_occ)),
        "joint_state_tv": candidate_joint,
        "raw_baseline_js_bits": raw_js,
        "raw_max_js_bits": None if not raw_js else float(max(raw_js)),
        "raw_baseline_occupancy_error": raw_occ,
        "raw_max_occupancy_error": None if not raw_occ else float(max(raw_occ)),
        "raw_joint_state_tv": raw_joint,
        "structural_pass": structural_pass,
    }


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
    distance = np.linalg.norm(
        flat[:, pair_idx[:, 0]] - flat[:, pair_idx[:, 1]],
        axis=-1,
    )
    return float(np.mean(distance < floor_nm)), float(np.mean(np.any(distance < floor_nm, axis=1)))


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
                {
                    neighbor
                    for atom in frontier
                    for neighbor in adjacency[atom]
                    if graph[source, neighbor] > depth
                }
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
    distance = np.asarray((0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04))
    frames = np.repeat(reference[None], len(distance), axis=0)
    frames[:, pair[1]] = frames[:, pair[0]] + distance[:, None] * direction
    raw = np.max(np.linalg.norm(physical.forces(frames, platform="CUDA"), axis=2), axis=1)
    candidate = np.max(
        np.linalg.norm(regularized.forces(frames, platform="CUDA"), axis=2),
        axis=1,
    )
    mask = distance <= 0.10
    return {
        "pair": list(pair),
        "distance_nm": distance.tolist(),
        "raw_force_kj_mol_nm": raw.tolist(),
        "candidate_force_kj_mol_nm": candidate.tolist(),
        "median_log10_attenuation_below_0p10_nm": float(
            np.median(np.log10(raw[mask] / candidate[mask]))
        ),
    }


def force_quantiles(potential, frames: np.ndarray, count: int) -> list[float]:
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    force = potential.forces(frames[indices], platform="CUDA")
    maxima = np.max(np.linalg.norm(force, axis=2), axis=1)
    return np.quantile(maxima, (0.5, 0.95, 0.99, 1.0)).tolist()


def evaluate_energies(potential, frames: np.ndarray, kind: str, label: str) -> np.ndarray:
    values = []
    batch = 1000
    method = potential.physical_energy if kind == "physical" else potential.regularized_energy
    for start in range(0, len(frames), batch):
        stop = min(start + batch, len(frames))
        print(f"ENERGY {label} {stop}/{len(frames)}", flush=True)
        values.append(np.asarray(method(frames[start:stop], platform="CUDA")))
    return np.concatenate(values)


def distance_metrics(raw_positions, candidate_positions, definitions, labels) -> list[dict]:
    raw = distances(raw_positions, definitions)
    candidate = distances(candidate_positions, definitions)
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "label": label,
                "raw_quantiles_nm": np.quantile(raw[:, index], (0.05, 0.5, 0.95)).tolist(),
                "candidate_quantiles_nm": np.quantile(
                    candidate[:, index], (0.05, 0.5, 0.95)
                ).tolist(),
                "mean_shift_nm": float(candidate[:, index].mean() - raw[:, index].mean()),
            }
        )
    return rows


def plot_pair(
    metric: dict,
    raw_phi: np.ndarray,
    candidate_phi: np.ndarray,
    config: dict,
) -> None:
    molecule = metric["molecule"]
    final_paths = {name: ROOT / value for name, value in metric["plots"].items()}
    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing plot: {path}")
    temporary_paths = {
        name: path.with_name(f"{path.name}.tmp-{os.getpid()}")
        for name, path in final_paths.items()
    }
    if any(path.exists() for path in temporary_paths.values()):
        raise FileExistsError(f"plot staging path already exists: {temporary_paths}")

    try:
        fig, axis = plt.subplots(figsize=(4.8, 3.5))
        values = np.asarray((metric["raw_target_ress"], metric["reverse_ress"]))
        low = np.asarray(
            (metric["raw_target_ress_ci_low"], metric["reverse_ress_ci_low"])
        )
        high = np.asarray(
            (metric["raw_target_ress_ci_high"], metric["reverse_ress_ci_high"])
        )
        axis.bar((0, 1), values, color=("#2864a8", "#d17a22"), width=0.58)
        axis.errorbar(
            (0, 1),
            values,
            yerr=np.vstack((values - low, high - values)),
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        axis.axhline(
            config["primary_ress_min"], color="black", linestyle=":", linewidth=1
        )
        axis.set_xticks((0, 1), ("candidate to raw", "raw to candidate"))
        axis.set_ylim(0, 1.03)
        axis.set_ylabel("normalized regularization ESS")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(temporary_paths["ress"], dpi=220, format="png")
        plt.close(fig)

        labels = config["molecules"][molecule]["torsion_labels"]
        columns = 2
        rows = math.ceil(len(labels) / columns)
        fig, axes = plt.subplots(
            rows, columns, figsize=(6.6, 2.8 * rows), squeeze=False
        )
        edges = np.linspace(-180, 180, config["histogram_bins"] + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for index, axis in enumerate(axes.ravel()):
            if index >= len(labels):
                axis.axis("off")
                continue
            axis.plot(
                centers,
                histogram(raw_phi[:, index], config["histogram_bins"]),
                label="raw",
            )
            axis.plot(
                centers,
                histogram(candidate_phi[:, index], config["histogram_bins"]),
                label="candidate",
            )
            axis.set_title(labels[index])
            axis.set_xlim(-180, 180)
            axis.set_xticks((-180, -90, 0, 90, 180))
            axis.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False)
        fig.suptitle(config["molecules"][molecule]["display_name"])
        fig.tight_layout()
        fig.savefig(temporary_paths["torsions"], dpi=220, format="png")
        plt.close(fig)

        stress = metric["stress"]
        fig, axis = plt.subplots(figsize=(4.8, 3.5))
        axis.semilogy(
            stress["distance_nm"],
            stress["raw_force_kj_mol_nm"],
            "o-",
            label="raw",
        )
        axis.semilogy(
            stress["distance_nm"],
            stress["candidate_force_kj_mol_nm"],
            "o-",
            label="candidate",
        )
        axis.set_xlabel("compressed distance (nm)")
        axis.set_ylabel("maximum force (kJ mol$^{-1}$ nm$^{-1}$)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(temporary_paths["stress"], dpi=220, format="png")
        plt.close(fig)

        for name in ("ress", "torsions", "stress"):
            temporary_paths[name].replace(final_paths[name])
    finally:
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()


def validate_metric(
    config: dict,
    phase: str,
    molecule: str,
    rg_param: tuple[float, float],
    metric: dict,
) -> None:
    seeds = list(map(int, config["phases"][phase]["seeds"]))
    info = config["molecules"][molecule]
    source = verified_jflows_md_source(config)
    expected_scalars = {
        "schema_version": 1,
        "phase": phase,
        "molecule": molecule,
        "display_name": info["display_name"],
        "dimension": info["dimension"],
        "seeds": seeds,
        "e_kj_mol": rg_param[0],
        "r_nm": rg_param[1],
        "config_sha256": sha256(CONFIG_PATH),
        "analysis_sha256": sha256(ANALYZE_PATH),
        "current_runner_sha256": sha256(ROOT / "run.py"),
        "bundle_sha256": bundle_hashes(config, molecule),
        "jflows_md_commit_observed": source["commit"],
    }
    for key, expected in expected_scalars.items():
        if metric.get(key) != expected:
            raise ValueError(
                f"stale or invalid metric field {key} for {phase} {molecule} "
                f"{rg_param}: {metric.get(key)!r} != {expected!r}"
            )
    for key in ("per_seed_raw_target_ress", "per_seed_reverse_ress"):
        if [item.get("seed") for item in metric.get(key, [])] != seeds:
            raise ValueError(f"metric {key} does not match frozen seeds: {metric.get(key)}")

    payload = dict(metric)
    fingerprint = payload.pop("metric_fingerprint", None)
    plots = payload.pop("plots", None)
    if fingerprint != metric_fingerprint(payload):
        raise ValueError(f"metric fingerprint mismatch for {phase} {molecule} {rg_param}")
    expected_plots = expected_plot_paths(
        phase, molecule, rg_param, fingerprint
    )
    if plots != expected_plots:
        raise ValueError(f"metric plot manifest mismatch: {plots} != {expected_plots}")
    for relative in plots.values():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"metric plot is missing or empty: {path}")

    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    centers = checked_stereocenters(config, molecule, bundle)
    expected_artifacts = {"starts": [], "raw": [], "candidate": []}
    expected_artifact_hashes = {"starts": [], "raw": [], "candidate": []}
    for seed in seeds:
        start = start_path(molecule, seed, False)
        if not start.is_file():
            raise ValueError(f"metric start artifact is missing: {start}")
        expected_artifacts["starts"].append(str(start.relative_to(ROOT)))
        expected_artifact_hashes["starts"].append(sha256(start))
        for key, pair in (("raw", None), ("candidate", rg_param)):
            path = artifact_path(phase, molecule, seed, pair)
            expected = run_metadata(config, phase, molecule, seed, pair)
            if not artifact_complete(
                path,
                expected,
                bundle,
                centers,
                config["stereo_min_margin_nm3"],
            ):
                raise ValueError(f"metric source artifact is stale or invalid: {path}")
            expected_artifacts[key].append(str(path.relative_to(ROOT)))
            expected_artifact_hashes[key].append(sha256(path))
    if metric.get("artifact_paths") != expected_artifacts:
        raise ValueError(
            f"metric artifact manifest mismatch: {metric.get('artifact_paths')} "
            f"!= {expected_artifacts}"
        )
    if metric.get("artifact_sha256") != expected_artifact_hashes:
        raise ValueError(
            f"metric artifact hashes mismatch: {metric.get('artifact_sha256')} "
            f"!= {expected_artifact_hashes}"
        )

    per_seed_gate = phase != "verification" or all(
        item["ci_low"] >= config["primary_ress_min"]
        for item in metric["per_seed_raw_target_ress"]
    )
    expected_primary = bool(
        metric["raw_target_ress_ci_low"] >= config["primary_ress_min"]
        and per_seed_gate
        and metric["tail_second_moment_pass"]
        and metric["energy_crosscheck_pass"]
        and metric["stereo_pass"]
    )
    if metric.get("primary_pass") is not expected_primary:
        raise ValueError(
            f"metric primary_pass is inconsistent: {metric.get('primary_pass')} "
            f"!= {expected_primary}"
        )


def load_metric(config: dict, phase: str, molecule: str, rg_param):
    path = metric_path(phase, molecule, rg_param)
    if not path.is_file():
        return None
    metric = json.loads(path.read_text(encoding="utf-8"))
    validate_metric(config, phase, molecule, rg_param, metric)
    return metric


def evaluate(config: dict, phase: str, molecule: str, rg_param) -> dict:
    if phase not in ("pilot", "verification"):
        raise ValueError(f"unsupported analysis phase: {phase}")
    if not candidate_eligible(config, molecule, rg_param):
        raise ValueError(f"candidate {rg_param} is not eligible for {molecule}")
    existing = load_metric(config, phase, molecule, rg_param)
    if existing is not None:
        print(
            f"SKIP validated metric {metric_path(phase, molecule, rg_param).relative_to(ROOT)}",
            flush=True,
        )
        return existing
    info = config["molecules"][molecule]
    source = verified_jflows_md_source(config)
    seeds = list(map(int, config["phases"][phase]["seeds"]))
    raw_runs = [load_run(config, phase, molecule, seed, None) for seed in seeds]
    candidate_runs = [load_run(config, phase, molecule, seed, rg_param) for seed in seeds]
    raw_phi = [torsions(run["positions"], info["torsions"]) for run in raw_runs]
    candidate_phi = [torsions(run["positions"], info["torsions"]) for run in candidate_runs]
    raw_pool = np.concatenate([run["positions"] for run in raw_runs])
    candidate_pool = np.concatenate([run["positions"] for run in candidate_runs])

    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    regularized = physical.regularized(rg_param)
    forward_chains = []
    reverse_chains = []
    delta = []
    raw_energy_errors = []
    candidate_energy_errors = []
    for seed, raw_run, candidate_run in zip(seeds, raw_runs, candidate_runs, strict=True):
        raw_all = raw_run["all_positions"].reshape((-1, physical.n_atoms, 3))
        candidate_all = candidate_run["all_positions"].reshape((-1, physical.n_atoms, 3))
        raw_evaluated = evaluate_energies(
            physical,
            raw_all,
            "physical",
            f"{phase} {molecule} raw seed={seed}",
        ).reshape(raw_run["all_energies"].shape)
        candidate_evaluated = evaluate_energies(
            regularized,
            candidate_all,
            "regularized",
            f"{phase} {molecule} candidate seed={seed}",
        ).reshape(candidate_run["all_energies"].shape)
        raw_on_candidate = evaluate_energies(
            physical,
            candidate_run["positions"],
            "physical",
            f"{phase} {molecule} raw-on-candidate seed={seed}",
        )
        candidate_on_raw = evaluate_energies(
            regularized,
            raw_run["positions"],
            "regularized",
            f"{phase} {molecule} candidate-on-raw seed={seed}",
        )
        forward_chains.append(
            -physical.beta * (raw_on_candidate - candidate_evaluated[:, 0])
        )
        reverse_chains.append(
            physical.beta * (raw_evaluated[:, 0] - candidate_on_raw)
        )
        delta.append(raw_on_candidate - candidate_evaluated[:, 0])
        raw_energy_errors.append(
            float(np.max(np.abs(raw_evaluated - raw_run["all_energies"])))
        )
        candidate_energy_errors.append(
            float(
                np.max(
                    np.abs(candidate_evaluated - candidate_run["all_energies"])
                )
            )
        )

    replicates = int(config["bootstrap_replicates"])
    seed_offset = int(rg_param[0] * 10 + rg_param[1] * 1000 + info["dimension"])
    forward_low, forward_high, forward_block = bootstrap_ess(
        forward_chains,
        replicates,
        41000 + seed_offset,
    )
    reverse_low, reverse_high, reverse_block = bootstrap_ess(
        reverse_chains,
        replicates,
        51000 + seed_offset,
    )
    per_seed_forward = []
    per_seed_reverse = []
    for index, (seed, forward, reverse) in enumerate(
        zip(seeds, forward_chains, reverse_chains, strict=True)
    ):
        low, high, block = bootstrap_ess(
            [forward], replicates, 61000 + seed_offset + index
        )
        rev_low, rev_high, rev_block = bootstrap_ess(
            [reverse], replicates, 71000 + seed_offset + index
        )
        per_seed_forward.append(
            {
                "seed": seed,
                "ress": importance_ess(forward),
                "ci_low": low,
                "ci_high": high,
                "block_length": block,
            }
        )
        per_seed_reverse.append(
            {
                "seed": seed,
                "ress": importance_ess(reverse),
                "ci_low": rev_low,
                "ci_high": rev_high,
                "block_length": rev_block,
            }
        )

    structural = structural_metrics(raw_phi, candidate_phi, config)
    raw_mixing = mixing_metrics(raw_runs, raw_phi, config["mixing"])
    candidate_mixing = mixing_metrics(candidate_runs, candidate_phi, config["mixing"])
    centers = checked_stereocenters(config, molecule, physical.bundle)
    raw_stereo = np.min(
        np.stack([run["stereo_min_margin_nm3"] for run in raw_runs]),
        axis=0,
    )
    candidate_stereo = np.min(
        np.stack([run["stereo_min_margin_nm3"] for run in candidate_runs]),
        axis=0,
    )
    stereo_pass = bool(
        np.all(raw_stereo > config["stereo_min_margin_nm3"])
        and np.all(candidate_stereo > config["stereo_min_margin_nm3"])
    )
    maximum_energy_error = max(raw_energy_errors + candidate_energy_errors)
    energy_pass = bool(maximum_energy_error <= config["energy_crosscheck_atol_kj_mol"])
    exponent = tail_exponent(config, rg_param)
    tail_threshold = info["dimension"] + config["tail_moment_order"]
    tail_pass = bool(exponent > tail_threshold)
    per_seed_pass = phase == "pilot" or all(
        item["ci_low"] >= config["primary_ress_min"] for item in per_seed_forward
    )
    primary_pass = bool(
        forward_low >= config["primary_ress_min"]
        and per_seed_pass
        and tail_pass
        and energy_pass
        and stereo_pass
    )
    equilibrium_resolved = bool(
        phase == "verification"
        and raw_mixing["pass"]
        and candidate_mixing["pass"]
        and structural["structural_pass"]
    )
    floor_300 = floor_fraction(physical.bundle, candidate_pool, rg_param[1])
    floor_all = floor_fraction(
        physical.bundle,
        np.concatenate([run["all_positions"] for run in candidate_runs]),
        rg_param[1],
    )
    delta_pool = np.concatenate(delta)
    metric = {
        "schema_version": 1,
        "phase": phase,
        "molecule": molecule,
        "display_name": info["display_name"],
        "dimension": info["dimension"],
        "seeds": seeds,
        "e_kj_mol": rg_param[0],
        "r_nm": rg_param[1],
        "tail_exponent_at_400K": exponent,
        "tail_required_exponent": tail_threshold,
        "tail_second_moment_pass": tail_pass,
        "raw_target_ress": importance_ess(np.concatenate(forward_chains)),
        "raw_target_ress_ci_low": forward_low,
        "raw_target_ress_ci_high": forward_high,
        "raw_target_ress_block_length": forward_block,
        "per_seed_raw_target_ress": per_seed_forward,
        "reverse_ress": importance_ess(np.concatenate(reverse_chains)),
        "reverse_ress_ci_low": reverse_low,
        "reverse_ress_ci_high": reverse_high,
        "reverse_ress_block_length": reverse_block,
        "per_seed_reverse_ress": per_seed_reverse,
        **structural,
        "raw_mixing": raw_mixing,
        "candidate_mixing": candidate_mixing,
        "equilibrium_resolved": equilibrium_resolved,
        "raw_stereo_min_margin_nm3": {
            center["label"]: float(value)
            for center, value in zip(centers, raw_stereo, strict=True)
        },
        "candidate_stereo_min_margin_nm3": {
            center["label"]: float(value)
            for center, value in zip(centers, candidate_stereo, strict=True)
        },
        "stereo_pass": stereo_pass,
        "raw_energy_crosscheck_max_abs_kj_mol": float(max(raw_energy_errors)),
        "candidate_energy_crosscheck_max_abs_kj_mol": float(
            max(candidate_energy_errors)
        ),
        "energy_crosscheck_max_abs_kj_mol": float(maximum_energy_error),
        "energy_crosscheck_pass": energy_pass,
        "energy_difference_quantiles_kj_mol": np.quantile(
            delta_pool, (0, 0.5, 0.95, 0.99, 1)
        ).tolist(),
        "materially_modified_fraction": float(np.mean(np.abs(delta_pool) > 0.1)),
        "floor_interaction_fraction_300K": floor_300[0],
        "floor_frame_fraction_300K": floor_300[1],
        "floor_interaction_fraction_all_replicas": floor_all[0],
        "floor_frame_fraction_all_replicas": floor_all[1],
        "raw_force_quantiles_kj_mol_nm": force_quantiles(
            physical, raw_pool, config["force_frames"]
        ),
        "candidate_force_quantiles_kj_mol_nm": force_quantiles(
            regularized, candidate_pool, config["force_frames"]
        ),
        "distance_metrics": distance_metrics(
            raw_pool,
            candidate_pool,
            info.get("distance_pairs", []),
            info.get("distance_labels", []),
        ),
        "stress": stress_metrics(physical, regularized),
        "primary_pass": primary_pass,
        "artifact_paths": {
            "starts": [
                str(start_path(molecule, seed, False).relative_to(ROOT))
                for seed in seeds
            ],
            "raw": [run["path"] for run in raw_runs],
            "candidate": [run["path"] for run in candidate_runs],
        },
        "artifact_sha256": {
            "starts": [sha256(start_path(molecule, seed, False)) for seed in seeds],
            "raw": [sha256(ROOT / run["path"]) for run in raw_runs],
            "candidate": [sha256(ROOT / run["path"]) for run in candidate_runs],
        },
        "config_sha256": sha256(CONFIG_PATH),
        "analysis_sha256": sha256(ANALYZE_PATH),
        "current_runner_sha256": sha256(ROOT / "run.py"),
        "bundle_sha256": bundle_hashes(config, molecule),
        "jflows_md_commit_observed": source["commit"],
    }
    metric["metric_fingerprint"] = metric_fingerprint(metric)
    metric["plots"] = expected_plot_paths(
        phase,
        molecule,
        rg_param,
        metric["metric_fingerprint"],
    )
    plot_pair(
        metric,
        np.concatenate(raw_phi),
        np.concatenate(candidate_phi),
        config,
    )
    path = metric_path(phase, molecule, rg_param)
    atomic_write_text(
        path,
        json.dumps(metric, indent=2, sort_keys=True) + "\n",
        replace_existing=False,
    )
    print(
        f"RESULT {phase} {molecule} {rg_param} "
        f"candidate-to-raw={metric['raw_target_ress']:.9f} "
        f"CI=[{forward_low:.9f},{forward_high:.9f}] "
        f"reverse={metric['reverse_ress']:.9f} primary_pass={primary_pass} "
        f"equilibrium_resolved={equilibrium_resolved}",
        flush=True,
    )
    return metric


def select(config: dict) -> dict:
    source = verified_jflows_md_source(config)
    record = {
        "schema_version": 1,
        "decision_rule": "first frozen candidate passing pilot and verification primary gates",
        "primary_metric": "candidate-to-raw normalized importance ESS",
        "reverse_ess_role": "audit only",
        "config_sha256": sha256(CONFIG_PATH),
        "analysis_sha256": sha256(ANALYZE_PATH),
        "current_runner_sha256": sha256(ROOT / "run.py"),
        "jflows_md_commit_observed": source["commit"],
        "molecules": {},
    }
    for molecule in config["molecules"]:
        history = []
        outcome = None
        for pair in candidate_order(config):
            if not candidate_eligible(config, molecule, pair):
                history.append({"rg_param": list(pair), "status": "tail-ineligible"})
                continue
            pilot = load_metric(config, "pilot", molecule, pair)
            if pilot is None:
                outcome = {
                    "status": "needs-pilot",
                    "rg_param": list(pair),
                    "next_action": "run pilot raw/candidate and evaluate",
                }
                break
            if not pilot["primary_pass"]:
                history.append({"rg_param": list(pair), "status": "pilot-failed"})
                continue
            verification = load_metric(config, "verification", molecule, pair)
            if verification is None:
                outcome = {
                    "status": "pilot-pass",
                    "rg_param": list(pair),
                    "next_action": "run both verification seeds and evaluate",
                }
                break
            if not verification["primary_pass"]:
                history.append(
                    {"rg_param": list(pair), "status": "verification-failed"}
                )
                continue
            outcome = {
                "status": "verified",
                "rg_param": list(pair),
                "next_action": None,
                "equilibrium_resolved": verification["equilibrium_resolved"],
            }
            break
        if outcome is None:
            outcome = {
                "status": "exhausted",
                "rg_param": None,
                "next_action": "no frozen candidate passed",
            }
        outcome["history"] = history
        record["molecules"][molecule] = outcome
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / "selection.json"
    atomic_write_text(
        destination,
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        replace_existing=True,
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return record


def write_final_csv(rows) -> None:
    fields = (
        "molecule",
        "display_name",
        "dimension",
        "e_kj_mol",
        "r_nm",
        "seeds",
        "raw_target_ress",
        "raw_target_ress_ci_low",
        "raw_target_ress_ci_high",
        "reverse_ress",
        "max_js_bits",
        "joint_state_tv",
        "floor_frame_fraction_300K",
        "energy_crosscheck_max_abs_kj_mol",
        "primary_pass",
        "equilibrium_resolved",
    )
    destination = RESULTS / "selected_metrics.csv"
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"staging path already exists: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_report(rows, config: dict) -> None:
    all_equilibrium = all(row["equilibrium_resolved"] for row in rows)
    qualification = (
        "The parameter-fidelity and equilibrium diagnostics both pass."
        if all_equilibrium
        else "The parameter-fidelity gates pass, but at least one conformational equilibrium audit is unresolved; the choices are supported for sharpening fidelity, not as proof that every equilibrium population converged."
    )
    lines = [
        "# Regularization choices for three chiral targets",
        "",
        "## Recommendation",
        "",
        qualification,
        "",
        "| molecule | `rg_param = (e, r)` | verification seeds | candidate-to-raw RESS [95% CI] | equilibrium audit |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | `({row['e_kj_mol']:g}, {row['r_nm']:g})` | "
            f"{len(row['seeds'])} | {row['raw_target_ress']:.9f} "
            f"[{row['raw_target_ress_ci_low']:.9f}, {row['raw_target_ress_ci_high']:.9f}] | "
            f"{'pass' if row['equilibrium_resolved'] else 'unresolved'} |"
        )
    lines.extend(
        [
            "",
            "Candidate-to-raw RESS is the primary sharpening metric. Raw-to-candidate RESS is reported below only as an audit because the singular raw target need not represent the broader regularized tail efficiently.",
            "",
            "The default `(100, 0.10)` is eligible for both alcohols. For 60D ADP it gives tail exponent 60.136 at 400 K and fails the preregistered finite-second-moment requirement `exponent > d + 2`; therefore ADP starts at the stricter `(125, 0.10)` candidate.",
            "",
            "## Numerical audit",
            "",
            "| molecule | reverse RESS | max torsion JS (bits) | joint TV | changed frames | floor-entry frames | max energy error (kJ/mol) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['reverse_ress']:.9f} | "
            f"{row['max_js_bits']:.6f} | {row['joint_state_tv']:.6f} | "
            f"{row['materially_modified_fraction']:.3e} | "
            f"{row['floor_frame_fraction_300K']:.3e} | "
            f"{row['energy_crosscheck_max_abs_kj_mol']:.3e} |"
        )
    lines.extend(
        [
            "",
            "The energy-parity gate is a same-platform CUDA mixed-precision check. Independent CPU re-evaluation differed from CUDA by as much as `4.36e-3 kJ/mol`, so the `5e-4 kJ/mol` gate must not be interpreted as cross-platform bitwise parity.",
            "",
            "## Stereochemical support",
            "",
            "Every saved frame at every replica retained every named signed-volume component with margin above `1e-6 nm^3`. The check covers saved frames at 100-step intervals; it does not prove that unconstrained Cartesian dynamics could not cross and recross between observations.",
            "",
            "| molecule | raw minimum margins (nm^3) | candidate minimum margins (nm^3) |",
            "|---|---|---|",
        ]
    )
    for row in rows:
        raw = ", ".join(
            f"{name}: {value:.3e}"
            for name, value in row["raw_stereo_min_margin_nm3"].items()
        )
        candidate = ", ".join(
            f"{name}: {value:.3e}"
            for name, value in row["candidate_stereo_min_margin_nm3"].items()
        )
        lines.append(f"| {row['display_name']} | {raw} | {candidate} |")
    lines.extend(
        [
            "",
            "## Mixing and structural scope",
            "",
            "| molecule | raw/candidate min Markov ESS | raw/candidate max R-hat | raw/candidate min swap | raw/candidate round trips per seed |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        raw = row["raw_mixing"]
        candidate = row["candidate_mixing"]
        lines.append(
            f"| {row['display_name']} | {raw['min_markov_ess']:.2f}/{candidate['min_markov_ess']:.2f} | "
            f"{raw['max_split_rhat']:.5f}/{candidate['max_split_rhat']:.5f} | "
            f"{raw['min_swap_acceptance']:.3f}/{candidate['min_swap_acceptance']:.3f} | "
            f"{raw['round_trips_per_seed']}/{candidate['round_trips_per_seed']} |"
        )
    lines.extend(
        [
            "",
            "Exact per-seed RESS intervals, raw-seed structural baselines, half-window diagnostics, force quantiles, distance diagnostics, stress arrays, hashes, and artifact paths are in `results/selected_metrics.json`. The frozen protocol and independent review are in `.aris/`.",
            "",
            "## Provenance limitation",
            "",
            "The exact start and run NPZ bytes used by every metric are SHA-256-bound in the metric JSON, and `results/provenance.json` inventories every current data artifact. The local `run.py` source hash was not embedded in the NPZ metadata when trajectories were generated. Its current hash is recorded post hoc, so it identifies the audited validator/runner snapshot but is not cryptographic proof of the historical generation source. The live source commit identifies `jflows_md` behavior, but its source release is 0.5.3 while installed distribution metadata reports 0.5.2. At finalization this entire study tree was absent from the project `HEAD`: code and results were untracked, while `.aris`, logs, and NPZs were ignored, so Git did not provide immutable provenance.",
            "",
            "## Figures",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['display_name']}",
                "",
                f"![Directional RESS]({row['plots']['ress']})",
                "",
                f"![Dihedral comparison]({row['plots']['torsions']})",
                "",
                f"![Short-distance stress test]({row['plots']['stress']})",
                "",
            ]
        )
    atomic_write_text(
        ROOT / "REPORT.md",
        "\n".join(lines) + "\n",
        replace_existing=True,
    )


def finalize(config: dict) -> None:
    source = verified_jflows_md_source(config)
    selection_path = RESULTS / "selection.json"
    if not selection_path.is_file():
        raise ValueError("run `python analyze.py select` first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != 1:
        raise ValueError(f"unsupported selection schema: {selection.get('schema_version')}")
    if selection.get("config_sha256") != sha256(CONFIG_PATH):
        raise ValueError("selection does not match the frozen config")
    if selection.get("analysis_sha256") != sha256(ANALYZE_PATH):
        raise ValueError("selection does not match the current analysis implementation")
    if selection.get("current_runner_sha256") != sha256(ROOT / "run.py"):
        raise ValueError("selection does not match the current runner/validator snapshot")
    if selection.get("jflows_md_commit_observed") != source["commit"]:
        raise ValueError("selection does not match the verified live jflows_md commit")
    if set(selection.get("molecules", {})) != set(config["molecules"]):
        raise ValueError("selection molecule set does not match the frozen config")
    rows = []
    for molecule in config["molecules"]:
        outcome = selection["molecules"][molecule]
        if outcome["status"] != "verified":
            raise ValueError(f"{molecule} is not verified: {outcome}")
        pair = tuple(map(float, outcome["rg_param"]))
        metric = load_metric(config, "verification", molecule, pair)
        if metric is None or not metric["primary_pass"]:
            raise ValueError(f"verified metric missing or failed for {molecule} {pair}")
        rows.append(metric)
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        RESULTS / "selected_metrics.json",
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        replace_existing=True,
    )
    provenance = {
        "config_sha256": sha256(CONFIG_PATH),
        "analysis_sha256": sha256(ANALYZE_PATH),
        "current_runner_sha256": sha256(ROOT / "run.py"),
        "runner_hash_role": "post-hoc audited snapshot; not embedded at trajectory generation",
        "jflows_md_source": source,
        "openmm_version": mm.version.full_version,
        "bundles": {name: bundle_hashes(config, name) for name in config["molecules"]},
        "data_artifacts_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "data").rglob("*.npz"))
        },
    }
    atomic_write_text(
        RESULTS / "provenance.json",
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        replace_existing=True,
    )
    write_final_csv(rows)
    write_report(rows, config)
    print(
        json.dumps(
            {
                row["molecule"]: {
                    "rg_param": [row["e_kj_mol"], row["r_nm"]],
                    "candidate_to_raw_ress": row["raw_target_ress"],
                    "equilibrium_resolved": row["equilibrium_resolved"],
                }
                for row in rows
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("phase", choices=("pilot", "verification"))
    evaluate_parser.add_argument("--molecule", required=True)
    evaluate_parser.add_argument("--e", type=float, required=True)
    evaluate_parser.add_argument("--r", type=float, required=True)
    subparsers.add_parser("select")
    subparsers.add_parser("finalize")
    args = parser.parse_args()

    config = load_config()
    verified_jflows_md_source(config)
    platform = mm.Platform.getPlatformByName(config["platform"])
    platform.setPropertyDefaultValue("Precision", config["precision"])
    if args.command == "evaluate":
        if args.molecule not in config["molecules"]:
            parser.error(f"unknown molecule: {args.molecule}")
        pair = (float(args.e), float(args.r))
        if pair not in candidate_order(config):
            parser.error(f"{pair} is not in the frozen candidate order")
        evaluate(config, args.phase, args.molecule, pair)
    elif args.command == "select":
        select(config)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
