#!/usr/bin/env python
"""Analyze candidate regularizers against the completed raw 300 K controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm

from jflows_md.openmm import OpenMM_Potential
from jflows_md.system import Molecular_Bundle

from analyze import (
    RESULTS,
    bootstrap_ess,
    floor_fraction,
    force_quantiles,
    importance_ess,
    load_run,
    mixing_metrics,
    signed_volume,
    stress_metrics,
    structural_metrics,
    torsions,
)
from run import ROOT, bundle_hashes, bundle_path, load_config, sha256
from run_candidates import (
    CANDIDATE_CONFIG_PATH,
    candidate_metadata,
    candidate_path,
    complete,
    load_candidate_config,
)


def load_candidate(base, candidates, phase, molecule, seed, rg_param):
    path = candidate_path(phase, molecule, seed, rg_param)
    expected = candidate_metadata(base, candidates, phase, molecule, seed, rg_param)
    bundle = Molecular_Bundle.load(bundle_path(base, molecule))
    if not complete(path, expected, bundle.n_atoms):
        raise ValueError(f"missing, stale, or invalid candidate artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        burnin = expected["burnin_rounds"]
        return {
            "metadata": expected,
            "positions": np.asarray(data["positions_nm"])[burnin:, 0],
            "all_positions": np.asarray(data["positions_nm"])[burnin:],
            "energies": np.asarray(data["energies_kj_mol"])[burnin:, 0],
            "acceptance": np.asarray(data["swap_acceptance"]),
        }


def evaluate(base, candidates, phase, molecule, rg_param):
    info = base["molecules"][molecule]
    raw_runs = [load_run(base, "production", molecule, seed, None) for seed in base["seeds"]]
    candidate_runs = [
        load_candidate(base, candidates, phase, molecule, seed, rg_param)
        for seed in candidates[phase]["seeds"]
    ]
    raw_phi = [torsions(run["positions"], info["torsions"]) for run in raw_runs]
    candidate_phi = [torsions(run["positions"], info["torsions"]) for run in candidate_runs]
    volume_atoms = info.get("signed_volume_atoms")
    raw_volume = None if volume_atoms is None else [signed_volume(run["positions"], volume_atoms) for run in raw_runs]
    candidate_volume = None if volume_atoms is None else [signed_volume(run["positions"], volume_atoms) for run in candidate_runs]
    physical = OpenMM_Potential.from_bundle(bundle_path(base, molecule))
    regularized = physical.regularized(rg_param)
    forward_chains = []
    reverse_chains = []
    deltas = []
    energy_errors = []
    for run in candidate_runs:
        raw_energy = physical.physical_energy(run["positions"], platform="CUDA")
        candidate_energy = regularized.regularized_energy(run["positions"], platform="CUDA")
        forward_chains.append(-physical.beta * (raw_energy - candidate_energy))
        deltas.append(raw_energy - candidate_energy)
        energy_errors.append(float(np.max(np.abs(candidate_energy - run["energies"]))))
    for run in raw_runs:
        raw_energy = physical.physical_energy(run["positions"], platform="CUDA")
        candidate_energy = regularized.regularized_energy(run["positions"], platform="CUDA")
        reverse_chains.append(physical.beta * (raw_energy - candidate_energy))
    seed_offset = int(rg_param[0] * 10 + rg_param[1] * 1000 + info["dimension"])
    forward_low, forward_high, forward_block = bootstrap_ess(
        forward_chains, base["bootstrap_replicates"], 18000 + seed_offset
    )
    reverse_low, reverse_high, reverse_block = bootstrap_ess(
        reverse_chains, base["bootstrap_replicates"], 28000 + seed_offset
    )
    structural = structural_metrics(raw_phi, candidate_phi, base["histogram_bins"])
    volume = None
    volume_pass = True
    if raw_volume is not None:
        raw_positive = [float(np.mean(values > 0)) for values in raw_volume]
        candidate_positive = [float(np.mean(values > 0)) for values in candidate_volume]
        baseline = abs(raw_positive[0] - raw_positive[1])
        shift = abs(np.mean(raw_positive) - np.mean(candidate_positive))
        volume_pass = shift <= max(0.05, 2 * baseline)
        volume = {
            "raw_positive_fraction_per_seed": raw_positive,
            "candidate_positive_fraction_per_seed": candidate_positive,
            "pooled_positive_fraction_shift": float(shift),
            "raw_seed_difference": float(baseline),
            "pass": bool(volume_pass),
        }
    gate = candidates[f"{phase}_gates"] if f"{phase}_gates" in candidates else candidates["verification_gates"]
    multiplier = gate["raw_baseline_multiplier"]
    js_pass = structural["max_js_bits"] <= max(gate["max_js_bits_floor"], multiplier * structural["raw_max_js_bits"])
    occupancy_pass = structural["max_occupancy_error"] <= max(
        gate["max_occupancy_error_floor"], multiplier * structural["raw_max_occupancy_error"]
    )
    joint_pass = structural["joint_state_tv"] <= max(
        gate["joint_state_tv_floor"], multiplier * structural["raw_joint_state_tv"]
    )
    candidate_pool = np.concatenate([run["positions"] for run in candidate_runs])
    all_candidate = np.concatenate([run["all_positions"] for run in candidate_runs])
    delta = np.concatenate(deltas)
    floor_300 = floor_fraction(physical.bundle, candidate_pool, rg_param[1])
    floor_all = floor_fraction(physical.bundle, all_candidate, rg_param[1])
    candidate_mixing = mixing_metrics(
        candidate_runs, candidate_phi, candidate_volume, base["mixing"]
    )
    stress = stress_metrics(physical, regularized)
    fidelity_pass = bool(
        forward_low >= gate["directional_ress_ci_low"]
        and reverse_low >= gate["directional_ress_ci_low"]
        and js_pass
        and occupancy_pass
        and joint_pass
        and volume_pass
        and max(energy_errors) <= base["energy_crosscheck_atol_kj_mol"]
    )
    return {
        "molecule": molecule,
        "display_name": info["display_name"],
        "dimension": info["dimension"],
        "phase": phase,
        "e_kj_mol": rg_param[0],
        "r_nm": rg_param[1],
        "raw_target_ress": importance_ess(np.concatenate(forward_chains)),
        "raw_target_ress_ci_low": forward_low,
        "raw_target_ress_ci_high": forward_high,
        "raw_target_ress_block_length": forward_block,
        "per_seed_raw_target_ress": [importance_ess(values) for values in forward_chains],
        "reverse_ress": importance_ess(np.concatenate(reverse_chains)),
        "reverse_ress_ci_low": reverse_low,
        "reverse_ress_ci_high": reverse_high,
        "reverse_ress_block_length": reverse_block,
        "per_seed_reverse_ress": [importance_ess(values) for values in reverse_chains],
        **structural,
        "signed_volume": volume,
        "candidate_mixing": candidate_mixing,
        "energy_difference_quantiles_kj_mol": np.quantile(delta, (0, 0.5, 0.95, 0.99, 1)).tolist(),
        "materially_modified_fraction": float(np.mean(np.abs(delta) > 0.1)),
        "floor_frame_fraction_300K": floor_300[1],
        "floor_frame_fraction_all_replicas": floor_all[1],
        "energy_crosscheck_max_abs_kj_mol": float(max(energy_errors)),
        "candidate_force_quantiles_kj_mol_nm": force_quantiles(
            regularized, candidate_pool, base["force_frames"]
        ),
        "stress": stress,
        "js_gate_pass": bool(js_pass),
        "occupancy_gate_pass": bool(occupancy_pass),
        "joint_gate_pass": bool(joint_pass),
        "fidelity_pass": fidelity_pass,
    }


def select(rows):
    selections = {}
    for molecule in sorted({row["molecule"] for row in rows}):
        eligible = [row for row in rows if row["molecule"] == molecule and row["fidelity_pass"]]
        eligible.sort(key=lambda row: (row["e_kj_mol"], -row["r_nm"]))
        selections[molecule] = None if not eligible else {
            "e_kj_mol": eligible[0]["e_kj_mol"],
            "r_nm": eligible[0]["r_nm"],
            "basis": "lowest e, then largest r, among fidelity-passing candidates",
        }
    return selections


def write_outputs(rows, phase, selections):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"candidate_{phase}"
    (RESULTS / f"{stem}_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = (
        "molecule", "display_name", "dimension", "e_kj_mol", "r_nm",
        "raw_target_ress", "raw_target_ress_ci_low", "raw_target_ress_ci_high",
        "reverse_ress", "reverse_ress_ci_low", "reverse_ress_ci_high",
        "max_js_bits", "max_occupancy_error", "joint_state_tv",
        "materially_modified_fraction", "floor_frame_fraction_300K",
        "energy_crosscheck_max_abs_kj_mol", "fidelity_pass",
    )
    with (RESULTS / f"{stem}_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    record = {"phase": phase, "selection": selections}
    (RESULTS / f"{stem}_selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    molecules = list(dict.fromkeys(row["molecule"] for row in rows))
    fig, axes = plt.subplots(1, len(molecules), figsize=(5.0 * len(molecules), 3.7), squeeze=False)
    for axis, molecule in zip(axes[0], molecules, strict=True):
        subset = [row for row in rows if row["molecule"] == molecule]
        labels = [f"({row['e_kj_mol']:g}, {row['r_nm']:g})" for row in subset]
        x = np.arange(len(subset)); width = 0.34
        for offset, prefix, label, color in (
            (-width / 2, "raw_target_ress", "candidate to raw", "#2864a8"),
            (width / 2, "reverse_ress", "raw to candidate", "#d17a22"),
        ):
            value = np.asarray([row[prefix] for row in subset])
            low = np.asarray([row[f"{prefix}_ci_low"] for row in subset])
            high = np.asarray([row[f"{prefix}_ci_high"] for row in subset])
            axis.bar(x + offset, value, width, color=color, alpha=0.86, label=label)
            axis.errorbar(x + offset, value, yerr=np.vstack((value - low, high - value)), fmt="none", ecolor="black", capsize=3)
        axis.axhline(0.8, color="black", linestyle=":", linewidth=1)
        axis.set_xticks(x, labels, rotation=20)
        axis.set_title(subset[0]["display_name"])
        axis.set_ylim(0, 1.03)
        axis.set_ylabel("normalized regularization ESS")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_ress.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pilot", "verification"))
    parser.add_argument("--molecule", choices=("glycerol", "diethanolamine"))
    parser.add_argument("--e", type=float)
    parser.add_argument("--r", type=float)
    args = parser.parse_args()
    base = load_config()
    candidates = load_candidate_config()
    if sha256(ROOT / "config.json") != candidates["base_config_sha256"]:
        parser.error("base config hash no longer matches candidate_config.json")
    rows = []
    molecules = list(candidates["candidates"]) if args.molecule is None else [args.molecule]
    for molecule in molecules:
        available = [tuple(map(float, pair)) for pair in candidates["candidates"][molecule]]
        if args.e is None and args.r is None:
            if args.phase == "verification":
                parser.error("verification requires --molecule, --e, and --r")
            pairs = available
        else:
            if args.e is None or args.r is None:
                parser.error("--e and --r must be supplied together")
            pairs = [(args.e, args.r)]
            if pairs[0] not in available:
                parser.error(f"{pairs[0]} is not a frozen candidate for {molecule}")
        for rg_param in pairs:
            print(f"ANALYZE candidate-{args.phase} {molecule} {rg_param}", flush=True)
            row = evaluate(base, candidates, args.phase, molecule, rg_param)
            rows.append(row)
            print(
                f"RESULT {molecule} {rg_param} raw-target-RESS={row['raw_target_ress']:.6f} "
                f"reverse-RESS={row['reverse_ress']:.6f} fidelity={row['fidelity_pass']}",
                flush=True,
            )
    selections = select(rows)
    write_outputs(rows, args.phase, selections)
    print(json.dumps(selections, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
