#!/usr/bin/env python
"""Write the final molecule-specific regularization report and audit artifacts."""

from __future__ import annotations

import copy
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from analyze import KB_KJ_MOL_K, RESULTS
from analyze_candidates import evaluate
from run import ROOT, load_config, sha256
from run_candidates import CANDIDATE_CONFIG_PATH, load_candidate_config


SELECTION = {
    "nma": (100.0, 0.15),
    "glycerol": (100.0, 0.10),
    "diethanolamine": (150.0, 0.10),
}


def initial_nma() -> dict:
    rows = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    source = next(row for row in rows if row["molecule"] == "nma")
    return {
        "molecule": "nma",
        "display_name": source["display_name"],
        "dimension": source["dimension"],
        "e_kj_mol": 100.0,
        "r_nm": 0.15,
        "verification_seeds": [3401, 3402],
        "post_burnin_candidate_frames": 8000,
        "raw_target_ress": source["raw_target_ress"],
        "raw_target_ress_ci_low": source["raw_target_ress_ci_low"],
        "raw_target_ress_ci_high": source["raw_target_ress_ci_high"],
        "per_seed_raw_target_ress": source["per_seed_raw_target_ress"],
        "reverse_ress_audit": source["reverse_ress"],
        "reverse_ress_ci_low_audit": source["reverse_ress_ci_low"],
        "reverse_ress_ci_high_audit": source["reverse_ress_ci_high"],
        "max_js_bits": source["max_js_bits"],
        "max_occupancy_error": source["max_occupancy_error"],
        "joint_state_tv": source["joint_state_tv"],
        "materially_modified_fraction": source["materially_modified_fraction"],
        "floor_frame_fraction_300K": source["floor_frame_fraction_300K"],
        "floor_frame_fraction_all_replicas": source["floor_frame_fraction_all_replicas"],
        "energy_crosscheck_max_abs_kj_mol": source["energy_crosscheck_max_abs_kj_mol"],
        "selected_mixing": source["regularized_mixing"],
        "signed_volume": source["signed_volume"],
        "evidence_scope": "two 5000-round verification seeds",
    }


def selected_candidate(base, candidates, molecule, rg_param, seeds) -> dict:
    scoped = copy.deepcopy(candidates)
    scoped["verification"]["seeds"] = list(seeds)
    source = evaluate(base, scoped, "verification", molecule, rg_param)
    return {
        "molecule": molecule,
        "display_name": source["display_name"],
        "dimension": source["dimension"],
        "e_kj_mol": rg_param[0],
        "r_nm": rg_param[1],
        "verification_seeds": list(seeds),
        "post_burnin_candidate_frames": 4000 * len(seeds),
        "raw_target_ress": source["raw_target_ress"],
        "raw_target_ress_ci_low": source["raw_target_ress_ci_low"],
        "raw_target_ress_ci_high": source["raw_target_ress_ci_high"],
        "per_seed_raw_target_ress": source["per_seed_raw_target_ress"],
        "reverse_ress_audit": source["reverse_ress"],
        "reverse_ress_ci_low_audit": source["reverse_ress_ci_low"],
        "reverse_ress_ci_high_audit": source["reverse_ress_ci_high"],
        "max_js_bits": source["max_js_bits"],
        "max_occupancy_error": source["max_occupancy_error"],
        "joint_state_tv": source["joint_state_tv"],
        "materially_modified_fraction": source["materially_modified_fraction"],
        "floor_frame_fraction_300K": source["floor_frame_fraction_300K"],
        "floor_frame_fraction_all_replicas": source["floor_frame_fraction_all_replicas"],
        "energy_crosscheck_max_abs_kj_mol": source["energy_crosscheck_max_abs_kj_mol"],
        "selected_mixing": source["candidate_mixing"],
        "signed_volume": source["signed_volume"],
        "evidence_scope": f"{len(seeds)} completed 5000-round verification seed" + ("" if len(seeds) == 1 else "s"),
    }


def plot_ress(rows) -> None:
    names = [row["display_name"] for row in rows]
    values = np.asarray([row["raw_target_ress"] for row in rows])
    low = np.asarray([row["raw_target_ress_ci_low"] for row in rows])
    high = np.asarray([row["raw_target_ress_ci_high"] for row in rows])
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(7.1, 3.8))
    axis.bar(x, values, width=0.58, color="#2864a8", alpha=0.88)
    axis.errorbar(x, values, yerr=np.vstack((values - low, high - values)), fmt="none", ecolor="black", capsize=4)
    axis.axhline(0.8, color="black", linestyle=":", linewidth=1)
    axis.set_xticks(x, names)
    axis.set_ylabel("candidate-to-raw regularization ESS")
    axis.set_ylim(0.8, 1.002)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "selected_candidate_to_raw_ress.png", dpi=220)
    plt.close(fig)


def write_report(rows) -> None:
    lines = [
        "# Molecule-specific regularization choices for three achiral targets",
        "",
        "## Recommendation",
        "",
        "Use the following molecule-specific OpenMM regularization parameters:",
        "",
        "| molecule | `rg_param = (e, r)` | completed full verification seeds | candidate-to-raw RESS [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | `({row['e_kj_mol']:g}, {row['r_nm']:g})` | "
            f"{len(row['verification_seeds'])} | {row['raw_target_ress']:.9f} "
            f"[{row['raw_target_ress_ci_low']:.9f}, {row['raw_target_ress_ci_high']:.9f}] |"
        )
    lines.extend([
        "",
        "These choices pass the primary overlap test for the intended workflow: samples are drawn from "
        "the broader regularized candidate and sharpened to the raw singular target. Candidate-to-raw "
        "RESS is the normalized importance ESS for that direction. Reverse ESS is retained in "
        "`results/selected_metrics.json` only as an audit diagnostic; it is not a selection gate because "
        "raw samples need not efficiently represent the extra tail mass admitted by the regularized target.",
        "",
        "They also define proper softened targets throughout the replica ladder. At the 400 K top "
        "replica, `2e/(k_B T)` is 60.136 for NMA and glycerol, exceeding dimensions 30 and 36, and "
        "90.204 for neutral diethanolamine, exceeding dimension 48.",
        "",
        "![Candidate-to-raw regularization ESS](results/selected_candidate_to_raw_ress.png)",
        "",
        "## Why the common `(100, 0.15)` choice was split",
        "",
        "The original two-seed 5000-round screen found that `(100, 0.15)` is effectively identical to "
        "the raw NMA target: its 0.15 nm floor was never entered at 300 K. The same floor was too large "
        "for glycerol and neutral diethanolamine. It clipped common repulsive H--H contacts in 33.1% "
        "and 66.7% of their regularized 300 K frames, reducing candidate-to-raw RESS to 0.667695 and "
        "0.332600, respectively.",
        "",
        "Reducing the floor to 0.10 nm removes those sampled floor contacts. Glycerol retains `e=100`; "
        "neutral diethanolamine uses `e=150` so the high-energy softening also remains inactive on the "
        "tested 300 K ensemble. The more conservative `(200, 0.05)` candidate was therefore unnecessary.",
        "",
        "## Selected-target diagnostics at 300 K",
        "",
        "| molecule | modified frames | floor-entry frames | max torsion JS (bits) | max occupancy difference | energy cross-check (kJ/mol) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['materially_modified_fraction']:.3e} | "
            f"{row['floor_frame_fraction_300K']:.3e} | {row['max_js_bits']:.6f} | "
            f"{row['max_occupancy_error']:.6f} | {row['energy_crosscheck_max_abs_kj_mol']:.3e} |"
        )
    lines.extend([
        "",
        "All reported distributional metrics use only the fixed 300 K replica after the 1000-round "
        "burn-in. The 300--400 K ladder is used only to aid mixing. NMA and glycerol each have two "
        "completed 5000-round verification seeds. Diethanolamine has one completed full verification "
        "seed (3401); seed 3402 was stopped at the user's request before any artifact was written.",
        "",
        "## Sampling limitation",
        "",
        "The parameter-overlap conclusion is stronger than the equilibrium-convergence conclusion. "
        "NMA's two seeds disagree on the slowly interconverting amide cis/trans mixture. Glycerol's "
        "selected chains miss the strict torsional mixing gate. The single full diethanolamine chain "
        "has candidate-to-raw RESS essentially one and passes its Markov ESS and split-R-hat checks, "
        "but its first-half/second-half marginal-state TV remains above the frozen threshold. Thus the "
        "regularization choices are supported for sharpening fidelity; formal convergence of every "
        "conformational population is not claimed.",
        "",
        "## Reproducibility",
        "",
        "The original common-pair settings are frozen in `config.json`; the candidate addendum is "
        "`candidate_config.json`. Complete raw and selected trajectories are under `data/production/` "
        "and `data/candidate_verification/`. Exact per-seed values, reverse-direction audit ESS, mixing "
        "statistics, signed-volume diagnostics, and artifact scope are in "
        "`results/selected_metrics.json`. The full protocol history is in `.aris/EXPERIMENT_PLAN.md`.",
    ])
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base = load_config()
    candidates = load_candidate_config()
    rows = [
        initial_nma(),
        selected_candidate(base, candidates, "glycerol", SELECTION["glycerol"], (3401, 3402)),
        selected_candidate(base, candidates, "diethanolamine", SELECTION["diethanolamine"], (3401,)),
    ]
    for row in rows:
        row["tail_exponent_at_top_replica"] = float(
            2 * row["e_kj_mol"] / (KB_KJ_MOL_K * max(base["temperatures_kelvin"]))
        )
        row["tail_proper"] = bool(row["tail_exponent_at_top_replica"] > row["dimension"])
        row["primary_overlap_pass"] = bool(
            row["tail_proper"]
            and
            row["raw_target_ress_ci_low"] >= 0.8
            and row["energy_crosscheck_max_abs_kj_mol"] <= base["energy_crosscheck_atol_kj_mol"]
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "selected_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = (
        "molecule", "display_name", "dimension", "e_kj_mol", "r_nm",
        "verification_seeds", "post_burnin_candidate_frames", "raw_target_ress",
        "raw_target_ress_ci_low", "raw_target_ress_ci_high", "max_js_bits",
        "max_occupancy_error", "joint_state_tv", "materially_modified_fraction",
        "floor_frame_fraction_300K", "energy_crosscheck_max_abs_kj_mol",
        "tail_exponent_at_top_replica", "tail_proper", "primary_overlap_pass",
        "evidence_scope",
    )
    with (RESULTS / "selected_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "decision": "user-selected after candidate pilot",
        "primary_metric": "candidate-to-raw normalized importance ESS",
        "reverse_ess_role": "machine-readable audit diagnostic only",
        "molecules": {
            row["molecule"]: {
                "rg_param": [row["e_kj_mol"], row["r_nm"]],
                "verification_seeds": row["verification_seeds"],
                "primary_overlap_pass": row["primary_overlap_pass"],
            }
            for row in rows
        },
        "base_config_sha256": sha256(ROOT / "config.json"),
        "candidate_config_sha256": sha256(CANDIDATE_CONFIG_PATH),
    }
    (RESULTS / "molecule_specific_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_ress(rows)
    write_report(rows)
    for row in rows:
        print(
            f"{row['display_name']}: ({row['e_kj_mol']:g}, {row['r_nm']:g}) "
            f"candidate-to-raw RESS={row['raw_target_ress']:.9f} "
            f"CI-low={row['raw_target_ress_ci_low']:.9f} "
            f"primary-pass={row['primary_overlap_pass']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
