#!/usr/bin/env python
"""Sequential 1000-step solvent KLXX sweep over CH4 energy cutoffs.

The cutoff order is frozen as c50, c20, c10, c100.  Every optimizer-step ESS
is saved, while the live log reports every tenth step.  Target pools, coverage
pools, audit weights, and trained flows are retained so plotting and auditing
never require retraining.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jflows.train import Monitor
from jflows_md import (
    Mixed_NSF,
    Molecular_Potential,
    mixed_quench_and_temper,
    package_source_sha256,
    train_molecular_forward_KLXX_G,
)

import parameters as P
from audit_target_pools import compare as compare_pools
from train_alkanes import chunked_log_weight, normalized_ess, run_smc_pool


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE / "bundles"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_state(path: Path) -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=path, check=True,
        text=True, capture_output=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status": status}


def save_h5(path: Path, arrays: dict[str, np.ndarray], attributes: dict) -> None:
    with h5py.File(path, "w") as handle:
        for name, value in arrays.items():
            array = np.asarray(value)
            options = {}
            if array.ndim > 0 and array.size > 1024:
                options = {
                    "compression": "gzip",
                    "compression_opts": 1,
                    "shuffle": True,
                }
            handle.create_dataset(name, data=array, **options)
        for name, value in attributes.items():
            handle.attrs[name] = value


def zero_flow(domain):
    return Mixed_NSF(
        jax.random.key(P.FLOW_SEED),
        domain,
        bins=P.BINS,
        transforms=P.TRANSFORMS,
        euclidean_bound=P.EUCLIDEAN_BOUND,
        hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE,
    ).zeros()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_results(path: Path, cases: list[dict]) -> None:
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    figure, axes = plt.subplots(
        2, 1, figsize=(9.0, 8.0),
        gridspec_kw={"height_ratios": (3.2, 1.2)}, constrained_layout=True,
    )
    curve_axis, audit_axis = axes
    steps = np.arange(1, P.SOLVENT_KLXX_STEPS + 1)
    for color, case in zip(colors, cases, strict=True):
        curve = np.asarray(case["ess_history"])
        label = f"c{case['cut_kj_mol']:g}"
        curve_axis.plot(steps, curve, color=color, alpha=0.18, linewidth=0.7)
        curve_axis.plot(
            steps, rolling_mean(curve, 25), color=color, linewidth=2.0,
            label=f"{label} (25-step mean)",
        )
    curve_axis.set(
        xlim=(1, P.SOLVENT_KLXX_STEPS), ylim=(0.0, 1.0),
        xlabel="KLXX optimizer step", ylabel="training batch ESS",
        title="Solvated methane: 1000-step KLXX ESS histories",
    )
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(frameon=False, ncol=2)

    labels = [f"c{case['cut_kj_mol']:g}" for case in cases]
    positions = np.arange(len(cases))
    identity = [case["identity_audit_ess"] for case in cases]
    final = [case["final_audit_ess"] for case in cases]
    width = 0.36
    audit_axis.bar(
        positions - width / 2, identity, width, color="#999999",
        label="identity",
    )
    audit_axis.bar(
        positions + width / 2, final, width, color=colors,
        label="trained KLXX",
    )
    audit_axis.set(
        xticks=positions, xticklabels=labels, ylim=(0.0, 1.0),
        ylabel="1M-sample audit ESS", xlabel="energy cutoff (kJ/mol)",
    )
    audit_axis.grid(axis="y", alpha=0.2)
    audit_axis.legend(frameon=False, ncol=2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def bundle_record() -> tuple[Path, dict]:
    registry = json.loads((BUNDLE_ROOT / "registry.json").read_text())
    record = registry["bundles"]["methane"]
    bundle = BUNDLE_ROOT / record["path"]
    if sha256(bundle / "manifest.json") != record["manifest_sha256"]:
        raise RuntimeError("promoted methane bundle manifest hash mismatch")
    return bundle, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    staging = output.with_name(f".{output.name}.inprogress-{os.getpid()}")
    if staging.exists():
        raise SystemExit(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    log_path = staging / "sweep.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started_total = time.time()
    try:
        bundle, registry_record = bundle_record()
        physical = Molecular_Potential.from_bundle(bundle)
        if physical.dimension != P.DIMENSIONS[0]:
            raise RuntimeError(
                f"methane dimension mismatch: {physical.dimension} != {P.DIMENSIONS[0]}"
            )
        source = physical.source()
        source_train = source.samples(
            jax.random.key(P.SOURCE_TRAIN_SEED), N=P.N_SOURCE_TRAIN
        )
        source_audit = jnp.concatenate(
            [
                source.samples(jax.random.key(seed), N=P.N_AUDIT_BLOCK)
                for seed in P.AUDIT_SEEDS
            ],
            axis=0,
        )
        jax.block_until_ready((source_train, source_audit))
        save_h5(
            staging / "shared_source_samples.h5",
            {
                "source_train": np.asarray(source_train),
                "source_audit": np.asarray(source_audit),
            },
            {
                "source_train_seed": P.SOURCE_TRAIN_SEED,
                "audit_seeds": json.dumps(P.AUDIT_SEEDS),
            },
        )
        log(
            "START solvent KLXX cutoff order="
            f"{P.SOLVENT_KLXX_CUTS_KJ_MOL} steps={P.SOLVENT_KLXX_STEPS} "
            f"batch={P.N_BATCH} lr={P.SOLVENT_KLXX_LR:g} ladder={P.SMC_LADDER}"
        )

        case_summaries: list[dict] = []
        plot_cases: list[dict] = []
        for cut_index, cut in enumerate(P.SOLVENT_KLXX_CUTS_KJ_MOL):
            label = f"c{cut:g}"
            case_path = staging / label
            case_path.mkdir()
            case_started = time.time()
            target = physical.regularized(
                cut,
                energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
                tail_fraction=P.TAIL_FRACTION,
            )
            energy_origin = float(target(physical.reference_internal()[None])[0])
            log(f"[{label}] prepare target energy_origin={energy_origin:.8g}")

            pools = []
            pool_records = []
            pool_arrays = {}
            for role, seed, count in (
                ("train", P.TARGET_POOL_SEEDS[0], P.N_TARGET_TRAIN),
                ("holdout", P.TARGET_POOL_SEEDS[1], P.N_TARGET_HOLDOUT),
            ):
                pool_started = time.time()
                pool, level_ess, acceptance = run_smc_pool(
                    seed=seed,
                    count=count,
                    source=source,
                    target=target,
                    domain=physical.domain,
                    step=1e-2,
                    ladder=P.SMC_LADDER,
                )
                jax.block_until_ready((pool, level_ess, acceptance))
                record = {
                    "role": role,
                    "seed": seed,
                    "samples": count,
                    "ladder": P.SMC_LADDER,
                    "mala_step": 1e-2,
                    "minimum_incremental_ess": float(np.min(level_ess)),
                    "median_mala_acceptance": float(np.median(acceptance)),
                    "minimum_mala_acceptance": float(np.min(acceptance)),
                    "below_nominal_ess_floor": bool(
                        np.min(level_ess) < P.SMC_MIN_ESS
                    ),
                    "wall_seconds": time.time() - pool_started,
                }
                log(
                    f"[{label}] target {role} min_SMC_ESS="
                    f"{record['minimum_incremental_ess']:.5f} "
                    f"median_MALA={record['median_mala_acceptance']:.4f}"
                )
                pools.append(pool)
                pool_records.append(record)
                pool_arrays[f"target_{role}_level_ess"] = level_ess
                pool_arrays[f"target_{role}_mala_acceptance"] = acceptance
            target_train, target_holdout = pools
            pool_audit = compare_pools(
                np.asarray(target_train), np.asarray(target_holdout), target,
                physical.domain.euclidean_dim,
            )
            pool_gates = {
                "maximum_euclidean_ks": P.EUCLIDEAN_KS_MAX,
                "maximum_periodic_marginal_js_bits": P.MARGINAL_JS_BITS_MAX,
                "maximum_adjacent_periodic_pair_js_bits": P.JOINT_ROTAMER_JS_BITS_MAX,
                "target_energy_ks": P.ENERGY_KS_MAX,
            }
            pool_passed = all(
                pool_audit[name] <= threshold
                for name, threshold in pool_gates.items()
            )
            log(
                f"[{label}] independent target pools "
                f"{'PASS' if pool_passed else 'FAIL'} "
                f"max_KS={pool_audit['maximum_euclidean_ks']:.5f} "
                f"energy_KS={pool_audit['target_energy_ks']:.5f}"
            )
            if not pool_passed:
                raise RuntimeError(f"{label} independent target pools disagree")

            identity_log_weight = chunked_log_weight(
                source_audit, source, target, zero_flow(physical.domain), P.CHUNK
            )
            identity_ess = normalized_ess(identity_log_weight)
            log(f"[{label}] identity 1M audit ESS={identity_ess:.8f}")

            hat_initial_key, hat_key = jax.random.split(
                jax.random.key(P.KLXX_HAT_SEED)
            )
            hat_initial = source.samples(hat_initial_key, N=P.N_TARGET_TRAIN)
            hat_started = time.time()
            hat_pool, hat_acceptance = mixed_quench_and_temper(
                hat_key,
                hat_initial,
                target,
                physical.domain,
                melt=P.KLXX_MELT,
                opt_step=P.KLXX_OPT_STEP,
                opt_iters=P.KLXX_OPT_ITERS,
                mc_step=1e-2,
                mc_iters=P.MC_ITERS,
                images=P.WRAPPED_IMAGES,
                chunk=P.CHUNK,
            )
            jax.block_until_ready((hat_pool, hat_acceptance))
            hat_wall = time.time() - hat_started
            log(
                f"[{label}] KLXX coverage pool ready N={hat_pool.shape[0]} "
                f"MALA={float(jnp.mean(hat_acceptance)):.4f} wall={hat_wall:.1f}s"
            )

            def monitor(message: str) -> None:
                log(f"[{label}] {message}")

            train_started = time.time()
            trained, ess, kept, updated = train_molecular_forward_KLXX_G(
                target_train,
                source_train,
                hat_pool,
                source,
                target,
                zero_flow(physical.domain),
                physical.domain,
                n_batch=P.N_BATCH,
                steps=P.SOLVENT_KLXX_STEPS,
                lr=P.SOLVENT_KLXX_LR,
                coeff_lambda=P.KLXX_COEFF_LAMBDA,
                coeff_alpha=P.KLXX_COEFF_ALPHA,
                coeff_beta=P.KLXX_COEFF_BETA,
                mc_step=1e-2,
                mc_iters=P.MC_ITERS,
                images=P.WRAPPED_IMAGES,
                energy_origin=energy_origin,
                e_clip=P.E_CLIP,
                g_clip=P.G_CLIP,
                monitor=Monitor(P.SOLVENT_KLXX_MONITOR_EVERY, "", monitor),
                seed=P.TRAINER_SEED,
                lr_warmup=P.LR_WARMUP,
            )
            jax.block_until_ready((trained, ess, kept, updated))
            jax.effects_barrier()
            train_wall = time.time() - train_started
            ess_history = np.asarray(ess)
            if ess_history.shape != (P.SOLVENT_KLXX_STEPS,):
                raise RuntimeError(
                    f"{label} ESS history has shape {ess_history.shape}, expected "
                    f"({P.SOLVENT_KLXX_STEPS},)"
                )
            if not np.isfinite(ess_history).all():
                raise RuntimeError(f"{label} ESS history contains nonfinite values")

            final_log_weight = chunked_log_weight(
                source_audit, source, target, trained, P.CHUNK
            )
            final_ess = normalized_ess(final_log_weight)
            block_identity = []
            block_final = []
            for block in range(P.N_AUDIT_BLOCKS):
                section = slice(
                    block * P.N_AUDIT_BLOCK, (block + 1) * P.N_AUDIT_BLOCK
                )
                block_identity.append(normalized_ess(identity_log_weight[section]))
                block_final.append(normalized_ess(final_log_weight[section]))

            save_h5(
                case_path / "diagnostics.h5",
                {
                    "target_train": np.asarray(target_train),
                    "target_holdout": np.asarray(target_holdout),
                    "hat_pool": np.asarray(hat_pool),
                    "hat_mala_acceptance": np.asarray(hat_acceptance),
                    "identity_log_weight": identity_log_weight,
                    "final_log_weight": final_log_weight,
                    "ess_history": ess_history,
                    "kept_history": np.asarray(kept),
                    "update_history": np.asarray(updated),
                    "block_identity_ess": np.asarray(block_identity),
                    "block_final_ess": np.asarray(block_final),
                    **pool_arrays,
                },
                {
                    "cut_kj_mol": cut,
                    "energy_scale_kj_mol": P.ENERGY_SCALE_KJ_MOL,
                    "tail_fraction": P.TAIL_FRACTION,
                },
            )
            eqx.tree_serialise_leaves(case_path / "flow.eqx", trained)
            case_summary = {
                "cut_kj_mol": cut,
                "energy_scale_kj_mol": P.ENERGY_SCALE_KJ_MOL,
                "tail_fraction": P.TAIL_FRACTION,
                "energy_origin_reduced": energy_origin,
                "target_pools": pool_records,
                "target_pool_audit": pool_audit,
                "target_pool_gates": pool_gates,
                "target_pool_audit_passed": pool_passed,
                "identity_audit_ess": identity_ess,
                "final_audit_ess": final_ess,
                "audit_ess_delta": final_ess - identity_ess,
                "block_identity_ess": block_identity,
                "block_final_ess": block_final,
                "initial_25_batch_ess_mean": float(np.mean(ess_history[:25])),
                "final_25_batch_ess_mean": float(np.mean(ess_history[-25:])),
                "minimum_batch_ess": float(np.min(ess_history)),
                "maximum_batch_ess": float(np.max(ess_history)),
                "minimum_kept_fraction": float(np.min(np.asarray(kept))),
                "all_updates_applied": bool(np.asarray(updated).all()),
                "hat_mala_acceptance_mean": float(jnp.mean(hat_acceptance)),
                "hat_wall_seconds": hat_wall,
                "training_wall_seconds": train_wall,
                "case_wall_seconds": time.time() - case_started,
                "diagnostics_sha256": sha256(case_path / "diagnostics.h5"),
                "flow_sha256": sha256(case_path / "flow.eqx"),
            }
            write_json(case_path / "summary.json", case_summary)
            case_summaries.append(case_summary)
            plot_cases.append(
                {
                    "cut_kj_mol": cut,
                    "ess_history": ess_history,
                    "identity_audit_ess": identity_ess,
                    "final_audit_ess": final_ess,
                }
            )
            write_json(
                staging / "progress.json",
                {
                    "complete": False,
                    "completed_cutoffs_kj_mol": [
                        item["cut_kj_mol"] for item in case_summaries
                    ],
                    "cases": case_summaries,
                },
            )
            log(
                f"[{label}] DONE batch_ESS first25="
                f"{case_summary['initial_25_batch_ess_mean']:.6f} last25="
                f"{case_summary['final_25_batch_ess_mean']:.6f} "
                f"audit_ESS={identity_ess:.8f}->{final_ess:.8f} "
                f"wall={case_summary['case_wall_seconds']:.1f}s"
            )

        combined = {
            "step": np.arange(1, P.SOLVENT_KLXX_STEPS + 1),
            **{
                f"c{case['cut_kj_mol']:g}_ess": case["ess_history"]
                for case in plot_cases
            },
        }
        save_h5(staging / "ess_curves.h5", combined, {"steps": P.SOLVENT_KLXX_STEPS})
        table = np.column_stack(list(combined.values()))
        np.savetxt(
            staging / "ess_curves.csv",
            table,
            delimiter=",",
            header=",".join(combined),
            comments="",
        )
        plot_results(staging / "ess_curves.png", plot_cases)
        summary = {
            "schema_version": 1,
            "complete": True,
            "molecule": "methane",
            "solvent_model": registry_record["model"]["implicit_solvent"],
            "bundle": registry_record,
            "cutoff_order_kj_mol": list(P.SOLVENT_KLXX_CUTS_KJ_MOL),
            "training": {
                "objective": "KLXX",
                "steps": P.SOLVENT_KLXX_STEPS,
                "n_batch": P.N_BATCH,
                "lr": P.SOLVENT_KLXX_LR,
                "lr_warmup": P.LR_WARMUP,
                "coeff_lambda": P.KLXX_COEFF_LAMBDA,
                "coeff_alpha": P.KLXX_COEFF_ALPHA,
                "coeff_beta": P.KLXX_COEFF_BETA,
                "e_clip": P.E_CLIP,
                "g_clip": P.G_CLIP,
                "monitor_every": P.SOLVENT_KLXX_MONITOR_EVERY,
            },
            "sampling": {
                "target_train": P.N_TARGET_TRAIN,
                "target_holdout": P.N_TARGET_HOLDOUT,
                "source_train": P.N_SOURCE_TRAIN,
                "source_audit": P.N_SELECTION,
                "smc_ladder": P.SMC_LADDER,
                "mala_step": 1e-2,
                "mala_iters": P.MC_ITERS,
                "chunk": P.CHUNK,
            },
            "cases": case_summaries,
            "runtime": {
                "command": sys.argv,
                "wall_seconds": time.time() - started_total,
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "jax_x64_enabled": bool(jax.config.x64_enabled),
                "jflows": git_state(Path("/mnt/projects/jflows")),
                "jflows_md": git_state(Path("/mnt/projects/jflows_md")),
                "jflows_package_source_sha256": package_source_sha256("jflows"),
                "jflows_md_package_source_sha256": package_source_sha256("jflows_md"),
                "driver_sha256": sha256(Path(__file__)),
                "parameters_sha256": sha256(HERE / "parameters.py"),
            },
            "artifacts": {
                "shared_source_samples_sha256": sha256(
                    staging / "shared_source_samples.h5"
                ),
                "ess_curves_h5_sha256": sha256(staging / "ess_curves.h5"),
                "ess_curves_csv_sha256": sha256(staging / "ess_curves.csv"),
                "ess_curves_png_sha256": sha256(staging / "ess_curves.png"),
            },
        }
        write_json(staging / "summary.json", summary)
        (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
        (staging / "progress.json").unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
        print(f"PASS solvent KLXX cutoff sweep: {output}", flush=True)
    except BaseException:
        log("FAILED; staged results retained")
        raise


if __name__ == "__main__":
    main()
