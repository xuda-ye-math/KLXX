#!/usr/bin/env python
"""Sequential c50 solvent KLXX sweep at 300 K, 600 K, and 900 K."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
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
    Molecular_Potential,
    mixed_quench_and_temper,
    package_source_sha256,
    train_molecular_forward_KLXX_G,
)

import parameters as P
from audit_target_pools import compare as compare_pools
from sweep_ch4_solvent_klxx import (
    bundle_record,
    git_state,
    rolling_mean,
    save_h5,
    sha256,
    write_json,
    zero_flow,
)
from train_alkanes import chunked_log_weight, normalized_ess, run_smc_pool


HERE = Path(__file__).resolve().parent


def plot_results(path: Path, cases: list[dict]) -> None:
    colors = ("#4D4D4D", "#0072B2", "#D55E00")
    figure, axes = plt.subplots(
        2, 1, figsize=(9.0, 8.0),
        gridspec_kw={"height_ratios": (3.2, 1.2)}, constrained_layout=True,
    )
    curve_axis, audit_axis = axes
    steps = np.arange(1, P.SOLVENT_KLXX_STEPS + 1)
    for color, case in zip(colors, cases, strict=True):
        history = np.asarray(case["ess_history"])
        label = f"{case['temperature_kelvin']:g} K"
        curve_axis.plot(steps, history, color=color, alpha=0.18, linewidth=0.7)
        curve_axis.plot(
            steps,
            rolling_mean(history, 25),
            color=color,
            linewidth=2.0,
            label=f"{label} (25-step mean)",
        )
    curve_axis.set(
        xlim=(1, P.SOLVENT_KLXX_STEPS),
        ylim=(0.0, 1.0),
        xlabel="KLXX optimizer step",
        ylabel="training batch ESS",
        title="Solvated methane c50: target-temperature ESS histories",
    )
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(frameon=False, ncol=3)

    labels = [f"{case['temperature_kelvin']:g} K" for case in cases]
    positions = np.arange(len(cases))
    width = 0.36
    audit_axis.bar(
        positions - width / 2,
        [case["identity_audit_ess"] for case in cases],
        width,
        color="#999999",
        label="identity",
    )
    audit_axis.bar(
        positions + width / 2,
        [case["final_audit_ess"] for case in cases],
        width,
        color=colors,
        label="trained KLXX",
    )
    audit_axis.set(
        xticks=positions,
        xticklabels=labels,
        ylim=(0.0, 1.0),
        ylabel="1M-sample audit ESS",
        xlabel="target temperature",
    )
    audit_axis.grid(axis="y", alpha=0.2)
    audit_axis.legend(frameon=False, ncol=2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


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
        bundle_temperature = Molecular_Potential.from_bundle(bundle).temperature_kelvin
        log(
            "START solvent c50 KLXX temperature order="
            f"{P.SOLVENT_KLXX_TEMPERATURES_K} steps={P.SOLVENT_KLXX_STEPS} "
            f"batch={P.N_BATCH} lr={P.SOLVENT_KLXX_LR:g}"
        )

        summaries = []
        plot_cases = []
        for temperature in P.SOLVENT_KLXX_TEMPERATURES_K:
            label = f"T{temperature:g}K"
            case_path = staging / label
            case_path.mkdir()
            case_started = time.time()
            physical = Molecular_Potential.from_bundle(
                bundle, temperature_kelvin=temperature
            )
            temperature_scale = temperature / bundle_temperature
            mala_step = 1e-2 * temperature_scale
            pool_ladder = P.SMC_LADDER
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
            target = physical.regularized(
                P.SOLVENT_KLXX_TEMPERATURE_CUT_KJ_MOL,
                energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
                tail_fraction=P.TAIL_FRACTION,
            )
            energy_origin = float(target(physical.reference_internal()[None])[0])
            log(
                f"[{label}] prepare c50 target beta={float(physical.beta):.8g} "
                f"energy_origin={energy_origin:.8g} "
                f"source_variance_scale={temperature_scale:g} "
                f"MALA_step={mala_step:g} SMC_ladder={pool_ladder}"
            )

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
                    step=mala_step,
                    ladder=pool_ladder,
                )
                jax.block_until_ready((pool, level_ess, acceptance))
                record = {
                    "role": role,
                    "seed": seed,
                    "samples": count,
                    "ladder": pool_ladder,
                    "mala_step": mala_step,
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
                np.asarray(target_train),
                np.asarray(target_holdout),
                target,
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
                mc_step=mala_step,
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
                mc_step=mala_step,
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
                raise RuntimeError(f"{label} returned incomplete ESS history")
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
                    "source_train": np.asarray(source_train),
                    "source_audit": np.asarray(source_audit),
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
                    "temperature_kelvin": temperature,
                    "cut_kj_mol": P.SOLVENT_KLXX_TEMPERATURE_CUT_KJ_MOL,
                },
            )
            eqx.tree_serialise_leaves(case_path / "flow.eqx", trained)
            summary = {
                "temperature_kelvin": temperature,
                "source_temperature_kelvin": temperature,
                "source_variance_scale_from_bundle": temperature_scale,
                "mala_step": mala_step,
                "beta_mol_per_kj": float(physical.beta),
                "cut_kj_mol": P.SOLVENT_KLXX_TEMPERATURE_CUT_KJ_MOL,
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
            write_json(case_path / "summary.json", summary)
            summaries.append(summary)
            plot_cases.append(
                {
                    "temperature_kelvin": temperature,
                    "ess_history": ess_history,
                    "identity_audit_ess": identity_ess,
                    "final_audit_ess": final_ess,
                }
            )
            write_json(
                staging / "progress.json",
                {
                    "complete": False,
                    "completed_temperatures_kelvin": [
                        item["temperature_kelvin"] for item in summaries
                    ],
                    "cases": summaries,
                },
            )
            log(
                f"[{label}] DONE batch_ESS first25="
                f"{summary['initial_25_batch_ess_mean']:.6f} last25="
                f"{summary['final_25_batch_ess_mean']:.6f} "
                f"audit_ESS={identity_ess:.8f}->{final_ess:.8f} "
                f"wall={summary['case_wall_seconds']:.1f}s"
            )

        combined = {
            "step": np.arange(1, P.SOLVENT_KLXX_STEPS + 1),
            **{
                f"T{case['temperature_kelvin']:g}K_ess": case["ess_history"]
                for case in plot_cases
            },
        }
        save_h5(staging / "ess_curves.h5", combined, {"steps": P.SOLVENT_KLXX_STEPS})
        np.savetxt(
            staging / "ess_curves.csv",
            np.column_stack(list(combined.values())),
            delimiter=",",
            header=",".join(combined),
            comments="",
        )
        plot_results(staging / "ess_curves.png", plot_cases)
        final_summary = {
            "schema_version": 1,
            "complete": True,
            "molecule": "methane",
            "solvent_model": registry_record["model"]["implicit_solvent"],
            "bundle": registry_record,
            "temperature_order_kelvin": list(P.SOLVENT_KLXX_TEMPERATURES_K),
            "cut_kj_mol": P.SOLVENT_KLXX_TEMPERATURE_CUT_KJ_MOL,
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
            },
            "sampling": {
                "target_train": P.N_TARGET_TRAIN,
                "target_holdout": P.N_TARGET_HOLDOUT,
                "source_train": P.N_SOURCE_TRAIN,
                "source_audit": P.N_SELECTION,
                "smc_ladder": P.SMC_LADDER,
                "mala_step_rule": "0.01 * temperature_kelvin / bundle_temperature_kelvin",
                "mala_iters": P.MC_ITERS,
                "chunk": P.CHUNK,
            },
            "cases": summaries,
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
                "ess_curves_h5_sha256": sha256(staging / "ess_curves.h5"),
                "ess_curves_csv_sha256": sha256(staging / "ess_curves.csv"),
                "ess_curves_png_sha256": sha256(staging / "ess_curves.png"),
            },
        }
        write_json(staging / "summary.json", final_summary)
        (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
        (staging / "progress.json").unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
        print(f"PASS solvent c50 temperature KLXX sweep: {output}", flush=True)
    except BaseException:
        log("FAILED; staged results retained")
        raise


if __name__ == "__main__":
    main()
