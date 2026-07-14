#!/usr/bin/env python
"""Production-capacity direct KLXX reconciliation for methane c50/300 K."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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
from scipy.stats import ks_2samp, kstest

from jflows.train import Monitor
from jflows_md import (
    Mixed_NSF,
    Molecular_Potential,
    package_source_sha256,
    train_molecular_forward_KLXX_G,
)

import parameters as P
from audit_target_pools import js_bits, pairwise_periodic_histogram
from sweep_ch4_solvent_klxx import (
    bundle_record,
    git_state,
    rolling_mean,
    save_h5,
    sha256,
    write_json,
)
from train_alkanes import chunked_log_weight, normalized_ess


HERE = Path(__file__).resolve().parent


def production_flow(domain):
    return Mixed_NSF(
        jax.random.key(P.FLOW_SEED),
        domain,
        bins=P.RECON_BINS,
        transforms=P.RECON_TRANSFORMS,
        euclidean_bound=P.EUCLIDEAN_BOUND,
        hidden_features=P.RECON_HIDDEN_FEATURES,
        slope=P.SLOPE,
        mask_strategy=P.RECON_MASK_STRATEGY,
    ).zeros()


def trainable_count(flow) -> int:
    return int(
        sum(
            value.size
            for value in jax.tree.leaves(flow)
            if hasattr(value, "dtype")
            and np.issubdtype(np.asarray(value).dtype, np.inexact)
        )
    )


def chunked_map(samples, function, chunk: int) -> np.ndarray:
    values = []
    for part in jnp.array_split(jnp.asarray(samples), chunk, axis=0):
        values.append(np.asarray(jax.block_until_ready(function(part))))
    return np.concatenate(values)


def mode_metrics(
    target: np.ndarray,
    generated: np.ndarray,
    latent_target: np.ndarray,
    log_weight: np.ndarray,
    euclidean_dim: int,
) -> dict:
    torsion = generated[:, euclidean_dim:]
    target_torsion = target[:, euclidean_dim:]
    same_sign = np.sin(torsion[:, 0]) * np.sin(torsion[:, 1]) >= 0.0
    first_mode = (~same_sign) & (np.sin(torsion[:, 0]) > 0.0)
    second_mode = (~same_sign) & ~first_mode
    valid_count = int((~same_sign).sum())
    proposal_modes = np.asarray(
        [first_mode.sum(), second_mode.sum()], dtype=np.float64
    ) / max(valid_count, 1)
    finite = np.isfinite(log_weight)
    shifted = np.where(finite, log_weight, -np.inf)
    maximum = float(np.max(shifted))
    weight = np.exp(shifted - maximum)
    weight /= weight.sum()
    weighted_modes = [
        float(weight[first_mode].sum()),
        float(weight[second_mode].sum()),
    ]
    weighted_off = float(weight[same_sign].sum())
    euclidean_ks = [
        float(ks_2samp(target[:, index], generated[:, index]).statistic)
        for index in range(euclidean_dim)
    ]
    joint_js = js_bits(
        pairwise_periodic_histogram(
            target_torsion[:, 0], target_torsion[:, 1], bins=72
        ),
        pairwise_periodic_histogram(torsion[:, 0], torsion[:, 1], bins=72),
    )
    latent_periodic = latent_target[:, euclidean_dim:]
    latent_uniform_ks = [
        float(kstest((latent_periodic[:, index] + np.pi) / (2.0 * np.pi), "uniform").statistic)
        for index in range(latent_periodic.shape[1])
    ]
    latent_sine_correlation = float(
        np.corrcoef(
            np.sin(latent_periodic[:, 0]), np.sin(latent_periodic[:, 1])
        )[0, 1]
    )
    target_same_sign = float(
        np.mean(
            np.sin(target_torsion[:, 0]) * np.sin(target_torsion[:, 1]) >= 0.0
        )
    )
    return {
        "samples": int(generated.shape[0]),
        "target_same_sign_fraction": target_same_sign,
        "proposal_same_sign_fraction": float(same_sign.mean()),
        "proposal_valid_mode_fractions": proposal_modes.tolist(),
        "proposal_mode_tv_from_uniform": float(abs(proposal_modes[0] - 0.5)),
        "weighted_mode_masses": weighted_modes,
        "weighted_same_sign_mass": weighted_off,
        "maximum_euclidean_ks": max(euclidean_ks, default=0.0),
        "euclidean_ks": euclidean_ks,
        "joint_torsion_js_bits": joint_js,
        "latent_torsion_uniform_ks": latent_uniform_ks,
        "latent_sine_correlation": latent_sine_correlation,
    }


def plot_reconciliation(path: Path, ess: np.ndarray, summary: dict) -> None:
    steps = np.arange(1, ess.size + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axis, bar_axis = axes
    axis.plot(steps, ess, color="#0072B2", alpha=0.18, linewidth=0.7)
    axis.plot(
        steps, rolling_mean(ess, 25), color="#0072B2", linewidth=2.0,
        label="balanced production flow (25-step mean)",
    )
    axis.axhline(
        summary["baseline_audit_ess"], color="#777777", linestyle="--",
        label="previous 1M audit ESS",
    )
    axis.set(
        xlim=(1, ess.size), ylim=(0.0, 1.0), xlabel="KLXX optimizer step",
        ylabel="training batch ESS", title="Methane c50/300 K reconciliation",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    audit = [
        summary["identity_audit_ess"],
        summary["baseline_audit_ess"],
        summary["final_audit_ess"],
    ]
    bar_axis.bar(
        np.arange(3), audit, color=("#999999", "#CC79A7", "#009E73")
    )
    bar_axis.set(
        xticks=np.arange(3), xticklabels=("identity", "old flow", "reconciled"),
        ylim=(0.0, 1.0), ylabel="1M-sample audit ESS",
        title="Independent audit",
    )
    bar_axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    staging = output.with_name(f".{output.name}.inprogress-{os.getpid()}")
    if staging.exists():
        raise SystemExit(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    log_path = staging / "train.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started = time.time()
    try:
        baseline_summary = json.loads((baseline.parent / "summary.json").read_text())
        baseline_case = json.loads((baseline / "summary.json").read_text())
        if baseline_case["temperature_kelvin"] != 300.0:
            raise ValueError("baseline must be the completed 300 K methane case")
        if baseline_case["cut_kj_mol"] != 50.0:
            raise ValueError("baseline must use c50")
        with h5py.File(baseline / "diagnostics.h5", "r") as handle:
            target_train = jnp.asarray(handle["target_train"][...])
            target_holdout = jnp.asarray(handle["target_holdout"][...])
            source_train = jnp.asarray(handle["source_train"][...])
            source_audit = jnp.asarray(handle["source_audit"][...])
            hat_pool = jnp.asarray(handle["hat_pool"][...])
            identity_log_weight = np.asarray(handle["identity_log_weight"][...])
            baseline_log_weight = np.asarray(handle["final_log_weight"][...])
        bundle, registry_record = bundle_record()
        physical = Molecular_Potential.from_bundle(bundle)
        target = physical.regularized(
            50.0,
            energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
            tail_fraction=P.TAIL_FRACTION,
        )
        source = physical.source()
        energy_origin = float(target(physical.reference_internal()[None])[0])
        flow0 = production_flow(physical.domain)
        masks = [list(layer.condition_indices) for layer in flow0.couplings]
        torsions = set(range(physical.domain.euclidean_dim, physical.dimension))
        for first, second in zip(flow0.couplings[::2], flow0.couplings[1::2], strict=True):
            first_set = set(first.condition_indices)
            second_set = set(second.condition_indices)
            if not first_set & torsions or not second_set & torsions:
                raise RuntimeError("balanced flow failed to split torsions")
        parameter_count = trainable_count(flow0)
        identity_ess = normalized_ess(identity_log_weight)
        baseline_ess = normalized_ess(baseline_log_weight)
        log(
            f"START direct KLXX c50/300K bins={P.RECON_BINS} "
            f"transforms={P.RECON_TRANSFORMS} hidden={P.RECON_HIDDEN_FEATURES} "
            f"params={parameter_count} mask={P.RECON_MASK_STRATEGY} "
            f"steps={P.RECON_STEPS} batch={P.RECON_N_BATCH} lr={P.RECON_LR:g} "
            f"identity_ESS={identity_ess:.8f} baseline_ESS={baseline_ess:.8f}"
        )
        log(f"condition masks={masks}")

        def monitor(message: str) -> None:
            log(message)

        train_started = time.time()
        trained, ess, kept, updated = train_molecular_forward_KLXX_G(
            target_train,
            source_train,
            hat_pool,
            source,
            target,
            flow0,
            physical.domain,
            n_batch=P.RECON_N_BATCH,
            steps=P.RECON_STEPS,
            lr=P.RECON_LR,
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
            checkpoint=P.RECON_CHECKPOINT,
            lr_warmup=P.LR_WARMUP,
        )
        jax.block_until_ready((trained, ess, kept, updated))
        jax.effects_barrier()
        training_wall = time.time() - train_started
        ess_history = np.asarray(ess)
        kept_history = np.asarray(kept)
        update_history = np.asarray(updated)
        if ess_history.shape != (P.RECON_STEPS,) or not np.isfinite(ess_history).all():
            raise RuntimeError("trainer returned an invalid ESS history")

        final_log_weight = chunked_log_weight(
            source_audit, source, target, trained, P.CHUNK
        )
        final_ess = normalized_ess(final_log_weight)
        block_ess = []
        for index in range(P.N_AUDIT_BLOCKS):
            section = slice(
                index * P.N_AUDIT_BLOCK, (index + 1) * P.N_AUDIT_BLOCK
            )
            block_ess.append(normalized_ess(final_log_weight[section]))

        geometry_n = P.N_TARGET_HOLDOUT
        generated = chunked_map(
            source_audit[:geometry_n], trained.inv, P.CHUNK
        )
        latent_target = chunked_map(
            target_holdout[:geometry_n], trained, P.CHUNK
        )
        geometry = mode_metrics(
            np.asarray(target_holdout[:geometry_n]),
            generated,
            latent_target,
            final_log_weight[:geometry_n],
            physical.domain.euclidean_dim,
        )
        criteria = {
            "audit_ess": final_ess >= P.RECON_AUDIT_ESS_MIN,
            "minimum_block_ess": min(block_ess) >= P.RECON_BLOCK_ESS_MIN,
            "same_sign_leakage": geometry["proposal_same_sign_fraction"]
            <= P.RECON_SAME_SIGN_MAX,
            "mode_balance": geometry["proposal_mode_tv_from_uniform"]
            <= P.RECON_MODE_TV_MAX,
            "torsion_joint_js": geometry["joint_torsion_js_bits"]
            <= P.RECON_TORSION_JS_BITS_MAX,
            "euclidean_marginals": geometry["maximum_euclidean_ks"]
            <= P.RECON_EUCLIDEAN_KS_MAX,
            "latent_torsion_copula": abs(geometry["latent_sine_correlation"])
            <= P.RECON_LATENT_SINE_CORR_MAX,
            "all_updates": bool(update_history.all()),
            "all_samples_kept": bool(np.all(kept_history == 1.0)),
        }
        passed = all(criteria.values())
        arrays = {
            "ess_history": ess_history,
            "kept_history": kept_history,
            "update_history": update_history,
            "identity_log_weight": identity_log_weight,
            "baseline_log_weight": baseline_log_weight,
            "final_log_weight": final_log_weight,
            "block_final_ess": np.asarray(block_ess),
            "generated_geometry": generated,
            "latent_target_geometry": latent_target,
        }
        diagnostics_path = staging / "diagnostics.h5"
        save_h5(
            diagnostics_path,
            arrays,
            {
                "temperature_kelvin": 300.0,
                "cut_kj_mol": 50.0,
                "baseline_diagnostics_sha256": sha256(baseline / "diagnostics.h5"),
            },
        )
        flow_path = staging / "flow.eqx"
        eqx.tree_serialise_leaves(flow_path, trained)
        summary = {
            "schema_version": 1,
            "complete": True,
            "passed": passed,
            "molecule": "methane",
            "temperature_kelvin": 300.0,
            "target": {
                "cut_kj_mol": 50.0,
                "energy_scale_kj_mol": P.ENERGY_SCALE_KJ_MOL,
                "tail_fraction": P.TAIL_FRACTION,
                "implicit_solvent": registry_record["model"]["implicit_solvent"],
            },
            "bundle": registry_record,
            "baseline": {
                "path": str(baseline),
                "summary_sha256": sha256(baseline / "summary.json"),
                "diagnostics_sha256": sha256(baseline / "diagnostics.h5"),
                "parent_summary_sha256": sha256(baseline.parent / "summary.json"),
                "audit_ess": baseline_ess,
            },
            "flow": {
                "class": "Mixed_NSF",
                "bins": P.RECON_BINS,
                "transforms": P.RECON_TRANSFORMS,
                "hidden_features": list(P.RECON_HIDDEN_FEATURES),
                "euclidean_bound": P.EUCLIDEAN_BOUND,
                "mask_strategy": P.RECON_MASK_STRATEGY,
                "condition_masks": masks,
                "trainable_parameters": parameter_count,
            },
            "training": {
                "objective": "single direct source-to-target KLXX stage",
                "steps": P.RECON_STEPS,
                "n_batch": P.RECON_N_BATCH,
                "lr": P.RECON_LR,
                "lr_warmup": P.LR_WARMUP,
                "checkpoint": P.RECON_CHECKPOINT,
                "coeff_lambda": P.KLXX_COEFF_LAMBDA,
                "coeff_alpha": P.KLXX_COEFF_ALPHA,
                "coeff_beta": P.KLXX_COEFF_BETA,
                "minimum_kept_fraction": float(kept_history.min()),
                "all_updates_applied": bool(update_history.all()),
                "initial_25_batch_ess_mean": float(ess_history[:25].mean()),
                "final_25_batch_ess_mean": float(ess_history[-25:].mean()),
                "wall_seconds": training_wall,
            },
            "audit": {
                "n": P.N_SELECTION,
                "identity_ess": identity_ess,
                "baseline_ess": baseline_ess,
                "final_ess": final_ess,
                "improvement_over_baseline": final_ess - baseline_ess,
                "block_final_ess": block_ess,
                "geometry": geometry,
            },
            "criteria": criteria,
            "thresholds": {
                "audit_ess_min": P.RECON_AUDIT_ESS_MIN,
                "block_ess_min": P.RECON_BLOCK_ESS_MIN,
                "same_sign_max": P.RECON_SAME_SIGN_MAX,
                "mode_tv_max": P.RECON_MODE_TV_MAX,
                "torsion_js_bits_max": P.RECON_TORSION_JS_BITS_MAX,
                "euclidean_ks_max": P.RECON_EUCLIDEAN_KS_MAX,
                "latent_sine_corr_max": P.RECON_LATENT_SINE_CORR_MAX,
            },
            "runtime": {
                "command": sys.argv,
                "wall_seconds": time.time() - started,
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
                "diagnostics_sha256": sha256(diagnostics_path),
                "flow_sha256": sha256(flow_path),
            },
        }
        plot_path = staging / "reconciliation.png"
        plot_reconciliation(
            plot_path,
            ess_history,
            {
                "identity_audit_ess": identity_ess,
                "baseline_audit_ess": baseline_ess,
                "final_audit_ess": final_ess,
            },
        )
        summary["artifacts"]["plot_sha256"] = sha256(plot_path)
        write_json(staging / "summary.json", summary)
        (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
        staging.rename(output)
        print(
            f"{'PASS' if passed else 'FAIL'} methane reconciliation: {output} "
            f"ESS={final_ess:.8f} same_sign="
            f"{geometry['proposal_same_sign_fraction']:.6f}",
            flush=True,
        )
    except BaseException:
        log("FAILED; staged results retained")
        raise


if __name__ == "__main__":
    main()
