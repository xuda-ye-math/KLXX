#!/usr/bin/env python
"""Matched vacuum CH4 c50 comparison of bare KL and KLXX."""

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
import numpy as np

from jflows.train import Monitor
from jflows_md import (
    mixed_quench_and_temper,
    train_molecular_forward_KLX_G,
    train_molecular_forward_KLXX_G,
)

import parameters as P
from audit_target_pools import compare as compare_pools
from compare_ch4_kl_klxx import flow0, parse_monitor, sha256
from train_alkanes import (
    chunked_log_weight,
    chunked_target_loss,
    normalized_ess,
    run_smc_pool,
)
from vacuum import Vacuum_Molecular_Potential


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--solvent-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    solvent_reference = args.solvent_reference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    staging = output.with_name(f".{output.name}.inprogress-{os.getpid()}")
    staging.mkdir(parents=True)
    log_path = staging / "comparison.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started_total = time.time()
    vacuum = Vacuum_Molecular_Potential.from_bundle(bundle)
    target = vacuum.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = vacuum.source()
    reference = vacuum.reference_internal()[None]
    energy_origin = float(target(reference)[0])
    mala_step = 1e-2
    log(
        f"START vacuum CH4 LR={P.COMPARISON_LR:g} steps={P.STEPS} "
        f"batch={P.N_BATCH} ladder={P.SMC_LADDER} MALA={mala_step:g}"
    )

    pools = []
    pool_records = []
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
            domain=vacuum.domain,
            step=mala_step,
            ladder=P.SMC_LADDER,
        )
        minimum_ess = float(np.min(level_ess))
        record = {
            "role": role,
            "seed": seed,
            "samples": count,
            "ladder": P.SMC_LADDER,
            "mala_step": mala_step,
            "minimum_incremental_ess": minimum_ess,
            "median_mala_acceptance": float(np.median(acceptance)),
            "minimum_mala_acceptance": float(np.min(acceptance)),
            "wall_seconds": time.time() - pool_started,
        }
        log(
            f"target pool {role} min_ESS={minimum_ess:.5f} "
            f"median_accept={record['median_mala_acceptance']:.4f}"
        )
        if minimum_ess < P.SMC_MIN_ESS:
            raise RuntimeError(
                f"vacuum {role} pool fails ladder-8 ESS floor: "
                f"{minimum_ess} < {P.SMC_MIN_ESS}"
            )
        pools.append(pool)
        pool_records.append((record, level_ess, acceptance))
    target_train, target_holdout = pools
    pool_audit = compare_pools(
        np.asarray(target_train),
        np.asarray(target_holdout),
        target,
        vacuum.domain.euclidean_dim,
    )
    pool_gates = {
        "maximum_euclidean_ks": P.EUCLIDEAN_KS_MAX,
        "maximum_periodic_marginal_js_bits": P.MARGINAL_JS_BITS_MAX,
        "maximum_adjacent_periodic_pair_js_bits": P.JOINT_ROTAMER_JS_BITS_MAX,
        "target_energy_ks": P.ENERGY_KS_MAX,
    }
    if any(pool_audit[name] > threshold for name, threshold in pool_gates.items()):
        raise RuntimeError(f"vacuum target pools fail distribution gates: {pool_audit}")
    log(
        f"target pools agree: max_KS={pool_audit['maximum_euclidean_ks']:.5f} "
        f"energy_KS={pool_audit['target_energy_ks']:.5f}"
    )

    source_train = source.samples(jax.random.key(P.SOURCE_TRAIN_SEED), P.N_SOURCE_TRAIN)
    audit_blocks = [
        source.samples(jax.random.key(seed), P.N_AUDIT_BLOCK)
        for seed in P.AUDIT_SEEDS
    ]
    source_audit = jnp.concatenate(audit_blocks, axis=0)
    identity_log_weight = chunked_log_weight(
        source_audit, source, target, flow0(vacuum.domain), P.CHUNK
    )
    identity_ess = normalized_ess(identity_log_weight)
    log(f"vacuum identity audit ESS={identity_ess:.8f}")

    hat_initial_key, hat_key = jax.random.split(jax.random.key(P.KLXX_HAT_SEED))
    hat_initial = source.samples(hat_initial_key, P.N_TARGET_TRAIN)
    hat_started = time.time()
    hat_pool, hat_acceptance = mixed_quench_and_temper(
        hat_key,
        hat_initial,
        target,
        vacuum.domain,
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
        f"KLXX hat pool ready N={hat_pool.shape[0]} "
        f"MALA={float(jnp.mean(hat_acceptance)):.4f} wall={hat_wall:.1f}s"
    )

    results = []
    saved = {}
    for objective in ("kl", "klxx"):
        messages: list[str] = []

        def monitor(message: str) -> None:
            messages.append(message)
            log(f"[{objective}] {message}")

        case_started = time.time()
        common = dict(
            n_batch=P.N_BATCH,
            steps=P.STEPS,
            lr=P.COMPARISON_LR,
            energy_origin=energy_origin,
            e_clip=P.E_CLIP,
            g_clip=P.G_CLIP,
            monitor=Monitor(1, "", monitor),
            seed=P.TRAINER_SEED,
            lr_warmup=P.LR_WARMUP,
        )
        if objective == "kl":
            trained, ess, kept, updated = train_molecular_forward_KLX_G(
                target_train,
                source_train,
                source,
                target,
                flow0(vacuum.domain),
                coeff_lambda=0.0,
                **common,
            )
        else:
            trained, ess, kept, updated = train_molecular_forward_KLXX_G(
                target_train,
                source_train,
                hat_pool,
                source,
                target,
                flow0(vacuum.domain),
                vacuum.domain,
                coeff_lambda=P.KLXX_COEFF_LAMBDA,
                coeff_alpha=P.KLXX_COEFF_ALPHA,
                coeff_beta=P.KLXX_COEFF_BETA,
                mc_step=mala_step,
                mc_iters=P.MC_ITERS,
                images=P.WRAPPED_IMAGES,
                **common,
            )
        jax.block_until_ready((trained, ess, kept, updated))
        jax.effects_barrier()
        final_log_weight = chunked_log_weight(
            source_audit, source, target, trained, P.CHUNK
        )
        holdout_loss = chunked_target_loss(
            target_holdout, source, target, trained, P.CHUNK
        )
        loss, printed_ess = parse_monitor(messages, P.STEPS)
        ess_np = np.asarray(ess)
        if not np.allclose(ess_np, printed_ess, atol=5e-5, rtol=5e-4):
            raise RuntimeError(f"{objective}: printed and returned ESS differ")
        final_ess = normalized_ess(final_log_weight)
        block_ess = [
            normalized_ess(
                final_log_weight[
                    index * P.N_AUDIT_BLOCK : (index + 1) * P.N_AUDIT_BLOCK
                ]
            )
            for index in range(P.N_AUDIT_BLOCKS)
        ]
        result = {
            "objective": objective,
            "audit_ess": final_ess,
            "audit_ess_delta": final_ess - identity_ess,
            "block_audit_ess": block_ess,
            "holdout_kl_loss_mean": float(np.mean(holdout_loss)),
            "initial_10_monitor_loss_mean": float(np.mean(loss[:10])),
            "final_10_monitor_loss_mean": float(np.mean(loss[-10:])),
            "initial_10_batch_ess_mean": float(np.mean(ess_np[:10])),
            "final_10_batch_ess_mean": float(np.mean(ess_np[-10:])),
            "minimum_kept_fraction": float(np.min(np.asarray(kept))),
            "all_updates_applied": bool(np.asarray(updated).all()),
            "wall_seconds": time.time() - case_started,
        }
        results.append(result)
        saved[objective] = {
            "flow": trained,
            "monitor_loss": loss,
            "ess_history": ess_np,
            "kept_history": np.asarray(kept),
            "update_history": np.asarray(updated),
            "final_log_weight": final_log_weight,
            "holdout_kl_loss": holdout_loss,
        }
        log(
            f"RESULT {objective} audit_ESS={final_ess:.8f} "
            f"holdout_KL={result['holdout_kl_loss_mean']:.6g}"
        )

    with h5py.File(staging / "samples.h5", "w") as handle:
        arrays = {
            "target_train": np.asarray(target_train),
            "target_holdout": np.asarray(target_holdout),
            "source_train": np.asarray(source_train),
            "source_audit": np.asarray(source_audit),
            "identity_log_weight": identity_log_weight,
            "hat_pool": np.asarray(hat_pool),
            "hat_mala_acceptance": np.asarray(hat_acceptance),
        }
        for index, (_, level_ess, acceptance) in enumerate(pool_records):
            arrays[f"target_pool_{index}_level_ess"] = level_ess
            arrays[f"target_pool_{index}_mala_acceptance"] = acceptance
        for name, array in arrays.items():
            kwargs = (
                {"compression": "gzip", "compression_opts": 1, "shuffle": True}
                if np.asarray(array).size > 1024 else {}
            )
            handle.create_dataset(name, data=array, **kwargs)
        for objective, values in saved.items():
            group = handle.create_group(objective)
            for name, array in values.items():
                if name == "flow":
                    continue
                kwargs = (
                    {"compression": "gzip", "compression_opts": 1, "shuffle": True}
                    if np.asarray(array).size > 1024 else {}
                )
                group.create_dataset(name, data=array, **kwargs)
    for objective, values in saved.items():
        eqx.tree_serialise_leaves(staging / f"{objective}_flow.eqx", values["flow"])

    solvent = json.loads((solvent_reference / "summary.json").read_text())
    solvent_results = {item["objective"]: item for item in solvent["results"]}
    comparison = {}
    for item in results:
        objective = item["objective"]
        solvent_ess = solvent_results[objective]["audit_ess"]
        comparison[objective] = {
            "vacuum_audit_ess": item["audit_ess"],
            "implicit_solvent_audit_ess": solvent_ess,
            "vacuum_minus_implicit_solvent": item["audit_ess"] - solvent_ess,
        }
    summary = {
        "schema_version": 1,
        "complete": True,
        "model": "GAFF2/AM1-BCC vacuum; OBC1 and ACE removed",
        "bundle": str(bundle),
        "bundle_manifest_sha256": vacuum.manifest_sha256,
        "target": {
            "id": P.TARGET_ID,
            "energy_cut_kj_mol": P.ENERGY_CUT_KJ_MOL,
            "energy_scale_kj_mol": P.ENERGY_SCALE_KJ_MOL,
            "tail_fraction": P.TAIL_FRACTION,
            "energy_origin_reduced": energy_origin,
        },
        "shared": {
            "lr": P.COMPARISON_LR,
            "steps": P.STEPS,
            "n_batch": P.N_BATCH,
            "ladder": P.SMC_LADDER,
            "mala_step": mala_step,
            "mc_iters": P.MC_ITERS,
        },
        "target_pools": [record for record, _, _ in pool_records],
        "target_pool_audit": pool_audit,
        "target_pool_gates": pool_gates,
        "identity_ess": identity_ess,
        "hat_pool": {
            "samples": int(hat_pool.shape[0]),
            "melt": P.KLXX_MELT,
            "opt_step": P.KLXX_OPT_STEP,
            "opt_iters": P.KLXX_OPT_ITERS,
            "mala_acceptance_mean": float(jnp.mean(hat_acceptance)),
            "wall_seconds": hat_wall,
        },
        "results": results,
        "implicit_solvent_reference": {
            "path": str(solvent_reference),
            "summary_sha256": sha256(solvent_reference / "summary.json"),
        },
        "comparison": comparison,
        "runtime": {
            "command": sys.argv,
            "wall_seconds": time.time() - started_total,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "driver_sha256": sha256(Path(__file__)),
            "vacuum_source_sha256": sha256(HERE / "vacuum.py"),
            "parameters_sha256": sha256(HERE / "parameters.py"),
        },
    }
    write_path = staging / "summary.json"
    write_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary["artifacts"] = {
        "samples_h5_sha256": sha256(staging / "samples.h5"),
        "kl_flow_sha256": sha256(staging / "kl_flow.eqx"),
        "klxx_flow_sha256": sha256(staging / "klxx_flow.eqx"),
    }
    write_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
    staging.rename(output)
    print(f"PASS matched vacuum CH4 KL/KLXX comparison: {output}", flush=True)


if __name__ == "__main__":
    main()
