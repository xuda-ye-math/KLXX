#!/usr/bin/env python
"""Matched LR=5e-3 CH4 comparison of bare KL and KLXX."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
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
    Mixed_NSF,
    Molecular_Potential,
    mixed_quench_and_temper,
    train_molecular_forward_KLX_G,
    train_molecular_forward_KLXX_G,
)

import parameters as P
from train_alkanes import chunked_log_weight, chunked_target_loss, normalized_ess


HERE = Path(__file__).resolve().parent
PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+loss = (?P<loss>[+-][0-9.eE+-]+)\s+"
    r"ESS = (?P<ess>[0-9.eE+-]+)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_monitor(messages: list[str], expected: int) -> tuple[np.ndarray, np.ndarray]:
    records = []
    for message in messages:
        match = PATTERN.search(message)
        if match:
            records.append(
                (int(match["step"]), float(match["loss"]), float(match["ess"]))
            )
    records.sort()
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} monitor records, got {len(records)}")
    return (
        np.asarray([record[1] for record in records]),
        np.asarray([record[2] for record in records]),
    )


def flow0(domain):
    return Mixed_NSF(
        jax.random.key(P.FLOW_SEED),
        domain,
        bins=P.BINS,
        transforms=P.TRANSFORMS,
        euclidean_bound=P.EUCLIDEAN_BOUND,
        hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE,
    ).zeros()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_run = args.input.resolve()
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

    baseline = json.loads((source_run / "summary.json").read_text())
    if baseline["molecule"] != "methane" or baseline["target"]["id"] != P.TARGET_ID:
        raise ValueError("input must be a completed methane c50_rho001 run")
    with h5py.File(source_run / "samples.h5", "r") as handle:
        target_train = jnp.asarray(handle["target_train"][...])
        target_holdout = jnp.asarray(handle["target_holdout"][...])
        source_train = jnp.asarray(handle["source_train"][...])
        source_audit = jnp.asarray(handle["source_audit"][...])
    bundle = HERE / "bundles" / baseline["bundle"]["path"]
    physical = Molecular_Potential.from_bundle(bundle)
    target = physical.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = physical.source()
    energy_origin = float(target(physical.reference_internal()[None])[0])
    mala_step = float(baseline["mala_step"])
    identity_log_weight = chunked_log_weight(
        source_audit, source, target, flow0(physical.domain), P.CHUNK
    )
    identity_ess = normalized_ess(identity_log_weight)
    log(
        f"START LR={P.COMPARISON_LR:g} steps={P.STEPS} batch={P.N_BATCH} "
        f"identity_ESS={identity_ess:.8f} MALA_step={mala_step:g}"
    )

    hat_initial_key, hat_key = jax.random.split(jax.random.key(P.KLXX_HAT_SEED))
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
        f"KLXX hat pool ready N={hat_pool.shape[0]} "
        f"MALA={float(jnp.mean(hat_acceptance)):.4f} wall={hat_wall:.1f}s"
    )

    results = []
    for objective in ("kl", "klxx"):
        case = staging / objective
        case.mkdir()
        messages: list[str] = []

        def monitor(message: str) -> None:
            messages.append(message)
            log(f"[{objective}] {message}")

        started = time.time()
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
                flow0(physical.domain),
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
                flow0(physical.domain),
                physical.domain,
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
        result = {
            "objective": objective,
            "lr": P.COMPARISON_LR,
            "steps": P.STEPS,
            "n_batch": P.N_BATCH,
            "audit_ess": final_ess,
            "audit_ess_delta": final_ess - identity_ess,
            "holdout_kl_loss_mean": float(np.mean(holdout_loss)),
            "initial_10_monitor_loss_mean": float(np.mean(loss[:10])),
            "final_10_monitor_loss_mean": float(np.mean(loss[-10:])),
            "initial_10_batch_ess_mean": float(np.mean(ess_np[:10])),
            "final_10_batch_ess_mean": float(np.mean(ess_np[-10:])),
            "minimum_kept_fraction": float(np.min(np.asarray(kept))),
            "all_updates_applied": bool(np.asarray(updated).all()),
            "wall_seconds": time.time() - started,
        }
        results.append(result)
        with h5py.File(case / "result.h5", "w") as handle:
            for name, value in (
                ("monitor_loss", loss),
                ("ess_history", ess_np),
                ("kept_history", np.asarray(kept)),
                ("update_history", np.asarray(updated)),
                ("final_log_weight", final_log_weight),
                ("holdout_kl_loss", holdout_loss),
            ):
                array = np.asarray(value)
                kwargs = (
                    {"compression": "gzip", "compression_opts": 1, "shuffle": True}
                    if array.size > 1024 else {}
                )
                handle.create_dataset(name, data=array, **kwargs)
        eqx.tree_serialise_leaves(case / "flow.eqx", trained)
        log(
            f"RESULT {objective} audit_ESS={final_ess:.8f} "
            f"holdout_KL={result['holdout_kl_loss_mean']:.6g} "
            f"batch_ESS_10={result['final_10_batch_ess_mean']:.5f}"
        )

    with h5py.File(staging / "hat_pool.h5", "w") as handle:
        handle.create_dataset(
            "hat_pool", data=np.asarray(hat_pool),
            compression="gzip", compression_opts=1, shuffle=True,
        )
        handle.create_dataset("hat_mala_acceptance", data=np.asarray(hat_acceptance))
    summary = {
        "schema_version": 1,
        "complete": True,
        "molecule": "methane",
        "target": P.TARGET_ID,
        "source_run": str(source_run),
        "source_summary_sha256": sha256(source_run / "summary.json"),
        "source_samples_sha256": sha256(source_run / "samples.h5"),
        "identity_ess": identity_ess,
        "shared": {
            "lr": P.COMPARISON_LR,
            "steps": P.STEPS,
            "n_batch": P.N_BATCH,
            "mala_step": mala_step,
            "mc_iters": P.MC_ITERS,
            "flow_seed": P.FLOW_SEED,
            "trainer_seed": P.TRAINER_SEED,
        },
        "klxx": {
            "coeff_lambda": P.KLXX_COEFF_LAMBDA,
            "coeff_alpha": P.KLXX_COEFF_ALPHA,
            "coeff_beta": P.KLXX_COEFF_BETA,
            "melt": P.KLXX_MELT,
            "opt_step": P.KLXX_OPT_STEP,
            "opt_iters": P.KLXX_OPT_ITERS,
            "hat_seed": P.KLXX_HAT_SEED,
            "hat_samples": int(hat_pool.shape[0]),
            "hat_mala_acceptance_mean": float(jnp.mean(hat_acceptance)),
            "hat_wall_seconds": hat_wall,
        },
        "results": results,
        "runtime": {
            "command": sys.argv,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "driver_sha256": sha256(Path(__file__)),
            "parameters_sha256": sha256(HERE / "parameters.py"),
        },
    }
    write_json(staging / "summary.json", summary)
    (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
    staging.rename(output)
    print(f"PASS matched CH4 KL/KLXX comparison: {output}", flush=True)


if __name__ == "__main__":
    main()
