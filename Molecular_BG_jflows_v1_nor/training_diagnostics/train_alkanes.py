#!/usr/bin/env python
"""Full-size direct-KL diagnostic for the controlled alkane series."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
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
    package_source_sha256,
    sequential_monte_carlo,
    train_molecular_forward_KLX_G,
)

import parameters as P


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE / "bundles"
RUN_ROOT = HERE / "runs"
MONITOR_PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+loss = (?P<loss>[+-][0-9.eE+-]+)\s+"
    r"ESS = (?P<ess>[0-9.eE+-]+)"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(path: Path) -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=path, check=True,
        text=True, capture_output=True,
    ).stdout
    return {"head": head, "dirty": bool(status), "status": status.splitlines()}


def json_write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalized_ess(log_weight: np.ndarray) -> float:
    value = np.asarray(log_weight, dtype=np.float64)
    finite = np.isfinite(value)
    if not finite.any():
        return 0.0
    maximum = float(np.max(value[finite]))
    weight = np.zeros_like(value)
    weight[finite] = np.exp(value[finite] - maximum)
    denominator = value.size * float(np.dot(weight, weight))
    if not math.isfinite(denominator) or denominator <= 0:
        return 0.0
    return float(weight.sum() ** 2 / denominator)


def chunked_log_weight(samples, source, target, flow, chunk: int) -> np.ndarray:
    values = []
    for part in jnp.array_split(samples, chunk, axis=0):
        proposal, inverse_ladj = flow.inv_and_ladj(part)
        log_weight = source(part) - target(proposal) + inverse_ladj
        values.append(np.asarray(jax.block_until_ready(log_weight)))
    return np.concatenate(values)


def chunked_target_loss(samples, source, target, flow, chunk: int) -> np.ndarray:
    values = []
    for part in jnp.array_split(samples, chunk, axis=0):
        latent, ladj = flow.call_and_ladj(part)
        loss = source(latent) - target(part) - ladj
        values.append(np.asarray(jax.block_until_ready(loss)))
    return np.concatenate(values)


def run_smc_pool(
    *, seed: int, count: int, source, target, domain, step: float, ladder: int,
) -> tuple:
    sample_key, smc_key = jax.random.split(jax.random.key(seed))
    initial = source.samples(sample_key, N=count)
    particles, level_ess, acceptance = sequential_monte_carlo(
        smc_key,
        initial,
        source,
        target,
        ladder=ladder,
        step=step,
        iters=P.MC_ITERS,
        images=P.WRAPPED_IMAGES,
        domain=domain,
        chunk=P.CHUNK,
    )
    jax.block_until_ready((particles, level_ess, acceptance))
    return particles, np.asarray(level_ess), np.asarray(acceptance)


def mala_pilot(source, target, domain, molecule_offset: int, log) -> tuple[float, list]:
    records = []
    sample_key, smc_key = jax.random.split(
        jax.random.key(P.MALA_PILOT_SEED + molecule_offset)
    )
    initial = source.samples(sample_key, N=P.MALA_PILOT_N)
    for index, step in enumerate(P.MALA_STEP_CANDIDATES):
        started = time.time()
        particles, level_ess, acceptance = sequential_monte_carlo(
            jax.random.fold_in(smc_key, index),
            initial,
            source,
            target,
            ladder=P.MALA_PILOT_LADDER,
            step=step,
            iters=P.MALA_PILOT_ITERS,
            images=P.WRAPPED_IMAGES,
            domain=domain,
            chunk=P.CHUNK,
        )
        jax.block_until_ready((particles, level_ess, acceptance))
        acceptance_np = np.asarray(acceptance)
        record = {
            "step": float(step),
            "minimum_incremental_ess": float(np.min(np.asarray(level_ess))),
            "minimum_mala_acceptance": float(np.min(acceptance_np)),
            "median_mala_acceptance": float(np.median(acceptance_np)),
            "wall_seconds": time.time() - started,
        }
        records.append(record)
        log(
            "MALA pilot "
            f"step={step:g} min_ESS={record['minimum_incremental_ess']:.3f} "
            f"median_accept={record['median_mala_acceptance']:.3f} "
            f"min_accept={record['minimum_mala_acceptance']:.3f}"
        )
    qualified = [
        item for item in records
        if P.MALA_ACCEPTANCE_MEDIAN[0] <= item["median_mala_acceptance"]
        <= P.MALA_ACCEPTANCE_MEDIAN[1]
        and item["minimum_mala_acceptance"] >= P.MALA_ACCEPTANCE_MIN
    ]
    if qualified:
        selected = max(qualified, key=lambda item: item["step"])
        reason = "largest candidate inside the preregistered acceptance window"
    else:
        safe = [
            item for item in records
            if item["minimum_mala_acceptance"] >= P.MALA_ACCEPTANCE_MIN
        ]
        candidates = safe if safe else records
        selected = min(
            candidates,
            key=lambda item: abs(item["median_mala_acceptance"] - 0.70),
        )
        reason = "closest median acceptance to 0.70 after the safety filter"
    selected["selection_reason"] = reason
    log(f"selected MALA step={selected['step']:g}: {reason}")
    return selected["step"], records


def bundle_path(molecule: str) -> tuple[Path, dict]:
    registry = json.loads((BUNDLE_ROOT / "registry.json").read_text())
    try:
        record = registry["bundles"][molecule]
    except KeyError as exc:
        raise SystemExit(
            f"no promoted bundle for {molecule!r}; build, validate, and promote it first"
        ) from exc
    path = BUNDLE_ROOT / record["path"]
    if sha256_file(path / "manifest.json") != record["manifest_sha256"]:
        raise RuntimeError("promoted bundle manifest does not match the private registry")
    return path, record


def save_h5(path: Path, arrays: dict[str, np.ndarray], attributes: dict) -> None:
    with h5py.File(path, "w") as handle:
        for name, value in arrays.items():
            array = np.asarray(value)
            kwargs = {}
            if array.ndim > 0 and array.size > 1024:
                kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
            handle.create_dataset(name, data=array, **kwargs)
        for name, value in attributes.items():
            handle.attrs[name] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", choices=P.MOLECULES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    molecule_index = P.MOLECULES.index(args.molecule)
    molecule_offset = P.SEED_MOLECULE_STRIDE * molecule_index
    expected_dimension = P.DIMENSIONS[molecule_index]
    bundle, bundle_record = bundle_path(args.molecule)

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    final = args.output or RUN_ROOT / f"{args.molecule}_{P.TARGET_ID}_{timestamp}"
    final = final.resolve()
    if final.exists():
        raise SystemExit(f"refusing to overwrite existing run: {final}")
    staging = final.with_name(f".{final.name}.inprogress-{os.getpid()}")
    if staging.exists():
        raise SystemExit(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    log_path = staging / "train.log"
    monitor_messages: list[str] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def monitor_log(message: str) -> None:
        monitor_messages.append(message)
        log(message)

    started = time.time()
    try:
        log(f"START molecule={args.molecule} target={P.TARGET_ID} output={final}")
        physical = Molecular_Potential.from_bundle(bundle)
        if physical.dimension != expected_dimension:
            raise RuntimeError(
                f"dimension mismatch: {physical.dimension} != {expected_dimension}"
            )
        target = physical.regularized(
            P.ENERGY_CUT_KJ_MOL,
            energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
            tail_fraction=P.TAIL_FRACTION,
        )
        source = physical.source()
        reference = physical.reference_internal()[None]
        energy_origin = float(target(reference)[0])
        log(
            f"bundle={bundle.name} d={physical.dimension} "
            f"domain=R{physical.domain.euclidean_dim}xT{physical.domain.periodic_dim} "
            f"energy_origin={energy_origin:.8g}"
        )

        mala_step, pilot_records = mala_pilot(
            source, target, physical.domain, molecule_offset, log
        )
        pool_results = []
        particles = []
        for role, seed in zip(("train", "holdout"), P.TARGET_POOL_SEEDS, strict=True):
            pool_started = time.time()
            result = run_smc_pool(
                seed=seed + molecule_offset,
                count=P.N_TARGET_TRAIN if role == "train" else P.N_TARGET_HOLDOUT,
                source=source,
                target=target,
                domain=physical.domain,
                step=mala_step,
                ladder=P.SMC_LADDER,
            )
            pool, level_ess, acceptance = result
            record = {
                "role": role,
                "seed": seed + molecule_offset,
                "ladder": P.SMC_LADDER,
                "minimum_incremental_ess": float(np.min(level_ess)),
                "median_mala_acceptance": float(np.median(acceptance)),
                "minimum_mala_acceptance": float(np.min(acceptance)),
                "wall_seconds": time.time() - pool_started,
                "repair": False,
            }
            log(
                f"target pool {role} ladder={P.SMC_LADDER} "
                f"min_ESS={record['minimum_incremental_ess']:.3f} "
                f"median_accept={record['median_mala_acceptance']:.3f}"
            )
            if record["minimum_incremental_ess"] < P.SMC_MIN_ESS:
                log(
                    f"target pool {role} failed min ESS {P.SMC_MIN_ESS}; "
                    f"rerunning once with ladder={P.SMC_LADDER_REPAIR}"
                )
                pool, level_ess, acceptance = run_smc_pool(
                    seed=seed + molecule_offset,
                    count=P.N_TARGET_TRAIN if role == "train" else P.N_TARGET_HOLDOUT,
                    source=source,
                    target=target,
                    domain=physical.domain,
                    step=mala_step,
                    ladder=P.SMC_LADDER_REPAIR,
                )
                record.update(
                    ladder=P.SMC_LADDER_REPAIR,
                    minimum_incremental_ess=float(np.min(level_ess)),
                    median_mala_acceptance=float(np.median(acceptance)),
                    minimum_mala_acceptance=float(np.min(acceptance)),
                    wall_seconds=time.time() - pool_started,
                    repair=True,
                )
                log(
                    f"target pool {role} repair min_ESS="
                    f"{record['minimum_incremental_ess']:.3f} "
                    f"median_accept={record['median_mala_acceptance']:.3f}"
                )
            if record["minimum_incremental_ess"] < P.SMC_MIN_ESS:
                raise RuntimeError(
                    f"target pool {role} still fails the minimum incremental "
                    f"ESS gate after its one allowed repair: "
                    f"{record['minimum_incremental_ess']:.6f} < {P.SMC_MIN_ESS}"
                )
            pool_results.append((record, level_ess, acceptance))
            particles.append(pool)
        target_train, target_holdout = particles

        source_train = source.samples(
            jax.random.key(P.SOURCE_TRAIN_SEED + molecule_offset),
            N=P.N_SOURCE_TRAIN,
        )
        audit_blocks = [
            source.samples(jax.random.key(seed + molecule_offset), N=P.N_AUDIT_BLOCK)
            for seed in P.AUDIT_SEEDS
        ]
        source_audit = jnp.concatenate(audit_blocks, axis=0)
        flow0 = Mixed_NSF(
            jax.random.key(P.FLOW_SEED + molecule_offset),
            physical.domain,
            bins=P.BINS,
            transforms=P.TRANSFORMS,
            euclidean_bound=P.EUCLIDEAN_BOUND,
            hidden_features=P.HIDDEN_FEATURES,
            slope=P.SLOPE,
        ).zeros()

        identity_log_weight = chunked_log_weight(
            source_audit, source, target, flow0, P.CHUNK
        )
        identity_ess = normalized_ess(identity_log_weight)
        identity_holdout = chunked_target_loss(
            target_holdout, source, target, flow0, P.CHUNK
        )
        log(
            f"identity audit ESS={identity_ess:.8f} "
            f"holdout_loss={float(np.mean(identity_holdout)):.8g}"
        )

        flow, ess_history, kept_history, update_history = (
            train_molecular_forward_KLX_G(
                target_train,
                source_train,
                source,
                target,
                flow0,
                n_batch=P.N_BATCH,
                steps=P.STEPS,
                lr=P.LR,
                coeff_lambda=P.COEFF_LAMBDA,
                energy_origin=energy_origin,
                e_clip=P.E_CLIP,
                g_clip=P.G_CLIP,
                monitor=Monitor(P.MONITOR_EVERY, f"[{args.molecule}] ", monitor_log),
                seed=P.TRAINER_SEED + molecule_offset,
                lr_warmup=P.LR_WARMUP,
            )
        )
        jax.block_until_ready((flow, ess_history, kept_history, update_history))
        jax.effects_barrier()

        final_log_weight = chunked_log_weight(
            source_audit, source, target, flow, P.CHUNK
        )
        final_ess = normalized_ess(final_log_weight)
        final_holdout = chunked_target_loss(
            target_holdout, source, target, flow, P.CHUNK
        )
        block_identity_ess = []
        block_final_ess = []
        for index in range(P.N_AUDIT_BLOCKS):
            section = slice(index * P.N_AUDIT_BLOCK, (index + 1) * P.N_AUDIT_BLOCK)
            block_identity_ess.append(normalized_ess(identity_log_weight[section]))
            block_final_ess.append(normalized_ess(final_log_weight[section]))

        monitor_records = []
        for message in monitor_messages:
            match = MONITOR_PATTERN.search(message)
            if match:
                monitor_records.append(
                    (int(match["step"]), float(match["loss"]), float(match["ess"]))
                )
        monitor_records.sort()
        if len(monitor_records) != P.STEPS:
            raise RuntimeError(
                f"expected {P.STEPS} monitor records, received {len(monitor_records)}"
            )
        monitored_loss = np.asarray([item[1] for item in monitor_records])
        returned_ess = np.asarray(ess_history)
        monitored_ess = np.asarray([item[2] for item in monitor_records])
        if not np.allclose(monitored_ess, returned_ess, atol=5e-5, rtol=5e-4):
            raise RuntimeError("printed and returned per-step ESS histories differ")

        initial_loss_mean = float(np.mean(monitored_loss[:10]))
        final_loss_mean = float(np.mean(monitored_loss[-10:]))
        loss_change = final_loss_mean - initial_loss_mean
        ess_delta = final_ess - identity_ess
        positive_blocks = int(
            np.sum(np.asarray(block_final_ess) > np.asarray(block_identity_ess))
        )
        all_updates = bool(np.asarray(update_history).all())
        finite = bool(
            np.isfinite(monitored_loss).all()
            and np.isfinite(returned_ess).all()
            and np.isfinite(final_log_weight).any()
        )
        stuck = bool(loss_change >= -1e-3 and ess_delta <= 0.0)
        progression_gate = bool(
            finite
            and all_updates
            and ess_delta >= P.AUDIT_DELTA_ESS_MIN
            and positive_blocks >= P.AUDIT_POSITIVE_BLOCKS_MIN
        )

        arrays = {
            "target_train": np.asarray(target_train),
            "target_holdout": np.asarray(target_holdout),
            "source_train": np.asarray(source_train),
            "source_audit": np.asarray(source_audit),
            "identity_log_weight": identity_log_weight,
            "final_log_weight": final_log_weight,
            "identity_holdout_loss": identity_holdout,
            "final_holdout_loss": final_holdout,
            "monitor_loss": monitored_loss,
            "ess_history": returned_ess,
            "kept_history": np.asarray(kept_history),
            "update_history": np.asarray(update_history),
            "block_identity_ess": np.asarray(block_identity_ess),
            "block_final_ess": np.asarray(block_final_ess),
        }
        for index, (_, level_ess, acceptance) in enumerate(pool_results):
            arrays[f"target_pool_{index}_level_ess"] = level_ess
            arrays[f"target_pool_{index}_mala_acceptance"] = acceptance
        sample_path = staging / "samples.h5"
        save_h5(
            sample_path,
            arrays,
            {
                "molecule": args.molecule,
                "target": P.TARGET_ID,
                "bundle_manifest_sha256": physical.manifest_sha256,
            },
        )
        flow_path = staging / "flow.eqx"
        eqx.tree_serialise_leaves(flow_path, flow)

        summary = {
            "schema_version": 1,
            "complete": True,
            "molecule": args.molecule,
            "formula": P.FORMULAS[molecule_index],
            "dimension": physical.dimension,
            "domain": {
                "euclidean": physical.domain.euclidean_dim,
                "periodic": physical.domain.periodic_dim,
            },
            "target": {
                "id": P.TARGET_ID,
                "energy_cut_kj_mol": P.ENERGY_CUT_KJ_MOL,
                "energy_scale_kj_mol": P.ENERGY_SCALE_KJ_MOL,
                "tail_fraction": P.TAIL_FRACTION,
                "energy_origin_reduced": energy_origin,
            },
            "bundle": bundle_record,
            "mala_step": mala_step,
            "mala_pilot": pilot_records,
            "target_pools": [item[0] for item in pool_results],
            "training": {
                "objective": "direct bare forward KL (coeff_lambda=0)",
                "n_target_train": P.N_TARGET_TRAIN,
                "n_target_holdout": P.N_TARGET_HOLDOUT,
                "n_source_train": P.N_SOURCE_TRAIN,
                "n_batch": P.N_BATCH,
                "steps": P.STEPS,
                "lr": P.LR,
                "lr_warmup": P.LR_WARMUP,
                "e_clip": P.E_CLIP,
                "g_clip": P.G_CLIP,
                "all_updates_applied": all_updates,
                "minimum_kept_fraction": float(np.min(np.asarray(kept_history))),
                "initial_10_step_loss_mean": initial_loss_mean,
                "final_10_step_loss_mean": final_loss_mean,
                "loss_change_final_minus_initial": loss_change,
                "initial_10_step_batch_ess_mean": float(np.mean(returned_ess[:10])),
                "final_10_step_batch_ess_mean": float(np.mean(returned_ess[-10:])),
            },
            "audit": {
                "n": P.N_SELECTION,
                "identity_ess": identity_ess,
                "final_ess": final_ess,
                "ess_delta": ess_delta,
                "block_identity_ess": block_identity_ess,
                "block_final_ess": block_final_ess,
                "positive_blocks": positive_blocks,
                "identity_holdout_loss_mean": float(np.mean(identity_holdout)),
                "final_holdout_loss_mean": float(np.mean(final_holdout)),
                "holdout_loss_delta": float(
                    np.mean(final_holdout) - np.mean(identity_holdout)
                ),
            },
            "decision": {
                "loss_and_ess_stuck": stuck,
                "progression_gate_passed": progression_gate,
                "ess_delta_threshold": P.AUDIT_DELTA_ESS_MIN,
                "positive_block_threshold": P.AUDIT_POSITIVE_BLOCKS_MIN,
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
                "parameters_sha256": sha256_file(HERE / "parameters.py"),
                "driver_sha256": sha256_file(Path(__file__)),
            },
            "artifacts": {
                "samples_h5_sha256": sha256_file(sample_path),
                "flow_eqx_sha256": sha256_file(flow_path),
            },
        }
        json_write(staging / "summary.json", summary)
        (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
        final.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
        log_path = final / "train.log"
        log(
            f"DONE identity_ESS={identity_ess:.8f} final_ESS={final_ess:.8f} "
            f"delta={ess_delta:+.8f} loss_change={loss_change:+.6g} "
            f"positive_blocks={positive_blocks}/{P.N_AUDIT_BLOCKS} "
            f"progression_gate={progression_gate} stuck={stuck}"
        )
    except BaseException:
        log("FAILED; staging directory retained")
        raise


if __name__ == "__main__":
    main()
