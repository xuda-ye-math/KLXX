#!/usr/bin/env python
"""Compare CH4 optimizer step size and batch size on one frozen data set."""

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
from jflows_md import Mixed_NSF, Molecular_Potential, train_molecular_forward_KLX_G

import parameters as P
from train_alkanes import chunked_log_weight, chunked_target_loss, normalized_ess


HERE = Path(__file__).resolve().parent
CASES = (
    ("lr_0p003_batch_50000", 3e-3, 50000),
    ("lr_0p01_batch_50000", 1e-2, 50000),
    ("lr_0p001_batch_100000", 1e-3, 100000),
)
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


def threshold_steps(history: np.ndarray) -> dict[str, int | None]:
    result = {}
    for threshold in (0.05, 0.10, 0.20):
        indices = np.flatnonzero(history >= threshold)
        result[f"ess_{threshold:.2f}"] = int(indices[0] + 1) if indices.size else None
    return result


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
    log_path = staging / "tuning.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    baseline = json.loads((source_run / "summary.json").read_text())
    if baseline["molecule"] != "methane" or baseline["target"]["id"] != P.TARGET_ID:
        raise ValueError("the input must be the completed CH4 c50_rho001 baseline")
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
    with h5py.File(source_run / "samples.h5", "r") as handle:
        baseline_ess_history = np.asarray(handle["ess_history"])
    results = [
        {
            "name": "lr_0p001_batch_50000_baseline",
            "lr": P.LR,
            "n_batch": P.N_BATCH,
            "steps": baseline["training"]["steps"],
            "audit_ess": baseline["audit"]["final_ess"],
            "audit_ess_delta": baseline["audit"]["ess_delta"],
            "holdout_loss_mean": baseline["audit"]["final_holdout_loss_mean"],
            "initial_10_step_loss_mean": baseline["training"]["initial_10_step_loss_mean"],
            "final_10_step_loss_mean": baseline["training"]["final_10_step_loss_mean"],
            "initial_10_step_batch_ess_mean": baseline["training"]["initial_10_step_batch_ess_mean"],
            "final_10_step_batch_ess_mean": baseline["training"]["final_10_step_batch_ess_mean"],
            "threshold_steps": threshold_steps(baseline_ess_history),
            "source_run": str(source_run),
        }
    ]

    for name, lr, n_batch in CASES:
        case_path = staging / name
        case_path.mkdir()
        messages: list[str] = []

        def monitor(message: str) -> None:
            messages.append(message)
            log(f"[{name}] {message}")

        flow0 = Mixed_NSF(
            jax.random.key(P.FLOW_SEED),
            physical.domain,
            bins=P.BINS,
            transforms=P.TRANSFORMS,
            euclidean_bound=P.EUCLIDEAN_BOUND,
            hidden_features=P.HIDDEN_FEATURES,
            slope=P.SLOPE,
        ).zeros()
        started = time.time()
        flow, ess, kept, updated = train_molecular_forward_KLX_G(
            target_train,
            source_train,
            source,
            target,
            flow0,
            n_batch=n_batch,
            steps=P.STEPS,
            lr=lr,
            coeff_lambda=0.0,
            energy_origin=energy_origin,
            e_clip=P.E_CLIP,
            g_clip=P.G_CLIP,
            monitor=Monitor(1, "", monitor),
            seed=P.TRAINER_SEED,
            lr_warmup=P.LR_WARMUP,
        )
        jax.block_until_ready((flow, ess, kept, updated))
        jax.effects_barrier()
        final_log_weight = chunked_log_weight(
            source_audit, source, target, flow, P.CHUNK
        )
        holdout_loss = chunked_target_loss(
            target_holdout, source, target, flow, P.CHUNK
        )
        records = []
        for message in messages:
            match = PATTERN.search(message)
            if match:
                records.append(
                    (int(match["step"]), float(match["loss"]), float(match["ess"]))
                )
        records.sort()
        if len(records) != P.STEPS:
            raise RuntimeError(f"{name}: expected {P.STEPS} monitor records")
        loss = np.asarray([record[1] for record in records])
        ess_np = np.asarray(ess)
        if not np.allclose(
            ess_np, np.asarray([record[2] for record in records]),
            atol=5e-5, rtol=5e-4,
        ):
            raise RuntimeError(f"{name}: printed and returned ESS differ")
        audit_ess = normalized_ess(final_log_weight)
        result = {
            "name": name,
            "lr": lr,
            "n_batch": n_batch,
            "steps": P.STEPS,
            "audit_ess": audit_ess,
            "audit_ess_delta": audit_ess - baseline["audit"]["identity_ess"],
            "holdout_loss_mean": float(np.mean(holdout_loss)),
            "initial_10_step_loss_mean": float(np.mean(loss[:10])),
            "final_10_step_loss_mean": float(np.mean(loss[-10:])),
            "initial_10_step_batch_ess_mean": float(np.mean(ess_np[:10])),
            "final_10_step_batch_ess_mean": float(np.mean(ess_np[-10:])),
            "threshold_steps": threshold_steps(ess_np),
            "all_updates_applied": bool(np.asarray(updated).all()),
            "minimum_kept_fraction": float(np.min(np.asarray(kept))),
            "wall_seconds": time.time() - started,
        }
        results.append(result)
        with h5py.File(case_path / "result.h5", "w") as handle:
            handle.create_dataset("loss", data=loss)
            handle.create_dataset("ess_history", data=ess_np)
            handle.create_dataset("kept_history", data=np.asarray(kept))
            handle.create_dataset("update_history", data=np.asarray(updated))
            handle.create_dataset(
                "final_log_weight", data=final_log_weight,
                compression="gzip", compression_opts=1, shuffle=True,
            )
            handle.create_dataset(
                "holdout_loss", data=holdout_loss,
                compression="gzip", compression_opts=1, shuffle=True,
            )
        eqx.tree_serialise_leaves(case_path / "flow.eqx", flow)
        log(
            f"RESULT {name} audit_ESS={audit_ess:.8f} "
            f"loss_10={result['final_10_step_loss_mean']:.6g} "
            f"batch_ESS_10={result['final_10_step_batch_ess_mean']:.6f}"
        )

    summary = {
        "schema_version": 1,
        "complete": True,
        "question": "CH4 convergence per optimizer iteration",
        "source_run": str(source_run),
        "source_summary_sha256": sha256(source_run / "summary.json"),
        "source_samples_sha256": sha256(source_run / "samples.h5"),
        "same_target_source_audit_and_seeds": True,
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
    print(f"PASS CH4 optimizer sweep: {output}", flush=True)


if __name__ == "__main__":
    main()
