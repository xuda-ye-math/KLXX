#!/usr/bin/env python
"""One-step diagnostics for the archived singular level-2 KLXX inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import linear_combination
from jflows.train import Monitor
from jflows.utils import compute_ESS_log, importance_weights_log
from jflows_md import Molecular_Potential, load_mixed_flow
from jflows_md.train import train_forward_KLXX_G

import parameters as P


LOSS_PATTERN = re.compile(
    r"loss = (?P<loss>[+-][0-9.]+e[+-][0-9]+)\s+ESS = (?P<ess>[0-9.]+)"
)

CASES = {
    "warm_b40000_lr1e-4": ("warm", 40000, 1e-4),
    "identity_b40000_lr1e-4": ("identity", 40000, 1e-4),
    "warm_b20000_lr1e-4": ("warm", 20000, 1e-4),
    "warm_b10000_lr1e-4": ("warm", 10000, 1e-4),
    "identity_b10000_lr1e-4": ("identity", 10000, 1e-4),
    "warm_b10000_lr1e-5": ("warm", 10000, 1e-5),
    "warm_b10000_lr1e-6": ("warm", 10000, 1e-6),
}


def full_ess(samples, source, target, flow) -> float:
    log_weight = importance_weights_log(
        samples, source, target, flow, type="G", chunks=P.CHUNKS
    )
    return float(compute_ESS_log(log_weight))


def parameter_delta(before, after) -> tuple[float, float]:
    differences = []
    for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        if hasattr(left, "dtype") and jnp.issubdtype(left.dtype, jnp.inexact):
            differences.append(jnp.ravel(right - left))
    flat = jnp.concatenate(differences)
    return float(jnp.linalg.norm(flat)), float(jnp.max(jnp.abs(flat)))


def one_step(
    *,
    label: str,
    flow,
    target_samples,
    source_samples,
    hat_samples,
    source,
    target,
    domain,
    energy_origin: float,
    trainer_seed: int,
    batch_size: int,
    lr: float,
) -> dict:
    messages: list[str] = []
    initial_ess = full_ess(source_samples, source, target, flow)
    started = time.time()
    candidate, ess_hist, kept_hist, update_hist = train_forward_KLXX_G(
        target_samples,
        source_samples,
        hat_samples,
        source,
        target,
        flow,
        domain,
        batch_size=batch_size,
        train_steps=1,
        lr=lr,
        coeff_lambda=P.COEFF_LAMBDA,
        coeff_alpha=P.COEFF_ALPHA,
        coeff_beta=P.COEFF_BETA,
        mc_dt=P.MC_DT,
        mc_steps=P.MC_STEPS,
        mc_image_radius=P.MC_IMAGE_RADIUS,
        energy_origin=energy_origin,
        e_clip=P.E_CLIP,
        g_clip=P.G_CLIP,
        monitor=Monitor(1, f"[{label}] ", messages.append),
        checkpoint=P.CHECKPOINT,
        lr_warmup=P.LR_WARMUP,
        seed=trainer_seed,
    )
    candidate = jax.block_until_ready(candidate)
    jax.effects_barrier()
    elapsed = time.time() - started
    match = LOSS_PATTERN.search(messages[-1]) if messages else None
    delta_l2, delta_max = parameter_delta(flow, candidate)
    result = {
        "label": label,
        "start": "identity" if label.startswith("identity") else "warm",
        "batch_size": batch_size,
        "lr": lr,
        "effective_first_lr": lr / P.LR_WARMUP,
        "pre_step_full_ess": initial_ess,
        "pre_step_batch_ess": float(ess_hist[0]),
        "pre_step_loss": float(match.group("loss")) if match else None,
        "kept_fraction": float(kept_hist[0]),
        "update_applied": bool(update_hist[0]),
        "post_step_full_ess": full_ess(source_samples, source, target, candidate),
        "parameter_delta_l2": delta_l2,
        "parameter_delta_max": delta_max,
        "elapsed_seconds": elapsed,
        "monitor": messages[-1] if messages else None,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(CASES))
    arguments = parser.parse_args()
    here = Path(__file__).resolve().parent
    reproduction = here / "reproduction"
    metadata = json.loads(
        (reproduction / "stage2_t0175.json").read_text(encoding="utf-8")
    )
    with h5py.File(reproduction / metadata["data"], "r") as handle:
        source_samples = jnp.asarray(handle["stage1_particles"][:])
        target_samples = jnp.asarray(handle["stage2_smc_particles"][:])
        hat_samples = jnp.asarray(handle["stage2_hat_particles"][:])

    physical = Molecular_Potential.from_bundle(
        here / "bundle", temperature_kelvin=300.0
    )
    molecular_target = physical.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = physical.source()
    previous_t = float(metadata["previous_t"])
    failed_t = float(metadata["failed_t"])
    previous = linear_combination(
        [molecular_target, source], [previous_t, 1.0 - previous_t]
    )
    current = linear_combination(
        [molecular_target, source], [failed_t, 1.0 - failed_t]
    )
    warm = load_mixed_flow(reproduction / metadata["accepted_flow"])
    identity = warm.zeros()

    start, batch_size, lr = CASES[arguments.case]
    result = one_step(
        label=arguments.case,
        flow=identity if start == "identity" else warm,
        target_samples=target_samples,
        source_samples=source_samples,
        hat_samples=hat_samples,
        source=previous,
        target=current,
        domain=physical.domain,
        energy_origin=float(metadata["energy_origin"]),
        trainer_seed=int(metadata["trainer_seed"]),
        batch_size=batch_size,
        lr=lr,
    )

    output_path = reproduction / "one_step_results.json"
    if output_path.is_file():
        output = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        output = {
            "schema_version": 1,
            "data": metadata["data"],
            "previous_t": previous_t,
            "failed_t": failed_t,
            "train_steps": 1,
            "lr_warmup": P.LR_WARMUP,
            "cases": [],
        }
    output["cases"] = [
        case for case in output["cases"] if case["label"] != arguments.case
    ]
    output["cases"].append(result)
    output["cases"].sort(key=lambda case: case["label"])
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
