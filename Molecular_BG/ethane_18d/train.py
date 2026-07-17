#!/usr/bin/env python
"""Run full-size molecular KLXX with e/r regularization and save per-step ESS."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jflows.train import Monitor
from jflows_md import (
    Mixed_NSF,
    Molecular_Potential,
    boltzmann_forward_KLXX_G,
)

import parameters as P


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"


def _json_write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output() -> None:
    for path in (ARTIFACTS, RESULTS):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite nonempty experiment output: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)


def _finite_stat(values: np.ndarray, operation) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(operation(finite))


def _history_rows(stages: list[dict]) -> list[dict]:
    """Flatten every accepted level's retry/step histories without dropping retries."""

    rows: list[dict] = []
    for level, stage in enumerate(stages, start=1):
        batch_ess = np.asarray(stage["batch_ess_hist"], dtype=np.float32)
        kept = np.asarray(stage["kept_fraction_hist"], dtype=np.float32)
        updated = np.asarray(stage["update_applied_hist"], dtype=np.bool_)
        if batch_ess.ndim == 1:
            batch_ess = batch_ess[None, :]
            kept = kept[None, :]
            updated = updated[None, :]
        t_hist = np.asarray(stage["t_hist"], dtype=np.float32).reshape(-1)
        statuses = tuple(str(value) for value in stage["attempt_status_hist"])
        attempts = batch_ess.shape[0]
        if not (
            kept.shape == batch_ess.shape
            and updated.shape == batch_ess.shape
            and batch_ess.shape[1] == P.TRAIN_STEPS
            and t_hist.size == attempts
            and len(statuses) == attempts
        ):
            raise ValueError(
                f"inconsistent KLXX history shapes at level {level}: "
                f"ESS={batch_ess.shape}, kept={kept.shape}, "
                f"updated={updated.shape}, t={t_hist.shape}, "
                f"statuses={len(statuses)}"
            )
        for attempt in range(attempts):
            for step in range(P.TRAIN_STEPS):
                rows.append(
                    {
                        "level": level,
                        "attempt": attempt + 1,
                        "t": float(t_hist[attempt]),
                        "attempt_status": statuses[attempt],
                        "step": step + 1,
                        "batch_ess": float(batch_ess[attempt, step]),
                        "kept_fraction": float(kept[attempt, step]),
                        "update_applied": bool(updated[attempt, step]),
                    }
                )
    return rows


def _write_per_step_csv(rows: list[dict]) -> None:
    columns = (
        "level",
        "attempt",
        "t",
        "attempt_status",
        "step",
        "batch_ess",
        "kept_fraction",
        "update_applied",
    )
    with (RESULTS / "per_step_ess.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_history_npz(rows: list[dict], stages: list[dict]) -> None:
    np.savez_compressed(
        ARTIFACTS / "per_step_ess.npz",
        schema_version=np.int32(1),
        energy_threshold_kj_mol=np.float32(P.ENERGY_THRESHOLD_KJ_MOL),
        pair_distance_floor_nm=np.float32(P.PAIR_DISTANCE_FLOOR_NM),
        level=np.asarray([row["level"] for row in rows], dtype=np.int32),
        attempt=np.asarray([row["attempt"] for row in rows], dtype=np.int32),
        t=np.asarray([row["t"] for row in rows], dtype=np.float32),
        attempt_status=np.asarray(
            [row["attempt_status"] for row in rows], dtype="U16"
        ),
        step=np.asarray([row["step"] for row in rows], dtype=np.int32),
        batch_ess=np.asarray([row["batch_ess"] for row in rows], dtype=np.float32),
        kept_fraction=np.asarray(
            [row["kept_fraction"] for row in rows], dtype=np.float32
        ),
        update_applied=np.asarray(
            [row["update_applied"] for row in rows], dtype=np.bool_
        ),
        level_t=np.asarray([stage["t"] for stage in stages], dtype=np.float32),
        level_selected_ess=np.asarray(
            [stage["valid_selected_ess"] for stage in stages], dtype=np.float32
        ),
        level_trained_ess=np.asarray(
            [stage["valid_trained_ess"] for stage in stages], dtype=np.float32
        ),
        level_identity_ess=np.asarray(
            [stage["valid_identity_ess"] for stage in stages], dtype=np.float32
        ),
        level_selected=np.asarray([stage["selected"] for stage in stages], dtype="U8"),
    )


def _write_results(rows: list[dict], stages: list[dict]) -> list[dict]:
    lines = [
        f"# {P.MOLECULE.replace('_', ' ').title()} {P.DIMENSION}D KLXX with e/r regularization",
        "",
        f"- `energy_threshold_kj_mol` (e): `{P.ENERGY_THRESHOLD_KJ_MOL:g}` kJ/mol",
        f"- `pair_distance_floor_nm` (r): `{P.PAIR_DISTANCE_FLOOR_NM:g}` nm",
        f"- Full validation population: `{P.N_VALID}`",
        f"- Mixed MALA: `mc_dt={P.MC_DT:g}`, `mc_steps={P.MC_STEPS}`",
        "- `per_step_ess.csv` records the pre-update KLXX batch ESS at every optimizer step.",
        "",
        "## Full-validation stage ESS",
        "",
        "| Level | t | Selected | Selected ESS | Trained ESS | Identity ESS |",
        "|---:|---:|:---:|---:|---:|---:|",
    ]
    for level, stage in enumerate(stages, start=1):
        lines.append(
            f"| {level} | {float(stage['t']):.6f} | {stage['selected']} | "
            f"{float(stage['valid_selected_ess']):.6f} | "
            f"{float(stage['valid_trained_ess']):.6f} | "
            f"{float(stage['valid_identity_ess']):.6f} |"
        )

    lines.extend(
        (
            "",
            "## Per-attempt optimizer ESS",
            "",
            "| Level | Attempt | t | Status | Steps | First | Last | Minimum | Maximum | Mean | Updates |",
            "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    attempt_summaries = []
    keys = sorted({(row["level"], row["attempt"]) for row in rows})
    for level, attempt in keys:
        selected = [
            row
            for row in rows
            if row["level"] == level and row["attempt"] == attempt
        ]
        ess = np.asarray([row["batch_ess"] for row in selected], dtype=np.float32)
        updates = int(sum(row["update_applied"] for row in selected))
        record = {
            "level": level,
            "attempt": attempt,
            "t": selected[0]["t"],
            "status": selected[0]["attempt_status"],
            "steps": len(selected),
            "first_batch_ess": float(ess[0]),
            "last_batch_ess": float(ess[-1]),
            "minimum_batch_ess": _finite_stat(ess, np.min),
            "maximum_batch_ess": _finite_stat(ess, np.max),
            "mean_batch_ess": _finite_stat(ess, np.mean),
            "updates_applied": updates,
        }
        attempt_summaries.append(record)
        format_stat = lambda value: "nonfinite" if value is None else f"{value:.6f}"
        lines.append(
            f"| {level} | {attempt} | {record['t']:.6f} | {record['status']} | "
            f"{record['steps']} | {record['first_batch_ess']:.6f} | "
            f"{record['last_batch_ess']:.6f} | "
            f"{format_stat(record['minimum_batch_ess'])} | "
            f"{format_stat(record['maximum_batch_ess'])} | "
            f"{format_stat(record['mean_batch_ess'])} | {updates} |"
        )
    (RESULTS / "ess.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return attempt_summaries


def main() -> None:
    _prepare_output()
    status_path = ARTIFACTS / "train.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with status_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started = time.time()
    physical = Molecular_Potential.from_bundle(
        BUNDLE,
        temperature_kelvin=P.TEMPERATURE_KELVIN,
    )
    if physical.dimension != P.DIMENSION:
        raise ValueError(f"bundle dimension {physical.dimension} != {P.DIMENSION}")
    target = physical.regularized(
        P.ENERGY_THRESHOLD_KJ_MOL,
        pair_distance_floor_nm=P.PAIR_DISTANCE_FLOOR_NM,
    )
    source = physical.source()
    reference = physical.reference_internal()[None, :]
    physical_reference = float(physical.physical_energy(reference)[0])
    floor_reference = float(target.reference_energy_kj_mol)
    surrogate_reference = float(target.regularized_energy(reference)[0])

    log(
        f"START {P.MOLECULE} {P.DIMENSION}D full adaptive KLXX | "
        f"e={P.ENERGY_THRESHOLD_KJ_MOL:g} "
        f"kJ/mol r={P.PAIR_DISTANCE_FLOOR_NM:g} nm | "
        f"jax={jax.__version__} backend={jax.default_backend()} dtype=float32"
    )
    log(
        f"reference energies [kJ/mol]: physical={physical_reference:.8g} "
        f"floor-aware={floor_reference:.8g} surrogate={surrogate_reference:.8g}"
    )
    log(
        f"N_VALID={P.N_VALID} POOL={'full' if P.POOL_SIZE == 0 else P.POOL_SIZE} "
        f"BATCH_SIZE={P.BATCH_SIZE} TRAIN_STEPS={P.TRAIN_STEPS} "
        f"initialize_from_identity={P.INITIALIZE_FROM_IDENTITY}"
    )

    source_key, flow_key = jax.random.split(jax.random.key(P.SEED))
    x_valid = source.samples(source_key, N=P.N_VALID)
    if x_valid.dtype != jnp.float32:
        raise TypeError(f"validation population is {x_valid.dtype}, expected float32")
    flow = Mixed_NSF(
        flow_key,
        physical.domain,
        bins=P.BINS,
        transforms=P.TRANSFORMS,
        euclidean_bound=P.NSF_LIM,
        hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE,
        mask_strategy="balanced",
    ).zeros()
    monitor = Monitor(P.MONITOR_EVERY, f"[{P.MOLECULE}] ", log)
    particles, stages = boltzmann_forward_KLXX_G(
        x_valid,
        source,
        target,
        flow,
        pool_size=P.POOL_SIZE,
        batch_size=P.BATCH_SIZE,
        train_steps=P.TRAIN_STEPS,
        lr=P.LR,
        ladder=P.LADDER,
        mc_dt=P.MC_DT,
        mc_steps=P.MC_STEPS,
        melt=P.MELT,
        opt_alpha=P.OPT_ALPHA,
        opt_steps=P.OPT_STEPS,
        initialize_from_identity=P.INITIALIZE_FROM_IDENTITY,
        coeff_lambda=P.COEFF_LAMBDA,
        coeff_alpha=P.COEFF_ALPHA,
        coeff_beta=P.COEFF_BETA,
        monitor=monitor,
        bg_param=P.BG_PARAM,
        chunks=P.CHUNKS,
        mc_image_radius=P.MC_IMAGE_RADIUS,
        e_clip=P.E_CLIP,
        g_clip=P.G_CLIP,
        seed=P.SEED,
        checkpoint=P.CHECKPOINT,
        lr_warmup=P.LR_WARMUP,
        flow_dir=ARTIFACTS / "flows",
    )
    particles = np.asarray(jax.block_until_ready(particles), dtype=np.float32)
    complete = bool(stages and math.isclose(float(stages[-1]["t"]), 1.0))
    rows = _history_rows(stages)
    _write_per_step_csv(rows)
    _write_history_npz(rows, stages)
    attempt_summaries = _write_results(rows, stages)
    np.savez_compressed(
        ARTIFACTS / "final_validation_particles.npz",
        particles=particles,
        sample_kind=np.asarray("final accepted bridge particles"),
    )
    summary = {
        "schema_version": 1,
        "complete": complete,
        "molecule": P.MOLECULE,
        "formula": P.FORMULA,
        "dimension": P.DIMENSION,
        "target": {
            "kind": "energy/distance-regularized surrogate",
            "physical_endpoint_preserved_separately": True,
            "temperature_kelvin": P.TEMPERATURE_KELVIN,
            "energy_threshold_kj_mol": P.ENERGY_THRESHOLD_KJ_MOL,
            "pair_distance_floor_nm": P.PAIR_DISTANCE_FLOOR_NM,
            "physical_reference_energy_kj_mol": physical_reference,
            "floor_reference_energy_kj_mol": floor_reference,
            "surrogate_reference_energy_kj_mol": surrogate_reference,
        },
        "training": {
            "objective": "adaptive KLXX",
            "n_valid": P.N_VALID,
            "pool_size": P.POOL_SIZE,
            "batch_size": P.BATCH_SIZE,
            "train_steps": P.TRAIN_STEPS,
            "lr": P.LR,
            "lr_warmup": P.LR_WARMUP,
            "ladder": P.LADDER,
            "mc_dt": P.MC_DT,
            "mc_steps": P.MC_STEPS,
            "mc_image_radius": P.MC_IMAGE_RADIUS,
            "chunks": P.CHUNKS,
            "melt": P.MELT,
            "opt_alpha": P.OPT_ALPHA,
            "opt_steps": P.OPT_STEPS,
            "coeff_lambda": P.COEFF_LAMBDA,
            "coeff_alpha": P.COEFF_ALPHA,
            "coeff_beta": P.COEFF_BETA,
            "initialize_from_identity": P.INITIALIZE_FROM_IDENTITY,
            "e_clip": P.E_CLIP,
            "g_clip": P.G_CLIP,
            "bg_param": P.BG_PARAM,
        },
        "attempts": attempt_summaries,
        "stages": [
            {
                "level": level,
                "t": float(stage["t"]),
                "selected": stage["selected"],
                "valid_selected_ess": float(stage["valid_selected_ess"]),
                "valid_trained_ess": float(stage["valid_trained_ess"]),
                "valid_identity_ess": float(stage["valid_identity_ess"]),
                "valid_sample_count": int(stage["valid_sample_count"]),
            }
            for level, stage in enumerate(stages, start=1)
        ],
        "runtime": {
            "wall_seconds": time.time() - started,
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
    }
    _json_write(ARTIFACTS / "summary.json", summary)
    log(
        f"DONE complete={complete} levels={len(stages)} "
        f"per_step_rows={len(rows)} results={RESULTS}"
    )
    if not complete:
        reached = float(stages[-1]["t"]) if stages else 0.0
        raise RuntimeError(f"adaptive KLXX ladder incomplete at t={reached}")


if __name__ == "__main__":
    main()
