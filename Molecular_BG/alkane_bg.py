"""Shared full adaptive-KLXX driver for the three small-alkane tests."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from jflows.train import Monitor
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann import boltzmann_forward_KLXX_G


def _json_write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@eqx.filter_jit
def _inverse_chunk(flow, samples):
    return flow.inv(samples)


@eqx.filter_jit
def _physical_energy_chunk(target, samples):
    return target.physical_energy(samples)


def _flow_chain_samples(source, stages, key, sample_count: int, chunks: int):
    samples = source.samples(key, N=sample_count)
    if samples.dtype != jnp.float32:
        raise TypeError(f"source produced {samples.dtype}, expected float32")
    for stage in stages:
        pieces = []
        for part in jnp.array_split(samples, chunks, axis=0):
            pieces.append(jax.block_until_ready(_inverse_chunk(stage["flow"], part)))
        samples = jnp.concatenate(pieces, axis=0)
    return np.asarray(jax.block_until_ready(samples), dtype=np.float32)


def _physical_energy(target, samples: np.ndarray, chunks: int) -> np.ndarray:
    pieces = []
    for part in np.array_split(samples, chunks, axis=0):
        value = jax.block_until_ready(
            _physical_energy_chunk(target, jnp.asarray(part))
        )
        pieces.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(pieces)


def _write_ess_table(path: Path, summary: dict) -> None:
    lines = [
        "# Full-validation ESS",
        "",
        "| Level | t | Selected | Validation ESS | Trained ESS | Identity ESS |",
        "|---:|---:|:---:|---:|---:|---:|",
    ]
    for record in summary["ladder"]:
        lines.append(
            f"| {record['level']} | {record['t']:.6f} | {record['selected']} | "
            f"{record['valid_selected_ess']:.6f} | "
            f"{record['valid_trained_ess']:.6f} | "
            f"{record['valid_identity_ess']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(parameters) -> None:
    here = Path(parameters.__file__).resolve().parent
    bundle_path = here / "bundle"
    final_artifacts = here / "artifacts"
    final_results = here / "results"
    existing = [path for path in (final_artifacts, final_results) if path.exists()]
    for path in existing:
        shutil.rmtree(path)
    final_artifacts.mkdir()
    final_results.mkdir()
    status_path = final_artifacts / "train.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with status_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started = time.time()
    physical = Molecular_Potential.from_bundle(bundle_path, temperature_kelvin=300.0)
    if physical.dimension != parameters.DIMENSION:
        raise ValueError(
            f"bundle dimension {physical.dimension} != {parameters.DIMENSION}"
        )
    target = physical.regularized(
        parameters.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=parameters.ENERGY_SCALE_KJ_MOL,
        tail_fraction=parameters.TAIL_FRACTION,
    )
    source = physical.source()
    log(
        f"START {parameters.MOLECULE}_{parameters.DIMENSION}d_c50 full adaptive "
        f"KLXX | jax={jax.__version__} backend={jax.default_backend()} "
        f"dtype=float32 domain=R{physical.domain.euclidean_dim}"
        f"xT{physical.domain.periodic_dim}"
    )
    log(
        f"flow balanced bins={parameters.BINS} "
        f"transforms={parameters.TRANSFORMS} hidden={parameters.HIDDEN_FEATURES} | "
        f"N_VALID={parameters.N_VALID} POOL_SIZE={parameters.POOL_SIZE} "
        f"BATCH_SIZE={parameters.BATCH_SIZE} train_steps={parameters.TRAIN_STEPS} "
        f"lr={parameters.LR} bg={parameters.BG_PARAM}"
    )

    source_key, flow_key, output_key = jax.random.split(
        jax.random.key(parameters.SEED), 3
    )
    x_valid = source.samples(source_key, N=parameters.N_VALID)
    if x_valid.dtype != jnp.float32:
        raise TypeError(f"validation population is {x_valid.dtype}, expected float32")
    flow0 = Mixed_NSF(
        flow_key,
        physical.domain,
        bins=parameters.BINS,
        transforms=parameters.TRANSFORMS,
        euclidean_bound=parameters.NSF_LIM,
        hidden_features=parameters.HIDDEN_FEATURES,
        slope=parameters.SLOPE,
        mask_strategy="balanced",
    ).zeros()
    monitor = Monitor(parameters.MONITOR_EVERY, f"[{parameters.MOLECULE}] ", log)
    particles, stages = boltzmann_forward_KLXX_G(
        x_valid,
        source,
        target,
        flow0,
        pool_size=parameters.POOL_SIZE,
        batch_size=parameters.BATCH_SIZE,
        train_steps=parameters.TRAIN_STEPS,
        lr=parameters.LR,
        ladder=parameters.LADDER,
        mc_dt=parameters.MC_DT,
        mc_steps=parameters.MC_STEPS,
        melt=parameters.MELT,
        opt_alpha=parameters.OPT_ALPHA,
        opt_steps=parameters.OPT_STEPS,
        coeff_lambda=parameters.COEFF_LAMBDA,
        coeff_alpha=parameters.COEFF_ALPHA,
        coeff_beta=parameters.COEFF_BETA,
        monitor=monitor,
        bg_param=parameters.BG_PARAM,
        chunks=parameters.CHUNKS,
        mc_image_radius=parameters.MC_IMAGE_RADIUS,
        e_clip=parameters.E_CLIP,
        g_clip=parameters.G_CLIP,
        seed=parameters.SEED,
        checkpoint=parameters.CHECKPOINT,
        lr_warmup=parameters.LR_WARMUP,
        flow_dir=final_artifacts / "flows",
    )
    jax.block_until_ready(particles)
    if not stages or float(stages[-1]["t"]) != 1.0:
        reached = float(stages[-1]["t"]) if stages else 0.0
        raise RuntimeError(f"adaptive ladder incomplete at t={reached}")

    flow_samples = _flow_chain_samples(
        source, stages, output_key, parameters.N_VALID, parameters.CHUNKS
    )
    flow_energy = _physical_energy(physical, flow_samples, parameters.CHUNKS)
    reference_energy = np.asarray(
        target.reference_energy_kj_mol, dtype=np.float32
    ).item()
    cap_active_fraction = float(
        np.mean(flow_energy - np.float32(reference_energy) > parameters.ENERGY_CUT_KJ_MOL)
    )

    batch_ess_hist = np.concatenate(
        [np.asarray(stage["batch_ess_hist"], dtype=np.float32).reshape(-1) for stage in stages]
    )
    kept_fraction_hist = np.concatenate(
        [
            np.asarray(stage["kept_fraction_hist"], dtype=np.float32).reshape(-1)
            for stage in stages
        ]
    )
    update_applied_hist = np.concatenate(
        [
            np.asarray(stage["update_applied_hist"], dtype=np.bool_).reshape(-1)
            for stage in stages
        ]
    )
    level_t = np.asarray([stage["t"] for stage in stages], dtype=np.float32)
    level_selected_ess = np.asarray(
        [stage["valid_selected_ess"] for stage in stages], dtype=np.float32
    )
    level_trained_ess = np.asarray(
        [stage["valid_trained_ess"] for stage in stages], dtype=np.float32
    )
    level_identity_ess = np.asarray(
        [stage["valid_identity_ess"] for stage in stages], dtype=np.float32
    )
    level_selected = np.asarray([stage["selected"] for stage in stages])
    np.savez_compressed(
        final_artifacts / "training_data.npz",
        schema_version=np.int32(1),
        bundle=np.asarray("../bundle"),
        stage_count=np.int32(len(stages)),
        bins=np.int32(parameters.BINS),
        transforms=np.int32(parameters.TRANSFORMS),
        hidden_features=np.asarray(parameters.HIDDEN_FEATURES, dtype=np.int32),
        slope=np.float32(parameters.SLOPE),
        nsf_lim=np.float32(parameters.NSF_LIM),
        flow_key_data=np.asarray(jax.random.key_data(flow_key), dtype=np.uint32),
        output_key_data=np.asarray(jax.random.key_data(output_key), dtype=np.uint32),
        jax_version=np.asarray(jax.__version__),
        equinox_version=np.asarray(eqx.__version__),
        level_t=level_t,
        level_selected_ess=level_selected_ess,
        level_trained_ess=level_trained_ess,
        level_identity_ess=level_identity_ess,
        level_selected=level_selected,
        batch_ess_hist=batch_ess_hist,
        kept_fraction_hist=kept_fraction_hist,
        update_applied_hist=update_applied_hist,
    )
    common_attributes = {
        "molecule": parameters.MOLECULE,
        "temperature_kelvin": np.float32(300.0),
        "energy_cut_kj_mol": np.float32(parameters.ENERGY_CUT_KJ_MOL),
        "floating_point": "float32",
    }
    with h5py.File(final_artifacts / "flow_samples.h5", "w") as handle:
        handle.create_dataset(
            "particles",
            data=flow_samples,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        handle.create_dataset(
            "physical_energy_kj_mol",
            data=flow_energy,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        for name, value in common_attributes.items():
            handle.attrs[name] = value
        handle.attrs["sample_kind"] = "raw composed inverse-flow proposal"
        handle.attrs["sample_count"] = parameters.N_VALID

    ladder_summary = []
    for index, stage in enumerate(stages):
        record = {
            "level": index + 1,
            "t": float(stage["t"]),
            "selected": stage["selected"],
            "valid_selected_ess": float(stage["valid_selected_ess"]),
            "valid_trained_ess": float(stage["valid_trained_ess"]),
            "valid_identity_ess": float(stage["valid_identity_ess"]),
            "valid_sample_count": int(stage["valid_sample_count"]),
            "selected_flow_path": stage["selected_flow_path"],
        }
        ladder_summary.append(record)
        log(
            f"RESULT level={record['level']} t={record['t']:.4f} "
            f"validation_ESS[N={record['valid_sample_count']}]="
            f"{record['valid_selected_ess']:.6f} selected={record['selected']} "
            f"trained={record['valid_trained_ess']:.6f} "
            f"identity={record['valid_identity_ess']:.6f}"
        )
    summary = {
        "schema_version": 1,
        "complete": True,
        "molecule": parameters.MOLECULE,
        "formula": parameters.FORMULA,
        "dimension": parameters.DIMENSION,
        "domain": {
            "euclidean": physical.domain.euclidean_dim,
            "periodic": physical.domain.periodic_dim,
        },
        "target": {
            "temperature_kelvin": 300.0,
            "energy_cut_kj_mol": parameters.ENERGY_CUT_KJ_MOL,
            "energy_scale_kj_mol": parameters.ENERGY_SCALE_KJ_MOL,
            "tail_fraction": parameters.TAIL_FRACTION,
            "flow_cap_active_fraction": cap_active_fraction,
        },
        "acceptance_rule": {
            "quantity": "per-level full-validation ESS",
            "tau_ess": parameters.BG_PARAM["tau_ess"],
            "n_valid": parameters.N_VALID,
        },
        "ladder": ladder_summary,
        "training": {
            "objective": "adaptive KLXX",
            "n_valid": parameters.N_VALID,
            "pool_size": parameters.POOL_SIZE,
            "batch_size": parameters.BATCH_SIZE,
            "train_steps_per_level": parameters.TRAIN_STEPS,
            "lr": parameters.LR,
            "minimum_kept_fraction": float(kept_fraction_hist.min()),
            "update_fraction": float(update_applied_hist.mean()),
            "bg_param": parameters.BG_PARAM,
        },
        "runtime": {
            "wall_seconds": time.time() - started,
            "jax": jax.__version__,
            "floating_point": "float32",
            "devices": [str(device) for device in jax.devices()],
        },
    }
    summary_path = final_artifacts / "summary.json"
    _json_write(summary_path, summary)
    ess_table_path = final_results / "ess.md"
    _write_ess_table(ess_table_path, summary)
    log(
        f"DONE adaptive ladder complete ({len(stages)} levels); "
        f"final level validation ESS={level_selected_ess[-1]:.6f}; "
        f"flow cap-active fraction={cap_active_fraction:.6g}"
    )
    print(f"PASS {parameters.MOLECULE}: {final_results}", flush=True)


def main(parameters) -> None:
    run(parameters)
