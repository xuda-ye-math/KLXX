#!/usr/bin/env python
"""Reconstruct and save the accepted level-1 state and failed level-2 pools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import linear_combination
from jflows.utils import (
    compute_ESS_log,
    importance_weights_log,
    linear_weights_from_log,
    resample,
)
from jflows_md import (
    Molecular_Potential,
    load_mixed_flow,
    mixed_mala,
    mixed_quench_and_temper,
    sequential_monte_carlo,
)

import parameters as P


PREVIOUS_T = 0.07
FAILED_T = 0.175


def operation_key(base_key, namespace: int, *indices: int):
    """Mirror the deterministic public-controller key partition."""

    key = jax.random.fold_in(base_key, namespace)
    for index in indices:
        key = jax.random.fold_in(key, index)
    return key


def as_float32(value):
    return np.asarray(jax.block_until_ready(value), dtype=np.float32)


@eqx.filter_jit
def inverse_chunk(flow, samples):
    return flow.inv(samples)


@eqx.filter_jit
def identity_weight_chunk(samples, previous, current):
    return previous(samples) - current(samples)


def chunked_inverse(flow, samples):
    values = [
        jax.block_until_ready(inverse_chunk(flow, part))
        for part in jnp.array_split(samples, P.CHUNKS, axis=0)
    ]
    return jnp.concatenate(values, axis=0)


def chunked_identity_weight(samples, previous, current):
    values = [
        jax.block_until_ready(identity_weight_chunk(part, previous, current))
        for part in jnp.array_split(samples, P.CHUNKS, axis=0)
    ]
    return jnp.concatenate(values, axis=0)


def main() -> None:
    here = Path(__file__).resolve().parent
    output_dir = here / "reproduction"
    output_dir.mkdir(exist_ok=True)
    data_path = output_dir / "stage2_t0175.h5"
    metadata_path = output_dir / "stage2_t0175.json"

    physical = Molecular_Potential.from_bundle(
        here / "bundle", temperature_kelvin=300.0
    )
    target = physical.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = physical.source()
    previous = linear_combination(
        [target, source], [PREVIOUS_T, 1.0 - PREVIOUS_T]
    )
    current = linear_combination(
        [target, source], [FAILED_T, 1.0 - FAILED_T]
    )

    source_key, _, _ = jax.random.split(jax.random.key(P.SEED), 3)
    initial_particles = source.samples(source_key, N=P.N_VALID)
    flow_path = here / "artifacts" / "flows" / "stage_0001" / "selected_flow.eqx"
    flow = load_mixed_flow(flow_path)
    archived_flow_path = output_dir / "stage1_selected_flow.eqx"
    shutil.copy2(flow_path, archived_flow_path)
    shutil.copy2(
        flow_path.with_suffix(flow_path.suffix + ".json"),
        archived_flow_path.with_suffix(archived_flow_path.suffix + ".json"),
    )

    stage1_log_weight = importance_weights_log(
        initial_particles,
        source,
        previous,
        flow,
        type="G",
        chunks=P.CHUNKS,
    )
    stage1_ess = float(compute_ESS_log(stage1_log_weight))
    proposal = chunked_inverse(flow, initial_particles)
    base_key = jax.random.fold_in(jax.random.key(41), P.SEED)
    resample_key, mala_key = jax.random.split(operation_key(base_key, 3, 1))
    stage1_particles = resample(
        resample_key,
        proposal,
        linear_weights_from_log(stage1_log_weight),
        N=P.N_VALID,
    )
    stage1_particles, stage1_mala = mixed_mala(
        mala_key,
        stage1_particles,
        previous,
        physical.domain,
        dt=P.MC_DT,
        steps=P.MC_STEPS,
        image_radius=P.MC_IMAGE_RADIUS,
        chunks=P.CHUNKS,
    )
    stage1_particles = jax.block_until_ready(stage1_particles)

    selection_base = operation_key(base_key, 1, 2)
    stage2_smc, stage2_smc_ess, stage2_smc_mala = sequential_monte_carlo(
        jax.random.fold_in(selection_base, 2),
        stage1_particles,
        previous,
        current,
        ladder=P.LADDER,
        mc_dt=P.MC_DT,
        mc_steps=P.MC_STEPS,
        mc_image_radius=P.MC_IMAGE_RADIUS,
        domain=physical.domain,
        chunks=P.CHUNKS,
    )
    stage2_smc = jax.block_until_ready(stage2_smc)

    qt_seed, qt_key = jax.random.split(operation_key(base_key, 4, 2, 1))
    qt_initial = source.samples(qt_seed, N=P.N_VALID)
    stage2_hat, stage2_hat_mala = mixed_quench_and_temper(
        qt_key,
        qt_initial,
        current,
        physical.domain,
        melt=P.MELT,
        opt_alpha=P.OPT_ALPHA,
        opt_steps=P.OPT_STEPS,
        mc_dt=P.MC_DT,
        mc_steps=P.MC_STEPS,
        mc_image_radius=P.MC_IMAGE_RADIUS,
        chunks=P.CHUNKS,
    )
    stage2_hat = jax.block_until_ready(stage2_hat)

    identity_log_weight = chunked_identity_weight(
        stage1_particles, previous, current
    )
    identity_ess = float(compute_ESS_log(identity_log_weight))
    energy_origin = float(current(target.reference_internal()[None, :])[0])
    trainer_seed = int(
        jax.random.key_data(operation_key(base_key, 2, 2, 1))[0]
    )

    with h5py.File(data_path, "w") as handle:
        arrays = {
            "initial_source_particles": initial_particles,
            "stage1_log_weight": stage1_log_weight,
            "stage1_particles": stage1_particles,
            "stage1_mala_acceptance": stage1_mala,
            "stage2_smc_particles": stage2_smc,
            "stage2_smc_ess": stage2_smc_ess,
            "stage2_smc_mala_acceptance": stage2_smc_mala,
            "stage2_hat_particles": stage2_hat,
            "stage2_hat_mala_acceptance": stage2_hat_mala,
        }
        for name, value in arrays.items():
            handle.create_dataset(
                name,
                data=as_float32(value),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
        handle.attrs["schema_version"] = 1
        handle.attrs["previous_t"] = PREVIOUS_T
        handle.attrs["failed_t"] = FAILED_T
        handle.attrs["trainer_seed"] = trainer_seed
        handle.attrs["energy_origin"] = energy_origin
        handle.attrs["stage1_validation_ess"] = stage1_ess
        handle.attrs["stage2_identity_ess"] = identity_ess

    metadata = {
        "schema_version": 1,
        "purpose": "Exact deterministic inputs for the singular n-butane stage-2 KLXX attempt",
        "data": data_path.name,
        "bundle": "../bundle",
        "accepted_flow": archived_flow_path.name,
        "dtype": "float32",
        "seed": P.SEED,
        "previous_t": PREVIOUS_T,
        "failed_t": FAILED_T,
        "trainer_seed": trainer_seed,
        "energy_origin": energy_origin,
        "stage1_validation_ess": stage1_ess,
        "stage2_identity_ess": identity_ess,
        "stage2_min_smc_ess": float(jnp.min(stage2_smc_ess)),
        "stage1_mean_mala_acceptance": float(jnp.mean(stage1_mala)),
        "stage2_mean_smc_mala_acceptance": float(jnp.mean(stage2_smc_mala)),
        "stage2_mean_hat_mala_acceptance": float(jnp.mean(stage2_hat_mala)),
        "n_valid": P.N_VALID,
        "ladder": P.LADDER,
        "mc_dt": P.MC_DT,
        "mc_steps": P.MC_STEPS,
        "mc_image_radius": P.MC_IMAGE_RADIUS,
        "chunks": P.CHUNKS,
        "energy_cut_kj_mol": P.ENERGY_CUT_KJ_MOL,
        "energy_scale_kj_mol": P.ENERGY_SCALE_KJ_MOL,
        "tail_fraction": P.TAIL_FRACTION,
        "jax_version": jax.__version__,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
