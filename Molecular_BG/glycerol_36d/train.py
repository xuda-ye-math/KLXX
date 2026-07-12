#!/usr/bin/env python
"""Train the 36D glycerol mixed-domain Boltzmann generator.

The workflow follows the experiment-driver style under ``Codes/``: literal
configuration is isolated in ``parameters.py``; this file owns deterministic
keys, timestamped logging, training, evaluation, and saved artifacts. The
target is the unmodified GAFF2/AM1-BCC/OBC1 bundle. There is no sharpening,
energy-cap anneal, distance-floor anneal, or delta-QT surrogate.

Run from the repository root:

    conda activate jflows
    PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
        python Molecular_BG/glycerol_36d/train.py

Use ``--smoke`` for the same end-to-end path at tiny sizes. Existing outputs
are not overwritten unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent

import jax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.train import Monitor  # noqa: E402
from jflows.utils import compute_ESS_log  # noqa: E402
from jflows_md import (  # noqa: E402
    Mixed_NSF,
    Molecular_Potential,
    molecular_boltzmann_forward_KLX_G,
    mixed_flow_metadata,
    package_source_sha256,
)
from jflows_md.system import sha256_file  # noqa: E402
from jflows_md.utils import wrapped_normal_relative_error_bound  # noqa: E402

import parameters as P  # noqa: E402


def log_factory(path: Path):
    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


@eqx.filter_jit
def _inverse(flow, samples):
    return flow.inv_and_ladj(samples)


@eqx.filter_jit
def _energy(potential, samples):
    return potential(samples)


@eqx.filter_jit
def _cartesian(potential, samples):
    return potential.cartesian(samples)


def compose_pushforward(stages, samples, chunk: int):
    """Apply G_1^-1, ..., G_K^-1 and accumulate inverse log-Jacobians."""

    value = samples
    total_ladj = jnp.zeros(samples.shape[0], dtype=samples.dtype)
    for stage in stages:
        value_parts, ladj_parts = [], []
        for part in jnp.array_split(value, chunk, axis=0):
            transformed, ladj = _inverse(stage["flow"], part)
            transformed, ladj = jax.block_until_ready((transformed, ladj))
            value_parts.append(transformed)
            ladj_parts.append(ladj)
        value = jnp.concatenate(value_parts, axis=0)
        total_ladj = total_ladj + jnp.concatenate(ladj_parts, axis=0)
    return value, total_ladj


def chunked_energy(potential, samples, chunk: int):
    values = []
    for part in jnp.array_split(samples, chunk, axis=0):
        values.append(jax.block_until_ready(_energy(potential, part)))
    return jnp.concatenate(values)


def chunked_cartesian(potential, samples, chunk: int):
    values = []
    for part in jnp.array_split(samples, chunk, axis=0):
        values.append(jax.block_until_ready(_cartesian(potential, part)))
    return jnp.concatenate(values)


def normalized_weights(log_weight):
    finite = jnp.isfinite(log_weight)
    if not bool(jnp.any(finite)):
        raise RuntimeError("all final importance weights are nonfinite")
    maximum = jnp.max(jnp.where(finite, log_weight, -jnp.inf))
    weight = jnp.where(finite, jnp.exp(log_weight - maximum), 0.0)
    total = weight.sum()
    if not bool(jnp.isfinite(total) & (total > 0)):
        raise RuntimeError("final importance weights cannot be normalized")
    return weight / total


def run_sizes(smoke: bool) -> dict[str, int]:
    if not smoke:
        return {
            "n_valid": P.N_VALID,
            "n_pool": P.N_POOL,
            "n_batch": P.N_BATCH,
            "steps": P.STEPS,
            "ladder": P.LADDER,
            "mc_iters": P.MC_ITERS,
            "chunk": P.CHUNK,
        }
    return {
        "n_valid": P.SMOKE_N_VALID,
        "n_pool": P.SMOKE_N_POOL,
        "n_batch": P.SMOKE_N_BATCH,
        "steps": P.SMOKE_STEPS,
        "ladder": P.SMOKE_LADDER,
        "mc_iters": P.SMOKE_MC_ITERS,
        "chunk": P.SMOKE_CHUNK,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    suffix = "_smoke" if args.smoke else ""
    output_path = HERE / f"run{suffix}"
    staging_path = HERE / f".run{suffix}.inprogress-{os.getpid()}"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"output exists ({output_path.name}); pass --overwrite to replace it"
        )
    if staging_path.exists():
        raise SystemExit(f"staging directory already exists: {staging_path}")
    staging_path.mkdir()
    status_path = staging_path / "train_status.log"
    data_path = staging_path / "data.npz"
    flow_path = staging_path / "flows.eqx"
    log = log_factory(status_path)
    sizes = run_sizes(args.smoke)
    wrapped_error_bound = wrapped_normal_relative_error_bound(
        P.MC_STEP, P.WRAPPED_IMAGES
    )

    target = Molecular_Potential.from_bundle(P.BUNDLE)
    source = target.source()
    jflows_source_hash = package_source_sha256("jflows")
    jflows_md_source_hash = package_source_sha256("jflows_md")
    if target.dimension != P.DIMENSION:
        raise ValueError(f"bundle dimension {target.dimension} != {P.DIMENSION}")
    if (target.domain.euclidean_dim, target.domain.periodic_dim) != (25, 11):
        raise ValueError("glycerol bundle must have mixed domain R^25 x T^11")

    log(
        f"START {P.RUN_NAME}{suffix} | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | bundle={target.bundle_name} "
        f"precision={target.reference_internal().dtype} | "
        f"manifest={target.manifest_sha256} | domain=R^25xT^11 | "
        f"N_VALID={sizes['n_valid']} N_POOL={sizes['n_pool']} "
        f"N_BATCH={sizes['n_batch']} STEPS={sizes['steps']} LR={P.LR} | "
        f"Mixed_NSF bins={P.BINS} transforms={P.TRANSFORMS} "
        f"hidden={P.HIDDEN_FEATURES} | SMC ladder={sizes['ladder']} "
        f"step={P.MC_STEP} iters={sizes['mc_iters']} | "
        f"post-MALA uses the same step/iters | "
        f"wrapped_images={P.WRAPPED_IMAGES} "
        f"relative_tail_bound<={wrapped_error_bound:.3e} | "
        f"e_clip={P.E_CLIP} relative g_clip={P.G_CLIP} bg={P.BG_PARAM}"
        f" checkpoint={P.CHECKPOINT}"
    )
    log(
        "model note: GAFF2/AM1-BCC/OBC1 differs from the old vacuum model; "
        "no sharpening or target deformation is used"
    )

    source_key, flow_key = jax.random.split(jax.random.key(P.SEED))
    x_valid = source.samples(source_key, sizes["n_valid"])
    flow0 = Mixed_NSF(
        flow_key,
        target.domain,
        bins=P.BINS,
        transforms=P.TRANSFORMS,
        euclidean_bound=P.NSF_LIM,
        hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE,
    ).zeros()
    flow_metadata = mixed_flow_metadata(flow0)

    monitor = Monitor(P.MONITOR_EVERY, f"[{P.RUN_NAME}] ", log)
    started = time.time()
    particles, stages = molecular_boltzmann_forward_KLX_G(
        x_valid,
        source,
        target,
        flow0,
        n_pool=sizes["n_pool"],
        n_batch=sizes["n_batch"],
        steps=sizes["steps"],
        lr=P.LR,
        ladder=sizes["ladder"],
        mc_step=P.MC_STEP,
        mc_iters=sizes["mc_iters"],
        coeff_lambda=P.COEFF_LAMBDA,
        monitor=monitor,
        bg_param=P.BG_PARAM,
        chunk=sizes["chunk"],
        images=P.WRAPPED_IMAGES,
        e_clip=P.E_CLIP,
        g_clip=P.G_CLIP,
        seed=P.SEED,
        checkpoint=P.CHECKPOINT,
    )
    particles = jax.block_until_ready(particles)
    jax.effects_barrier()
    wall = time.time() - started

    ladder = np.asarray([stage["t"] for stage in stages])
    stage_ess = np.asarray([stage["ess"] for stage in stages])
    complete = bool(len(ladder) and ladder[-1] == 1.0)
    if not complete:
        reached = float(ladder[-1]) if len(ladder) else 0.0
        log(f"WARNING: incomplete bridge ladder, reached t={reached:.4f}")

    y_push, inverse_ladj = compose_pushforward(stages, x_valid, sizes["chunk"])
    y_push = jax.block_until_ready(y_push)
    target_energy = chunked_energy(target, y_push, sizes["chunk"])
    source_energy = source(x_valid)
    log_weight = source_energy - target_energy + inverse_ladj
    log_weight = jnp.where(jnp.isfinite(log_weight), log_weight, -jnp.inf)
    final_ess = float(compute_ESS_log(log_weight))
    if not np.isfinite(final_ess) or final_ess <= 0:
        raise RuntimeError(f"invalid final ESS: {final_ess}")
    weight = normalized_weights(log_weight)

    cartesian_nm = chunked_cartesian(target, y_push, sizes["chunk"])
    np.savez_compressed(
        data_path,
        schema_version=2,
        run_name=P.RUN_NAME,
        bundle=P.BUNDLE,
        manifest_sha256=target.manifest_sha256,
        jflows_source_sha256=jflows_source_hash,
        jflows_md_source_sha256=jflows_md_source_hash,
        jax_version=jax.__version__,
        equinox_version=eqx.__version__,
        seed=P.SEED,
        flow_key_data=np.asarray(jax.random.key_data(flow_key)),
        activation_id=flow_metadata["activation_id"],
        parameter_dtype=flow_metadata["parameter_dtype"],
        condition_mask=flow_metadata["condition_mask"],
        bins=P.BINS,
        transforms=P.TRANSFORMS,
        hidden_features=np.asarray(P.HIDDEN_FEATURES),
        slope=P.SLOPE,
        nsf_lim=P.NSF_LIM,
        n_valid=sizes["n_valid"],
        n_pool=sizes["n_pool"],
        n_batch=sizes["n_batch"],
        steps=sizes["steps"],
        learning_rate=P.LR,
        smc_ladder=sizes["ladder"],
        mc_step=P.MC_STEP,
        mc_iters=sizes["mc_iters"],
        chunk=sizes["chunk"],
        checkpoint=P.CHECKPOINT,
        e_clip=P.E_CLIP,
        g_clip=P.G_CLIP,
        coeff_lambda=P.COEFF_LAMBDA,
        bg_param_json=json.dumps(P.BG_PARAM, sort_keys=True),
        dimension=P.DIMENSION,
        euclidean_dim=target.domain.euclidean_dim,
        periodic_dim=target.domain.periodic_dim,
        complete=complete,
        stage_count=len(stages),
        wall_s=wall,
        wrapped_normal_relative_error_bound=wrapped_error_bound,
        ladder=ladder,
        stage_ess=stage_ess,
        stage_trained_ess=np.asarray(
            [stage["trained_ess"] for stage in stages]
        ),
        stage_identity_ess=np.asarray(
            [stage["identity_ess"] for stage in stages]
        ),
        stage_selected=np.asarray(
            [stage["selected"] for stage in stages]
        ),
        stage_improvement=np.asarray(
            [stage["imp_history"] for stage in stages]
        ),
        stage_ess_samples=np.asarray(
            [stage["ess_samples"] for stage in stages]
        ),
        stage_ess_history=np.asarray(
            [np.asarray(stage["ess_history"]) for stage in stages]
        ) if stages else np.zeros((0, sizes["steps"])),
        kept_history=np.asarray(
            [np.asarray(stage["kept_history"]) for stage in stages]
        ) if stages else np.zeros((0, sizes["steps"])),
        update_history=np.asarray(
            [np.asarray(stage["update_history"]) for stage in stages]
        ) if stages else np.zeros((0, sizes["steps"]), dtype=bool),
        smc_ess=np.asarray(
            [np.asarray(stage["smc_ess"]) for stage in stages]
        ) if stages else np.zeros((0, sizes["ladder"])),
        smc_acceptance=np.asarray(
            [np.asarray(stage["smc_acceptance"]) for stage in stages]
        ) if stages else np.zeros(
            (0, sizes["ladder"], sizes["mc_iters"])
        ),
        post_mala_acceptance=np.asarray(
            [np.asarray(stage["mala_acceptance"]) for stage in stages]
        ) if stages else np.zeros((0, sizes["mc_iters"])),
        final_ess=final_ess,
        q_push=np.asarray(y_push),
        q_particles=np.asarray(particles),
        cartesian_push_nm=np.asarray(cartesian_nm),
        target_energy=np.asarray(target_energy),
        source_energy=np.asarray(source_energy),
        log_weight=np.asarray(log_weight),
        normalized_weight=np.asarray(weight),
    )
    eqx.tree_serialise_leaves(flow_path, [stage["flow"] for stage in stages])
    log(
        f"DONE complete={complete} K={len(stages)} "
        f"ladder={[f'{value:.4f}' for value in ladder]} "
        f"stage_ESS={[f'{value:.3f}' for value in stage_ess]} "
        f"final_ESS={final_ess:.4f} wall={wall:.0f}s | "
        f"saved {data_path.name}, {flow_path.name}"
    )
    marker = {
        "schema_version": 2,
        "run_name": P.RUN_NAME,
        "bridge_complete": complete,
        "stage_count": len(stages),
        "bundle": target.bundle_name,
        "manifest_sha256": target.manifest_sha256,
        "jflows_source_sha256": jflows_source_hash,
        "jflows_md_source_sha256": jflows_md_source_hash,
        "data_sha256": sha256_file(data_path),
        "flows_sha256": sha256_file(flow_path),
        "status_sha256": sha256_file(status_path),
    }
    (staging_path / "COMPLETE.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if output_path.is_dir():
        shutil.rmtree(output_path)
    elif output_path.exists():
        output_path.unlink()
    os.replace(staging_path, output_path)
    print(f"promoted completed artifacts to {output_path}", flush=True)


if __name__ == "__main__":
    main()
