#!/usr/bin/env python
"""Train the 36D glycerol mixed-domain Boltzmann generator.

The workflow follows the experiment-driver style under ``Codes/``: literal
configuration is isolated in ``parameters.py``; this file owns deterministic
keys, timestamped logging, training, evaluation, and saved artifacts. The
physical Hamiltonian is the GAFF2/AM1-BCC/OBC1 bundle. The active experiment
uses the explicitly configured energy-regularized surrogate and evaluates the
finished generator against both that surrogate and the exact physical target.

Run from the repository root:

    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
        python Molecular_BG/glycerol_36d/train.py

Use ``--smoke`` for the same end-to-end path at tiny sizes. Existing outputs
are not overwritten unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
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
    molecular_boltzmann_forward_KLXX_G,
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


def run_sizes(smoke: bool) -> dict[str, object]:
    if not smoke:
        return {
            "n_valid": P.N_VALID,
            "n_pool": P.N_POOL,
            "n_batch": P.N_BATCH,
            "steps": P.STEPS,
            "selection_steps": P.SELECTION_STEPS,
            "ladder": P.LADDER,
            "mc_iters": P.MC_ITERS,
            "chunk": P.CHUNK,
        }
    return {
        "n_valid": P.SMOKE_N_VALID,
        "n_pool": P.SMOKE_N_POOL,
        "n_batch": P.SMOKE_N_BATCH,
        "steps": P.SMOKE_STEPS,
        "selection_steps": tuple(
            step for step in P.SELECTION_STEPS if step <= P.SMOKE_STEPS
        ),
        "ladder": P.SMOKE_LADDER,
        "mc_iters": P.SMOKE_MC_ITERS,
        "chunk": P.SMOKE_CHUNK,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("kl", "klx", "klxx"), default="kl")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    suffix = "_smoke" if args.smoke else ""
    target_suffix = f"_{P.TARGET_TAG}"
    run_name = f"{P.RUN_NAME}_{args.method}{target_suffix}"
    output_path = HERE / f"run_{args.method}{target_suffix}{suffix}"
    staging_path = (
        HERE / f".run_{args.method}{target_suffix}{suffix}.inprogress-{os.getpid()}"
    )
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
    selection_steps = sizes["selection_steps"]
    checkpoint_count = 1 + (1 if selection_steps else 0) + len(selection_steps) + (
        0 if selection_steps and selection_steps[-1] == sizes["steps"] else 1
    )
    wrapped_error_bound = wrapped_normal_relative_error_bound(
        P.MC_STEP, P.WRAPPED_IMAGES
    )

    physical_target = Molecular_Potential.from_bundle(P.BUNDLE)
    target = physical_target.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.ENERGY_TAIL_FRACTION,
    )
    source = physical_target.source()
    reference_q = physical_target.reference_internal()[None]
    reference_energy_kj_mol = float(physical_target.physical_energy(reference_q)[0])
    target_spec = {
        "schema": 1,
        "kind": "lin_log_excess_energy",
        "bundle_manifest_sha256": physical_target.manifest_sha256,
        "reference_energy_kj_mol": reference_energy_kj_mol,
        "energy_cut_excess_kj_mol": P.ENERGY_CUT_KJ_MOL,
        "energy_scale_kj_mol": P.ENERGY_SCALE_KJ_MOL,
        "tail_fraction": P.ENERGY_TAIL_FRACTION,
        "jacobian": "unchanged",
    }
    target_spec_json = json.dumps(target_spec, sort_keys=True, separators=(",", ":"))
    target_spec_sha256 = hashlib.sha256(target_spec_json.encode()).hexdigest()
    jflows_source_hash = package_source_sha256("jflows")
    jflows_md_source_hash = package_source_sha256("jflows_md")
    if target.dimension != P.DIMENSION:
        raise ValueError(f"bundle dimension {target.dimension} != {P.DIMENSION}")
    if (target.domain.euclidean_dim, target.domain.periodic_dim) != (25, 11):
        raise ValueError("glycerol bundle must have mixed domain R^25 x T^11")

    log(
        f"START {run_name}{suffix} | jax {jax.__version__} | "
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
        f"objective={args.method} e_clip={P.E_CLIP} relative "
        f"g_clip={P.G_CLIP} bg={P.BG_PARAM} checkpoint={P.CHECKPOINT} "
        f"lr_warmup={P.LR_WARMUP} "
        f"selection_steps={sizes['selection_steps']} | "
        f"target=regularized c_excess={P.ENERGY_CUT_KJ_MOL:g} kJ/mol "
        f"scale={P.ENERGY_SCALE_KJ_MOL:g} kJ/mol "
        f"rho={P.ENERGY_TAIL_FRACTION:g} E_ref={reference_energy_kj_mol:.6f} "
        f"target_spec={target_spec_sha256}"
    )
    log(
        "model note: GAFF2/AM1-BCC/OBC1 differs from the old vacuum model; "
        "training and every sampler use the declared regularized target; "
        "the exact physical target is evaluation-only"
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

    monitor = Monitor(P.MONITOR_EVERY, f"[{run_name}] ", log)
    started = time.time()
    common = dict(
        n_pool=sizes["n_pool"],
        n_batch=sizes["n_batch"],
        steps=sizes["steps"],
        lr=P.LR,
        ladder=sizes["ladder"],
        mc_step=P.MC_STEP,
        mc_iters=sizes["mc_iters"],
        monitor=monitor,
        bg_param=P.BG_PARAM,
        chunk=sizes["chunk"],
        images=P.WRAPPED_IMAGES,
        e_clip=P.E_CLIP,
        g_clip=P.G_CLIP,
        seed=P.SEED,
        checkpoint=P.CHECKPOINT,
        lr_warmup=P.LR_WARMUP,
        selection_steps=sizes["selection_steps"],
    )
    if args.method == "klxx":
        particles, stages = molecular_boltzmann_forward_KLXX_G(
            x_valid,
            source,
            target,
            flow0,
            melt=P.MELT,
            opt_step=P.OPT_STEP,
            opt_iters=P.OPT_ITERS,
            coeff_lambda=P.COEFF_LAMBDA,
            coeff_alpha=P.COEFF_ALPHA,
            coeff_beta=P.COEFF_BETA,
            **common,
        )
    else:
        particles, stages = molecular_boltzmann_forward_KLX_G(
            x_valid,
            source,
            target,
            flow0,
            coeff_lambda=0.0 if args.method == "kl" else P.COEFF_LAMBDA,
            **common,
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
    physical_target_energy = chunked_energy(
        physical_target, y_push, sizes["chunk"]
    )
    source_energy = source(x_valid)
    log_weight = source_energy - target_energy + inverse_ladj
    log_weight = jnp.where(jnp.isfinite(log_weight), log_weight, -jnp.inf)
    final_ess = float(compute_ESS_log(log_weight))
    if not np.isfinite(final_ess) or final_ess <= 0:
        raise RuntimeError(f"invalid final ESS: {final_ess}")
    weight = normalized_weights(log_weight)
    physical_log_weight = source_energy - physical_target_energy + inverse_ladj
    physical_log_weight = jnp.where(
        jnp.isfinite(physical_log_weight), physical_log_weight, -jnp.inf
    )
    physical_final_ess = float(compute_ESS_log(physical_log_weight))

    cartesian_nm = chunked_cartesian(target, y_push, sizes["chunk"])
    np.savez_compressed(
        data_path,
        schema_version=2,
        run_name=run_name,
        method=args.method,
        bundle=P.BUNDLE,
        manifest_sha256=target.manifest_sha256,
        target_kind="lin_log_excess_energy",
        target_tag=P.TARGET_TAG,
        target_spec_json=target_spec_json,
        target_spec_sha256=target_spec_sha256,
        reference_energy_kj_mol=reference_energy_kj_mol,
        energy_cut_excess_kj_mol=P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        energy_tail_fraction=P.ENERGY_TAIL_FRACTION,
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
        lr_warmup=P.LR_WARMUP,
        selection_steps=np.asarray(selection_steps),
        smc_ladder=sizes["ladder"],
        mc_step=P.MC_STEP,
        mc_iters=sizes["mc_iters"],
        chunk=sizes["chunk"],
        checkpoint=P.CHECKPOINT,
        e_clip=P.E_CLIP,
        g_clip=P.G_CLIP,
        coeff_lambda=0.0 if args.method == "kl" else P.COEFF_LAMBDA,
        coeff_alpha=P.COEFF_ALPHA if args.method == "klxx" else 0.0,
        coeff_beta=P.COEFF_BETA if args.method == "klxx" else 0.0,
        melt=P.MELT if args.method == "klxx" else 0.0,
        opt_step=P.OPT_STEP if args.method == "klxx" else 0.0,
        opt_iters=P.OPT_ITERS if args.method == "klxx" else 0,
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
        stage_final_ess=np.asarray(
            [stage["final_ess"] for stage in stages]
        ),
        stage_identity_ess=np.asarray(
            [stage["identity_ess"] for stage in stages]
        ),
        stage_selected=np.asarray(
            [stage["selected"] for stage in stages]
        ),
        stage_selected_checkpoint=np.asarray(
            [stage["selected_checkpoint"] for stage in stages]
        ),
        stage_selected_step=np.asarray(
            [stage["selected_step"] for stage in stages]
        ),
        stage_checkpoint_steps=np.asarray(
            [np.asarray(stage["checkpoint_steps"]) for stage in stages]
        ) if stages else np.zeros((0, checkpoint_count), dtype=int),
        stage_checkpoint_ess=np.asarray(
            [np.asarray(stage["checkpoint_ess"]) for stage in stages]
        ) if stages else np.zeros((0, checkpoint_count)),
        stage_checkpoint_labels=np.asarray(
            [np.asarray(stage["checkpoint_labels"]) for stage in stages]
        ) if stages else np.zeros((0, checkpoint_count), dtype="U16"),
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
        hat_mala_acceptance=np.asarray(
            [np.asarray(stage["hat_mala_acceptance"]) for stage in stages]
        ) if stages and args.method == "klxx" else np.zeros((0, sizes["mc_iters"])),
        final_ess=final_ess,
        physical_final_ess=physical_final_ess,
        q_push=np.asarray(y_push),
        q_particles=np.asarray(particles),
        cartesian_push_nm=np.asarray(cartesian_nm),
        target_energy=np.asarray(target_energy),
        physical_target_energy=np.asarray(physical_target_energy),
        source_energy=np.asarray(source_energy),
        log_weight=np.asarray(log_weight),
        physical_log_weight=np.asarray(physical_log_weight),
        normalized_weight=np.asarray(weight),
    )
    eqx.tree_serialise_leaves(flow_path, [stage["flow"] for stage in stages])
    log(
        f"DONE complete={complete} K={len(stages)} "
        f"ladder={[f'{value:.4f}' for value in ladder]} "
        f"stage_ESS={[f'{value:.3f}' for value in stage_ess]} "
        f"surrogate_final_ESS={final_ess:.4f} "
        f"physical_final_ESS={physical_final_ess:.4f} wall={wall:.0f}s | "
        f"saved {data_path.name}, {flow_path.name}"
    )
    marker = {
        "schema_version": 2,
        "run_name": run_name,
        "bridge_complete": complete,
        "stage_count": len(stages),
        "bundle": target.bundle_name,
        "manifest_sha256": target.manifest_sha256,
        "target_spec_sha256": target_spec_sha256,
        "surrogate_final_ess": final_ess,
        "physical_final_ess": physical_final_ess,
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
