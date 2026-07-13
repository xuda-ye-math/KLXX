"""Shared full adaptive-KLXX driver for the three small-alkane tests."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jflows.train import Monitor
from jflows_md import (
    Mixed_NSF,
    Molecular_Potential,
    mixed_flow_metadata,
    molecular_boltzmann_forward_KLXX_G,
    package_source_sha256,
)
from jflows_md.system import sha256_file


def _json_write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_state(path: Path) -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status": status}


def _save_particles(path: Path, particles: np.ndarray, attributes: dict) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "particles",
            data=particles,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        for name, value in attributes.items():
            handle.attrs[name] = value


def _plot_ladder(path: Path, stages: list[dict]) -> None:
    level = np.arange(1, len(stages) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(level, [stage["t"] for stage in stages], "o-", lw=2)
    axes[0].set(
        xlabel="ladder level",
        ylabel="bridge coefficient",
        ylim=(0, 1.03),
        title="adaptive ladder",
    )
    axes[1].plot(
        level,
        [stage["ess"] for stage in stages],
        "o-",
        lw=2,
        label="selected",
    )
    axes[1].plot(
        level,
        [stage["trained_ess"] for stage in stages],
        "o--",
        label="trained",
    )
    axes[1].plot(
        level,
        [stage["identity_ess"] for stage in stages],
        "o:",
        label="identity",
    )
    axes[1].axhline(
        stages[0]["tau_ess"], color="0.6", ls="--", lw=1, label="ESS gate"
    )
    axes[1].set(
        xlabel="ladder level",
        ylabel="full-validation ESS",
        ylim=(0, 1.03),
        title="per-level validation ESS",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(parameters) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    here = Path(parameters.__file__).resolve().parent
    bundle_path = here / "bundle"
    final = here / "run"
    if final.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {final}; use --overwrite to replace it")
        shutil.rmtree(final)
    staging = here / f".run.inprogress-{os.getpid()}"
    if staging.exists():
        raise SystemExit(f"staging path exists: {staging}")
    staging.mkdir()
    status_path = staging / "train_status.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with status_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started = time.time()
    physical = Molecular_Potential.from_bundle(
        bundle_path, temperature_kelvin=300.0
    )
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
        f"domain=R{physical.domain.euclidean_dim}xT{physical.domain.periodic_dim}"
    )
    log(
        f"flow balanced bins={parameters.BINS} "
        f"transforms={parameters.TRANSFORMS} hidden={parameters.HIDDEN_FEATURES} | "
        f"N_VALID={parameters.N_VALID} N_POOL={parameters.N_POOL} "
        f"N_BATCH={parameters.N_BATCH} steps={parameters.STEPS} lr={parameters.LR} "
        f"bg={parameters.BG_PARAM}"
    )

    source_key, flow_key = jax.random.split(jax.random.key(parameters.SEED))
    x_valid = source.samples(source_key, parameters.N_VALID)
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
    monitor = Monitor(
        parameters.MONITOR_EVERY, f"[{parameters.MOLECULE}] ", log
    )
    particles, stages = molecular_boltzmann_forward_KLXX_G(
        x_valid,
        source,
        target,
        flow0,
        n_pool=parameters.N_POOL,
        n_batch=parameters.N_BATCH,
        steps=parameters.STEPS,
        lr=parameters.LR,
        ladder=parameters.LADDER,
        mc_step=parameters.MC_STEP,
        mc_iters=parameters.MC_ITERS,
        melt=parameters.MELT,
        opt_step=parameters.OPT_STEP,
        opt_iters=parameters.OPT_ITERS,
        coeff_lambda=parameters.COEFF_LAMBDA,
        coeff_alpha=parameters.COEFF_ALPHA,
        coeff_beta=parameters.COEFF_BETA,
        monitor=monitor,
        bg_param=parameters.BG_PARAM,
        chunk=parameters.CHUNK,
        images=parameters.WRAPPED_IMAGES,
        e_clip=parameters.E_CLIP,
        g_clip=parameters.G_CLIP,
        seed=parameters.SEED,
        checkpoint=parameters.CHECKPOINT,
        lr_warmup=parameters.LR_WARMUP,
    )
    particles = np.asarray(jax.block_until_ready(particles))
    if not stages or float(stages[-1]["t"]) != 1.0:
        reached = float(stages[-1]["t"]) if stages else 0.0
        raise RuntimeError(f"adaptive ladder incomplete at t={reached}")

    for stage in stages:
        stage["tau_ess"] = parameters.BG_PARAM["tau_ess"]
    ess_history = np.concatenate(
        [np.asarray(stage["ess_history"]) for stage in stages]
    )
    kept_history = np.concatenate(
        [np.asarray(stage["kept_history"]) for stage in stages]
    )
    update_history = np.concatenate(
        [np.asarray(stage["update_history"]) for stage in stages]
    )
    level_t = np.asarray([stage["t"] for stage in stages])
    level_ess = np.asarray([stage["ess"] for stage in stages])
    level_trained_ess = np.asarray([stage["trained_ess"] for stage in stages])
    level_identity_ess = np.asarray([stage["identity_ess"] for stage in stages])
    level_selected = np.asarray([stage["selected"] for stage in stages])

    data_path = staging / "data.npz"
    np.savez_compressed(
        data_path,
        schema_version=2,
        bundle="../bundle",
        manifest_sha256=physical.manifest_sha256,
        stage_count=len(stages),
        bins=parameters.BINS,
        transforms=parameters.TRANSFORMS,
        hidden_features=np.asarray(parameters.HIDDEN_FEATURES),
        slope=parameters.SLOPE,
        nsf_lim=parameters.NSF_LIM,
        flow_key_data=np.asarray(jax.random.key_data(flow_key)),
        jflows_source_sha256=package_source_sha256("jflows"),
        jflows_md_source_sha256=package_source_sha256("jflows_md"),
        jax_version=jax.__version__,
        equinox_version=eqx.__version__,
        level_t=level_t,
        level_ess=level_ess,
        level_trained_ess=level_trained_ess,
        level_identity_ess=level_identity_ess,
        level_selected=level_selected,
        ess_history=ess_history,
        kept_history=kept_history,
        update_history=update_history,
        **mixed_flow_metadata(flow0),
    )
    flows_path = staging / "flows.eqx"
    eqx.tree_serialise_leaves(
        flows_path, tuple(stage["flow"] for stage in stages)
    )
    particles_path = staging / "particles.h5"
    _save_particles(
        particles_path,
        particles,
        {
            "molecule": parameters.MOLECULE,
            "temperature_kelvin": 300.0,
            "energy_cut_kj_mol": parameters.ENERGY_CUT_KJ_MOL,
            "manifest_sha256": physical.manifest_sha256,
        },
    )
    figure_path = staging / "adaptive_bg.png"
    _plot_ladder(figure_path, stages)

    ladder_summary = []
    for index, stage in enumerate(stages):
        record = {
            "level": index + 1,
            "t": float(stage["t"]),
            "selected": stage["selected"],
            "validation_ess": float(stage["ess"]),
            "trained_ess": float(stage["trained_ess"]),
            "identity_ess": float(stage["identity_ess"]),
            "ess_samples": int(stage["ess_samples"]),
        }
        ladder_summary.append(record)
        log(
            f"RESULT level={record['level']} t={record['t']:.4f} "
            f"validation_ESS[N={record['ess_samples']}]="
            f"{record['validation_ess']:.6f} selected={record['selected']} "
            f"trained={record['trained_ess']:.6f} "
            f"identity={record['identity_ess']:.6f}"
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
            "n_pool": parameters.N_POOL,
            "n_batch": parameters.N_BATCH,
            "steps_per_level": parameters.STEPS,
            "lr": parameters.LR,
            "minimum_kept_fraction": float(kept_history.min()),
            "update_fraction": float(update_history.mean()),
            "bg_param": parameters.BG_PARAM,
        },
        "runtime": {
            "wall_seconds": time.time() - started,
            "command": sys.argv,
            "python": sys.version,
            "jax": jax.__version__,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "devices": [str(device) for device in jax.devices()],
            "jflows": _git_state(Path("/mnt/projects/jflows")),
            "jflows_md": _git_state(Path("/mnt/projects/jflows_md")),
            "driver_sha256": sha256_file(Path(__file__)),
            "parameters_sha256": sha256_file(Path(parameters.__file__)),
        },
    }
    summary_path = staging / "summary.json"
    _json_write(summary_path, summary)
    log(
        f"DONE adaptive ladder complete ({len(stages)} levels); "
        f"final level validation ESS={level_ess[-1]:.6f}"
    )
    marker = {
        "schema_version": 2,
        "data_sha256": sha256_file(data_path),
        "flows_sha256": sha256_file(flows_path),
        "status_sha256": sha256_file(status_path),
        "particles_sha256": sha256_file(particles_path),
        "summary_sha256": sha256_file(summary_path),
        "figure_sha256": sha256_file(figure_path),
    }
    _json_write(staging / "COMPLETE.json", marker)
    staging.rename(final)
    print(f"PASS {parameters.MOLECULE}: {final}", flush=True)


def main(parameters) -> None:
    run(parameters)
