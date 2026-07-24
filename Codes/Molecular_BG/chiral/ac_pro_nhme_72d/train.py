#!/usr/bin/env python
"""Train the KLXX Ac-Pro-NHMe Boltzmann generator."""

import argparse
import json
import os
import time
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from jflows.train import Monitor
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann import iterate_boltzmann
from jflows_md.boltzmann.load import run

import parameters as P


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("klxx",), default="klxx")
    args = parser.parse_args()

    artifacts = HERE / "artifacts"
    run_dir = artifacts / args.method
    log_path = artifacts / f"{args.method}.log"
    if run_dir.exists() or log_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing KLXX artifacts: {run_dir} or {log_path}"
        )
    artifacts.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=False)

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    settings = (
        f"VALID_SIZE={P.VALID_SIZE} LADDER={P.LADDER} "
        f"MC_DT={P.MC_DT} MC_STEPS={P.MC_STEPS} CHUNKS={P.CHUNKS} "
        f"POOL_SIZE={P.POOL_SIZE} BATCH_SIZE={P.BATCH_SIZE} "
        f"TRAIN_STEPS={P.TRAIN_STEPS} U_CLIP={P.U_CLIP} "
        f"G_CLIP={P.G_CLIP} LR_WARMUP={P.LR_WARMUP}"
    )
    log(
        f"START N-acetyl-L-proline N-methylamide 72D KLXX | "
        f"rg_param={P.RG_PARAM_0}->{P.RG_PARAM_1} | {settings}"
    )

    target = Molecular_Potential.from_bundle(
        BUNDLE, temperature_kelvin=P.TEMPERATURE_KELVIN
    )
    source = target.source()
    source_key, flow_key = jax.random.split(jax.random.key(P.SEED))
    x_valid = source.samples(source_key, N=P.VALID_SIZE)
    flow = Mixed_NSF(
        flow_key,
        target.domain,
        bins=P.BINS,
        transforms=P.TRANSFORMS,
        euclidean_bound=P.NSF_LIM,
        hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE,
        mask_strategy="balanced",
    ).zeros()

    controls = {
        "objective": "forward_klxx",
        "pool_size": P.POOL_SIZE,
        "batch_size": P.BATCH_SIZE,
        "train_steps": P.TRAIN_STEPS,
        "lr": P.LR,
        "ladder": P.LADDER,
        "mc_dt": P.MC_DT,
        "mc_steps": P.MC_STEPS,
        "rg_param_0": P.RG_PARAM_0,
        "rg_param_1": P.RG_PARAM_1,
        "initialize_from_identity": P.INITIALIZE_FROM_IDENTITY,
        "coeff_lambda": 1.0,
        "coeff_alpha": 0.5,
        "coeff_beta": 0.5,
        "melt": P.MELT,
        "opt_alpha": P.OPT_ALPHA,
        "opt_steps": P.OPT_STEPS,
        "monitor": Monitor(P.MONITOR_EVERY, f"[{P.MOLECULE} klxx] ", log),
        "bg_param": P.BG_PARAM,
        "chunks": P.CHUNKS,
        "mc_image_radius": P.MC_IMAGE_RADIUS,
        "checkpoint": P.CHECKPOINT,
        "u_clip": P.U_CLIP,
        "g_clip": P.G_CLIP,
        "lr_warmup": P.LR_WARMUP,
        "seed": P.SEED,
    }
    config = {
        "method": args.method,
        "valid_size": P.VALID_SIZE,
        **{key: value for key, value in controls.items() if key != "monitor"},
    }

    def iterate(samples, template, accepted, stage):
        return iterate_boltzmann(
            samples,
            source,
            target,
            template,
            accepted_t=accepted,
            start_stage=stage,
            **controls,
        )

    particles, stages = run(
        run_dir,
        "ac-pro-nhme-72d-klxx-e50-r025-to-e150-r010",
        config,
        x_valid,
        flow,
        iterate,
        resume=False,
    )
    jax.block_until_ready(particles)

    factor = 1.0
    for stage in stages:
        factor /= stage["valid_selected_ess"]
        if stage["rg_start"] != stage["rg_end"]:
            factor /= stage["sharpen_ess"]
    elapsed = sum(stage["elapsed_seconds"] for stage in stages)

    results = HERE / "results"
    results.mkdir(exist_ok=True)
    lines = [
        "# N-Acetyl-L-Proline N-Methylamide 72D KLXX",
        "",
        f"- Sharpening: `{P.RG_PARAM_0}` to `{P.RG_PARAM_1}`",
        f"- Complete: `{bool(stages and stages[-1]['t'] == 1.0)}`",
        f"- Total factor: `{factor:.6g}`",
        f"- Total time: `{elapsed / 60:.2f} min`",
        "",
        "| Stage | t | Selected | Validation ESS | Sharpening ESS |",
        "|---:|---:|:---:|---:|---:|",
    ]
    for index, stage in enumerate(stages, start=1):
        lines.append(
            f"| {index} | {stage['t']:.6f} | {stage['selected']} | "
            f"{stage['valid_selected_ess']:.6f} | {stage['sharpen_ess']:.6f} |"
        )
    (results / "klxx.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(
        json.dumps(
            {
                "method": "klxx",
                "rg_param_0": P.RG_PARAM_0,
                "rg_param_1": P.RG_PARAM_1,
                "stages": len(stages),
                "complete": bool(stages and stages[-1]["t"] == 1.0),
            }
        )
    )


if __name__ == "__main__":
    main()
