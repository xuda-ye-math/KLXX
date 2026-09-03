#!/usr/bin/env python
"""Train the ethane 18D Boltzmann generator: KL+X_pi first, then the other methods on its schedule.

Run from this folder with the jflows environment. The convention is that
KL+X_pi selects the adaptive stage schedule and every other method follows it
as a fixed schedule:

    python train.py --method klx                                   # adaptive, writes artifacts/klx
    python train.py --method klxx --schedule-from artifacts/klx    # KLXX on the klx schedule
    python train.py --method kll1 --schedule-from artifacts/klx    # KL + L1 dispersion on it

Options:
    --method {klx,kll1,klxx}     the loss (default klxx); all parameters come from parameters.py
    --schedule-from RUN          follow the accepted stage schedule of a complete stored run
    --resume-from RUN            fork a stored run and continue from its last accepted stage
    --tag SUFFIX                 name the run directory, log, and results file <method><SUFFIX>

Outputs: artifacts/<name>/ (stages, flows, populations), artifacts/<name>.log,
results/<name>.md (per-stage ESS and the factor F_hat = prod ESS^(-1/2)).

Manual rejection (adaptive runs only): press Ctrl+C in the run's terminal, or
send `kill -INT <pid>` to a detached run (the log prints the command at start),
to reject the running attempt within one gradient step; the controller
shrinks the endpoint and retries. A second Ctrl+C within three seconds stops
the program.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from jflows.train import Monitor
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann import Manual_Reject, iterate_boltzmann
from jflows_md.boltzmann.load import fork, manifest, run
from jflows_md.boltzmann.write import _value, jsonfile

import parameters as P


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("klx", "kll1", "klxx"), default="klxx")
    parser.add_argument("--tag", default="", help="suffix of the run directory, log, and results file")
    parser.add_argument("--resume-from", default=None, help="stored run to fork and continue from its last accepted stage")
    parser.add_argument("--schedule-from", default=None, help="complete stored run whose accepted stage schedule is followed as a fixed schedule")
    args = parser.parse_args()
    name = args.method + args.tag

    artifacts = HERE / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts / name
    log_path = artifacts / f"{name}.log"
    open(log_path, "w").close()

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a") as stream:
            stream.write(line + "\n")

    log(
        f"START ethane 18D {args.method} | "
        f"BG_PARAM={None if args.schedule_from else P.BG_PARAM} "
        f"SCHEDULE_FROM={args.schedule_from} SCREEN_FRACTION={P.SCREEN_FRACTION} | "
        f"VALID_SIZE={P.VALID_SIZE} LADDER={P.LADDER} MC_DT={P.MC_DT} "
        f"MC_STEPS_1={P.MC_STEPS_1} MC_STEPS_2={P.MC_STEPS_2} CHUNKS={P.CHUNKS} "
        f"POOL_SIZE={P.POOL_SIZE} BATCH_SIZE={P.BATCH_SIZE} "
        f"TRAIN_STEPS={P.TRAIN_STEPS} LR={P.LR} LR_WARMUP={P.LR_WARMUP} "
        f"U_CLIP={P.U_CLIP} G_CLIP={P.G_CLIP}"
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
        "objective": {"klx": "forward_klx", "kll1": "forward_kll1", "klxx": "forward_klxx"}[args.method],
        "pool_size": P.POOL_SIZE,
        "batch_size": P.BATCH_SIZE,
        "steps_total": P.TRAIN_STEPS,
        "lr": P.LR,
        "ladder": P.LADDER,
        "mc_dt": P.MC_DT,
        "mc_steps_1": P.MC_STEPS_1,
        "mc_steps_2": P.MC_STEPS_2,
        "initialize_from_identity": P.INITIALIZE_FROM_IDENTITY,
        "coeff_lambda": 1.0,
        "coeff_theta": 1.0,
        "coeff_alpha": 0.5,
        "coeff_qt": P.COEFF_QT if args.method == "klxx" else 0.0,
        "melt": P.MELT if args.method == "klxx" else 0.0,
        "opt_alpha": P.OPT_ALPHA if args.method == "klxx" else 1.0,
        "opt_steps": P.OPT_STEPS if args.method == "klxx" else 0,
        "monitor": Monitor(
            P.MONITOR_EVERY, f"[{P.MOLECULE} {args.method}] ", log
        ),
        "bg_param": None if args.schedule_from else P.BG_PARAM,
        "chunks": P.CHUNKS,
        "mc_image_radius": P.MC_IMAGE_RADIUS,
        "checkpoint": P.CHECKPOINT,
        "u_clip": P.U_CLIP,
        "g_clip": P.G_CLIP,
        "lr_warmup": P.LR_WARMUP,
        "screen_fraction": P.SCREEN_FRACTION,
        "rg_param_0": P.RG_PARAM_0,
        "rg_param_1": P.RG_PARAM_1,
        "seed": P.SEED,
    }
    if args.schedule_from is not None:
        saved = manifest(HERE / args.schedule_from)
        if saved["status"] != "complete":
            raise RuntimeError(f"{args.schedule_from} is not complete: {saved['status']}")
        controls["t_list"] = tuple(float(item["t"]) for item in saved["stages"])
    reject = Manual_Reject()
    if args.schedule_from is None:
        controls["reject_requested"] = reject
        log(f"manual rejection of the running attempt (received within one gradient step): {reject.command}")
    config = {
        "method": args.method,
        "valid_size": P.VALID_SIZE,
        **{key: value for key, value in controls.items() if key not in ("monitor", "reject_requested")},
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

    problem_id = f"ethane-18d-{args.method}"
    if args.resume_from is not None:
        # Fork the stored run and continue it under the current controls.
        record = fork(HERE / args.resume_from, run_dir, problem_id)
        record["config"] = _value(config)
        jsonfile(run_dir / "run.json", record)
        log(f"RESUME from {args.resume_from} at t={record['stages'][-1]['t']} with the current controls")
    with reject:
        particles, stages = run(
            run_dir,
            problem_id,
            config,
            x_valid,
            flow,
            iterate,
            resume=args.resume_from is not None,
        )
    jax.block_until_ready(particles)

    factor = 1.0
    for stage in stages:
        factor /= math.sqrt(stage["valid_selected_ess"])
    elapsed = sum(stage["elapsed_seconds"] for stage in stages)
    effective = sum(stage["accepted_attempt_seconds"] for stage in stages)

    results = HERE / "results"
    results.mkdir(exist_ok=True)
    lines = [
        f"# Ethane 18D {args.method.upper()}",
        "",
        f"- Stage policy: `{controls.get('t_list') or P.BG_PARAM}`",
        f"- Screen fraction: `{P.SCREEN_FRACTION}`",
        f"- Complete: `{bool(stages and stages[-1]['t'] == 1.0)}`",
        f"- Regularization path: `{P.RG_PARAM_0}` to `{P.RG_PARAM_1}`",
        f"- Factor F_hat = prod ESS^(-1/2): `{factor:.6g}`",
        f"- Total time: `{elapsed / 60:.2f} min`",
        f"- Effective training time (accepted attempts): `{effective / 60:.2f} min`",
        "",
        "| Stage | t | rho | Selected | Validation ESS | Trained ESS | Identity ESS | Time (min) |",
        "|---:|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    for index, stage in enumerate(stages, start=1):
        lines.append(
            f"| {index} | {stage['t']:.6f} | {stage['rg_end']} | {stage['selected']} | "
            f"{stage['valid_selected_ess']:.6f} | {stage['valid_trained_ess']:.6f} | "
            f"{stage['valid_identity_ess']:.6f} | {stage['elapsed_seconds'] / 60:.2f} |"
        )
    (results / f"{name}.md").write_text("\n".join(lines) + "\n")
    log(json.dumps({
        "method": args.method,
        "stages": len(stages),
        "complete": bool(stages and stages[-1]["t"] == 1.0),
        "factor": factor,
    }))


if __name__ == "__main__":
    main()
