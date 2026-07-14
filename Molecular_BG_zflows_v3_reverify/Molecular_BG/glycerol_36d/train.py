#!/usr/bin/env python
"""Run the standalone old-zflows molecular Boltzmann generator.

Every experiment choice is read from one JSON file.  The training logic keeps
the historical zflows_md names and algorithms; ``--config`` only selects which
complete parameter file to use.

Run from this folder or the repository root::

    source /home/xuda/.envs/zflows/bin/activate
    python train.py
    python train.py --config /absolute/path/to/config.json
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import time

import numpy as np
import parmed as pmd
import torch
from openmm import LangevinMiddleIntegrator, Platform, app, unit


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import zflows_md  # noqa: E402
from zflows_md.boltzmann import build, run_asmc, run_boltzmann  # noqa: E402
from zflows_md.flow import NCSF  # noqa: E402
from zflows_md.potential import linear_combination  # noqa: E402
from zflows_md.utils import (  # noqa: E402
    langevin,
    lbfgs,
    resample,
    set_cache_size_limit,
    set_ess_metric,
    suppress_warnings,
)


def load_config() -> tuple[dict, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = parser.parse_args()
    path = os.path.abspath(args.config)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle), path


def torch_dtype(name: str):
    try:
        return {"float32": torch.float32}[name]
    except KeyError as exc:
        raise ValueError("dtype must be 'float32' for this experiment") from exc


def short_md(prmtop: str, crd: str, p: dict) -> np.ndarray:
    """Generate whitening frames under the Hamiltonian declared in the config."""
    nonbonded = {"NoCutoff": app.NoCutoff}[p["nonbonded_method"]]
    implicit = {None: None, "OBC1": app.OBC1}[p["implicit_solvent"]]
    constraints = {None: None}[p["constraints"]]
    structure = pmd.load_file(prmtop, xyz=crd)
    system = structure.createSystem(
        nonbondedMethod=nonbonded,
        constraints=constraints,
        implicitSolvent=implicit,
    )
    temperature = float(p["temperature_K"])
    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        float(p["md_friction_per_ps"]) / unit.picosecond,
        float(p["md_timestep_fs"]) * unit.femtosecond,
    )
    try:
        platform = Platform.getPlatformByName(p["md_platform"])
        simulation = app.Simulation(structure.topology, system, integrator, platform)
    except Exception:
        if not p["md_platform_fallback"]:
            raise
        simulation = app.Simulation(structure.topology, system, integrator)
    simulation.context.setPositions(structure.positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    frames = []
    for _ in range(int(p["md_frames"])):
        simulation.step(int(p["md_stride"]))
        frames.append(
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )
    return np.asarray(frames)


def main() -> None:
    p, config_path = load_config()
    method = p["method"]
    if method not in {"asmc", "kl", "klxx"}:
        raise ValueError("method must be one of: asmc, kl, klxx")
    regularization = p.get("regularization", "er")
    if regularization not in {"er", "c"}:
        raise ValueError("regularization must be 'er' or 'c'")
    c_value = float(p.get("c", 50.0))
    c_scale = float(p.get("c_scale", 50.0))
    c_tail_fraction = float(p.get("c_tail_fraction", 0.0))
    raw = bool(p["raw"])
    if regularization == "c" and raw:
        raise ValueError("raw is an e/r option and must be false for c regularization")
    delta = float(p["delta"])
    if method != "klxx" and delta != 0.0:
        raise ValueError("delta must be zero unless method='klxx'")
    if p["seed"] is not None:
        seed = int(p["seed"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = p["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA, but PyTorch cannot access it")

    suppress_warnings()
    set_cache_size_limit(int(p["cache_size_limit"]))
    set_ess_metric(p["ess_metric"], c=float(p["ess_metric_c"]))

    data_dir = os.path.join(os.path.dirname(zflows_md.__file__), "data")
    prmtop = os.path.join(data_dir, p["prmtop"])
    crd = os.path.splitext(prmtop)[0] + ".rst7"
    dimension = int(p["d"])
    n_pool = int(p["n_pool"])
    n_batch = int(p["n_batch"])
    n_valid = int(p["n_valid"])
    steps = int(p["steps"])
    hidden = tuple(int(v) for v in p["hidden"])
    ess_gate = {int(k): float(v) for k, v in p["ess_gate"].items()}
    chunk = int(p["qt_chunk"]) * int(p["qt_chunk_multiplier"])

    method_tag = (
        ("asmc_raw" if raw else "asmc")
        if method == "asmc"
        else method + ("_delta" if delta > 0 else "")
    )
    if regularization == "c":
        c_label = f"{c_value:g}".replace(".", "p")
        tag = f"{method_tag}_c{c_label}"
    elif method == "asmc":
        tag = method_tag
    else:
        tag = method_tag + ("_raw" if raw else "_sharpen")
    output = os.path.join(HERE, f"data_{tag}.pth")
    resume_keep = int(p["resume_keep"])
    if resume_keep < 0:
        raise ValueError("resume_keep must be nonnegative")
    resume_stages = None
    if resume_keep:
        if not os.path.isfile(output):
            raise FileNotFoundError(f"resume checkpoint not found: {output}")
        previous = torch.load(output, map_location="cpu", weights_only=False)
        if previous.get("method") != tag:
            raise ValueError(
                f"resume method {previous.get('method')!r} does not match {tag!r}"
            )
        if int(previous.get("d", -1)) != dimension:
            raise ValueError(
                f"resume dimension {previous.get('d')!r} does not match {dimension}"
            )
        saved_stages = previous.get("stages", [])
        if resume_keep > len(saved_stages):
            raise ValueError(
                f"resume_keep={resume_keep} exceeds {len(saved_stages)} saved stages"
            )
        mutable_resume_keys = {"max_stages", "resume_keep"}
        saved_config = previous.get("config", {})
        changed = [
            key
            for key, value in saved_config.items()
            if key not in mutable_resume_keys and p.get(key) != value
        ]
        if changed:
            raise ValueError(
                "resume config changes scientific settings: " + ", ".join(changed)
            )
        resume_stages = saved_stages[:resume_keep]
    log_path = os.path.join(HERE, f"status_{tag}{p['log_suffix']}.log")
    if not resume_keep:
        open(log_path, "w", encoding="utf-8").close()

    def log(message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(
        f"##### {p['molecule'].upper()} {p['environment'].upper()} BG  d={dimension}  "
        f"method={tag}  pool={n_pool} batch={n_batch} valid={n_valid} "
        f"steps={steps} delta={delta} drop={p['drop']} device={device} #####"
    )
    if regularization == "c":
        energy_status = (
            f"reference-shifted c={c_value} scale={c_scale} "
            f"tail={c_tail_fraction} (no e/r sharpening)"
        )
    else:
        energy_status = (
            f"RAW fixed-cap={p['e_max']} (no anneal/sharpen)"
            if raw
            else f"e_anneal={p['e_min']}->{p['e_max']} ({p['anneal_mode']})  "
            f"r_anneal={p['r_max']}->{p['r_min']}"
        )
    log(
        f"  config={config_path} prmtop={os.path.basename(prmtop)}  "
        f"{energy_status}  grad_clip={p['grad_clip']} "
        f"mc_step={p['mc_step']} taming={p['taming']} "
        f"gate_snapshot={p['gate_snapshot']} compile_inv={p['compile_inv']}"
    )
    if resume_stages:
        log(
            f"[resume] keeping first {len(resume_stages)} stages of "
            f"{os.path.basename(output)} "
            f"(t={[round(stage['t'], 3) for stage in resume_stages]}); "
            f"retraining from stage {len(resume_stages) + 1}"
        )

    log(f"  short {p['environment']} MD for whitening stats ...")
    frames = short_md(prmtop, crd, p)
    if p["implicit_solvent"] is not None:
        raise NotImplementedError(
            "the independent OBC1 target is added only after the vacuum control passes"
        )
    problem = build(
        prmtop,
        crd,
        md_frames=frames,
        T=float(p["temperature_K"]),
        device=device,
        dtype=torch_dtype(p["dtype"]),
        r_floor=float(p["r_floor"]),
        e_cap=float(p["e_cap"]),
        environment=p["environment"],
        regularization=regularization,
        c=c_value,
        c_scale=c_scale,
        c_tail_fraction=c_tail_fraction,
    )
    target, source = problem["u"], problem["u0"]
    lower, upper, wrap = problem["a"], problem["b"], problem["wrap"]
    assert problem["n_internal"] == dimension, (problem["n_internal"], dimension)
    for potential in (source, target):
        potential.enable_grad(mode=p["compile_mode"])
        potential.enable_eval(mode=p["compile_mode"])

    if regularization == "c":
        e_min = e_max = r_min = r_max = None
    else:
        e_min, e_max = float(p["e_min"]), float(p["e_max"])
        r_min, r_max = p["r_min"], p["r_max"]
        if raw:
            e_min = e_max = r_min = r_max = None

    def flow_factory():
        flow = NCSF(
            a=lower.tolist(),
            b=upper.tolist(),
            bins=int(p["bins"]),
            transforms=int(p["transforms"]),
            hidden_features=hidden,
        ).to(device)
        flow.zeros()
        return flow

    def qt_fn(stage_target):
        x = source.samples(n_pool)
        if delta > 0.0:
            soft = linear_combination([stage_target], [1.0 - delta])
            x = lbfgs(
                x,
                soft,
                step=float(p["opt_step"]),
                iters=int(p["opt_iters"]),
                armijo=bool(p["armijo"]),
                chunk=chunk,
            )
            x = wrap(
                langevin(
                    x,
                    soft,
                    step=float(p["mc_step"]),
                    iters=int(p["mc_iters"]),
                    taming=float(p["taming"]),
                    chunk=chunk,
                )
            )
            logw = -delta * stage_target.eval(x)
            logw = logw.masked_fill(~torch.isfinite(logw), float("-inf"))
            maximum = logw.max()
            weights = (
                (logw - maximum).exp()
                if torch.isfinite(maximum)
                else torch.ones_like(logw)
            )
            x = resample(x, weights)
            x = wrap(
                langevin(
                    x,
                    stage_target,
                    step=float(p["mc_step"]),
                    iters=int(p["mc_iters"]),
                    taming=float(p["taming"]),
                    chunk=chunk,
                )
            )
        else:
            x = lbfgs(
                x,
                stage_target,
                step=float(p["opt_step"]),
                iters=int(p["opt_iters"]),
                armijo=bool(p["armijo"]),
                chunk=chunk,
            )
            x = wrap(
                langevin(
                    x,
                    stage_target,
                    step=float(p["mc_step"]),
                    iters=int(p["mc_iters"]),
                    taming=float(p["taming"]),
                    chunk=chunk,
                )
            )
        weights = torch.full((x.shape[0],), 1.0 / x.shape[0], device=x.device)
        return x, weights

    started = time.perf_counter()

    def checkpoint(stage_records, completed):
        if not bool(p["save_data"]):
            return
        elapsed = time.perf_counter() - started
        checkpoint_validation = [
            round(stage["val_ess"], 4) for stage in stage_records
        ]
        checkpoint_sharpening = [
            None
            if stage.get("sharpen_ess") is None
            else round(stage["sharpen_ess"], 4)
            for stage in stage_records
        ]
        torch.save(
            dict(
                name=p["molecule"],
                d=dimension,
                method=tag,
                loss=method,
                stages=stage_records,
                ladder=[stage["t"] for stage in stage_records],
                complete=completed,
                val_ess=checkpoint_validation,
                sharpen_ess=checkpoint_sharpening,
                wall_s=elapsed,
                config=p,
            ),
            output,
        )
        log(
            f"[checkpoint] {len(stage_records)} stage(s) -> "
            f"{os.path.basename(output)} (complete={completed}); "
            f"resume_keep={len(stage_records)}"
        )

    common = dict(
        n_valid=n_valid,
        n_pool=n_pool,
        mc_step=float(p["mc_step"]),
        mc_iters=int(p["mc_iters"]),
        smc_rungs=int(p["smc_rungs"]),
        smc_rung_iters=int(p["smc_rung_iters"]),
        adaptive_tau=float(p["adaptive_tau"]),
        validation_tau=float(p["validation_tau"]),
        shrink_factor=float(p["shrink_factor"]),
        enlarge_factor=float(p["enlarge_factor"]),
        wrap=wrap,
        device=device,
        status=log,
        max_stages=int(p["max_stages"]),
        max_retry=int(p["max_retry"]),
        t_tol=float(p["t_tol"]),
        t_safe=float(p["t_safe"]),
        e_min=e_min,
        e_max=e_max,
        r_min=r_min,
        r_max=r_max,
        drop=float(p["drop"]),
        anneal_mode=p["anneal_mode"],
        resume_stages=resume_stages,
        checkpoint_fn=checkpoint if bool(p["save_data"]) else None,
    )
    if method == "asmc":
        stages, particles, complete = run_asmc(source, target, **common)
    else:
        stages, particles, complete, flow, inverse = run_boltzmann(
            source,
            target,
            flow_factory,
            n_batch=n_batch,
            steps=steps,
            lr=float(p["lr"]),
            lam=float(p["lam"]),
            qt_fn=qt_fn,
            method=method,
            grad_clip=float(p["grad_clip"]),
            taming=float(p["taming"]),
            lr_warmup=int(p["lr_warmup"]),
            compile_inv=bool(p["compile_inv"]),
            ess_gate=ess_gate,
            release_cache=bool(p["release_cache"]),
            gate_snapshot=bool(p["gate_snapshot"]),
            **common,
        )

    wall = time.perf_counter() - started
    validation_ess = [round(stage["val_ess"], 4) for stage in stages]
    sharpen_ess = [
        None if stage.get("sharpen_ess") is None else round(stage["sharpen_ess"], 4)
        for stage in stages
    ]
    log(
        f"##### DONE complete={complete} K={len(stages)} "
        f"val_ess={validation_ess} sharpen_ess={sharpen_ess} wall={wall:.0f}s #####"
    )
    if bool(p["save_data"]):
        torch.save(
            dict(
                name=p["molecule"],
                d=dimension,
                method=tag,
                loss=method,
                stages=stages,
                ladder=[stage["t"] for stage in stages],
                complete=complete,
                val_ess=validation_ess,
                sharpen_ess=sharpen_ess,
                wall_s=wall,
                config=p,
            ),
            output,
        )
        log(f"saved -> {os.path.basename(output)}")
    else:
        log(f"execution-only run ({tag}); data_<TAG>.pth not saved")


if __name__ == "__main__":
    main()
