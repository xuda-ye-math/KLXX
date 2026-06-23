#!/usr/bin/env python
"""Per-molecule Boltzmann-generator training, driven entirely by the local config.json.

A self-contained, single-molecule replacement for the general hetero-BG driver.
Drop this file together with its ``config.json`` into a molecule folder and run it:
every training parameter is read from ``./config.json``, the molecule's ``<prmtop>``
/ ``<rst7>`` are loaded from the shared ``zflows_md/data`` asset folder, and every
output (``data_<TAG>.pth``, ``status_<TAG>.log``) is written next to this script. No
CLI overrides, no ``load_config`` template indirection.

This is the ORIGINAL VACUUM Boltzmann generator (Amber force field in vacuum,
``NoCutoff``). The implicit-solvent variant changes ONLY the OpenMM ``System`` (the
``short_md`` whitening run and ``build``'s force field) -- the ladder/sharpening/loss
pipeline below is unchanged.

Usage::

    python train.py [--method klxx|kl|asmc] [--raw] [--no-delta] [--smoke]
                    [--n_valid N --n_pool N --n_batch N --steps N]

      klxx (default)  the X-regularized, delta-QT, sharpening deliverable (klxx_delta_sharpen)
      kl              forward-KL sharpening baseline (no wide-coverage QT pool)
      asmc            identity-flow annealed-SMC baseline (no flow trained)
      --raw           fixed cap = e_max, no anneal / no sharpening (TAG _raw); else _sharpen
      --no-delta      klxx only: drop the delta-QT pool (klxx_sharpen vs klxx_delta_sharpen)
      --smoke         tiny end-to-end run; saves NO data_<TAG>.pth (logs status_<TAG>.smoke.log)
      --n_*/--steps   override per-run sizes (e.g. tiny all-method smokes)

    TAGs (data_<TAG>.pth): asmc, asmc_raw, kl_sharpen, kl_raw, klxx_sharpen, klxx_raw,
                           klxx_delta_sharpen, klxx_delta_raw  -- the full hetero_bg ablation.
"""
import os, sys, json, time, argparse
from datetime import datetime
import numpy as np
import torch
import parmed as pmd
from openmm import app, unit, LangevinMiddleIntegrator, Platform

HERE = os.path.dirname(os.path.abspath(__file__))

import zflows_md
from zflows_md.flow import NCSF
from zflows_md.utils import (compute_ESS_log, resample, suppress_warnings, lbfgs, langevin,
                             set_cache_size_limit, set_ess_metric)
from zflows_md.boltzmann import build, run_boltzmann, run_asmc
from zflows_md.potential import linear_combination

suppress_warnings()
set_cache_size_limit(128)                                    # raise dynamo cache so the compiled inverse/loss stay compiled
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def short_md(prmtop, crd, n_frames, stride=200):
    """Short VACUUM Langevin MD -> frames for the whitening (bond/angle) statistics."""
    s = pmd.load_file(prmtop, xyz=crd)
    system = s.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)   # vacuum: no solvent, no cutoff
    integ = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
    try:
        sim = app.Simulation(s.topology, system, integ, Platform.getPlatformByName("CUDA"))
    except Exception:
        sim = app.Simulation(s.topology, system, integ)
    sim.context.setPositions(s.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    frames = []
    for _ in range(n_frames):
        sim.step(stride)
        frames.append(sim.context.getState(getPositions=True)
                      .getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return np.array(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["klxx", "kl", "asmc"], default="klxx")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--raw", action="store_true")        # ablation: fixed cap = e_max, no anneal, no sharpening (TAG _raw); else _sharpen
    ap.add_argument("--no-delta", dest="no_delta", action="store_true")  # klxx: drop delta-QT (-> klxx_sharpen instead of klxx_delta_sharpen)
    ap.add_argument("--n_valid", type=int); ap.add_argument("--n_pool", type=int)   # size overrides (e.g. tiny all-method smokes)
    ap.add_argument("--n_batch", type=int); ap.add_argument("--steps", type=int)
    args = ap.parse_args()
    if args.no_delta and args.method != "klxx":
        raise SystemExit("--no-delta applies only to --method klxx (kl/asmc carry no delta-QT pool)")

    # ---- config.json is the SOLE, read-only runtime input (never written here) ----
    P = json.load(open(os.path.join(HERE, "config.json")))
    # legacy-config compatibility: the 60d config uses the new schema (dim, lambda, max_stages, ...);
    # the 36d/48d configs use d/lam/e_cap and omit max_stages/max_retry/release_cache. The reads below
    # fall back exactly as the old bgconfig.load_config(d, name) did (struct defaults max_stages=25, etc.).
    D = int(P.get("dim", P.get("d")))                        # 'dim' (60d) or legacy 'd' (36d/48d)
    DATA = os.path.join(os.path.dirname(zflows_md.__file__), "data")           # shared prmtop/rst7 assets (all molecules)
    prmtop = os.path.join(DATA, P["prmtop"])
    crd = prmtop[:-len(".prmtop")] + ".rst7"

    N_POOL, N_BATCH, N_VALID = P["n_pool"], P["n_batch"], P["n_valid"]
    STEPS, LR, LAMBDA = P["steps"], P["lr"], P.get("lambda", P.get("lam"))   # legacy 'lam'
    BINS, TRANSFORMS, HIDDEN = P["bins"], P["transforms"], tuple(P["hidden"])
    OPT_STEP, OPT_ITERS = P["opt_step"], P["opt_iters"]
    MC_STEP, MC_ITERS = P["mc_step"], P["mc_iters"]
    SMC_RUNGS, SMC_RUNG_ITERS = P["smc_rungs"], P["smc_rung_iters"]
    QT_CHUNK = P["qt_chunk"]
    ADAPTIVE_TAU, VALIDATION_TAU, T_SAFE = P["adaptive_tau"], P["validation_tau"], P["t_safe"]
    SHRINK, ENLARGE = P["shrink_factor"], P["enlarge_factor"]
    MAX_STAGES, MAX_RETRY = P.get("max_stages", 25), P.get("max_retry", 8)   # struct defaults if absent
    ESS_GATE = {int(k): float(v) for k, v in P["ess_gate"].items()}
    RELEASE_CACHE, GRAD_CLIP, LR_WARMUP = P.get("release_cache", True), P["grad_clip"], P["lr_warmup"]
    R_FLOOR = P.get("r_floor", P.get("r_max", 0.2))          # build base floor (soft r_max if the r-anneal is on)
    E_MIN, E_MAX = P["e_min"], (P.get("e_max") or P.get("e_cap"))   # cap anneal; legacy configs carry e_cap
    R_MIN, R_MAX = P.get("r_min"), P.get("r_max")           # nonbonded floor anneal r_max -> r_min (off if unset)
    DROP = P.get("drop", 0.0)
    GATE_SNAPSHOT = bool(P.get("gate_snapshot", False))
    ANNEAL_MODE = P.get("anneal_mode", "geometric")
    DELTA = (0.0 if args.no_delta else float(P.get("delta", 0.0))) if args.method == "klxx" else 0.0   # delta-QT: klxx-only, on by default (--no-delta disables)
    set_ess_metric(P.get("ess_metric", "raw"), c=P.get("ess_metric_c", 2.5))

    HIGHD = D > 60                                            # compile the inverse up to d=60; raw eager beyond
    COMPILE_INV = not HIGHD
    CH = QT_CHUNK * 4 if HIGHD else QT_CHUNK
    # smoke = tiny run for a fast end-to-end sanity check
    NV = args.n_valid or (4000 if args.smoke else N_VALID)
    NP = args.n_pool or (2000 if args.smoke else N_POOL)
    NB = args.n_batch or (500 if args.smoke else N_BATCH)
    ST = args.steps or (60 if args.smoke else STEPS)
    TAG = (("asmc_raw" if args.raw else "asmc") if args.method == "asmc"          # match hetero_bg TAGs
           else args.method + ("_delta" if DELTA > 0 else "") + ("_raw" if args.raw else "_sharpen"))

    LOG = os.path.join(HERE, f"status_{TAG}{'.smoke' if args.smoke else ''}.log"); open(LOG, "w").close()
    def log(m):
        line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")

    log(f"##### {P['molecule'].upper()} VACUUM BG  d={D}  method={TAG}  "
        f"pool={NP} batch={NB} valid={NV} steps={ST} delta={DELTA} drop={DROP} device={DEV} #####")
    e_status = (f"RAW fixed-cap={E_MAX} (no anneal/sharpen)" if args.raw
                else f"e_anneal={E_MIN}->{E_MAX} ({ANNEAL_MODE})  r_anneal={R_MAX}->{R_MIN}")
    log(f"  prmtop={os.path.basename(prmtop)}  {e_status}  grad_clip={GRAD_CLIP}  gate_snapshot={GATE_SNAPSHOT}")

    log("  short vacuum MD for whitening stats ...")
    frames = short_md(prmtop, crd, 200 if args.smoke else 2000)
    B = build(prmtop, crd, md_frames=frames, T=300.0, device=DEV, dtype=torch.float32,
              r_floor=R_FLOOR, e_cap=E_MAX)                  # initial cap = e_max; run_boltzmann anneals e_min->e_max
    u, u0 = B["u"], B["u0"]; a, b, wrap = B["a"], B["b"], B["wrap"]
    assert B["n_internal"] == D, (B["n_internal"], D)
    for pot in (u0, u):
        pot.enable_grad(mode="default"); pot.enable_eval(mode="default")

    if args.raw:                                             # raw ablation: no e/r anneal, no per-stage sharpening
        E_MIN = E_MAX = R_MIN = R_MAX = None                 # (build above already took the fixed cap e_cap = e_max)

    def flow_factory():
        f = NCSF(a=a.tolist(), b=b.tolist(), bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN).to(DEV); f.zeros(); return f

    def qt_fn(target):                                       # quench-and-temper wide-coverage pool on the stage target U_k
        x = u0.samples(NP)
        if DELTA > 0.0:                                      # delta-QT: quench+temper on (1-delta)U_k, reweight by e^{-delta U_k}
            soft = linear_combination([target], [1.0 - DELTA])
            x = lbfgs(x, soft, step=OPT_STEP, iters=OPT_ITERS, armijo=True, chunk=CH)
            x = wrap(langevin(x, soft, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))
            logw = -DELTA * target.eval(x)                  # importance weight w propto e^{-delta U_k}
            logw = logw.masked_fill(~torch.isfinite(logw), float("-inf"))   # softened temper can wander to NaN/Inf energies -> 0 weight
            m = logw.max()
            w_imp = (logw - m).exp() if torch.isfinite(m) else torch.ones_like(logw)
            x = resample(x, w_imp)                           # reweight back to U_k
            x = wrap(langevin(x, target, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))   # rejuvenate on U_k
        else:
            x = lbfgs(x, target, step=OPT_STEP, iters=OPT_ITERS, armijo=True, chunk=CH)
            x = wrap(langevin(x, target, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))
        w = torch.full((x.shape[0],), 1.0 / x.shape[0], device=x.device)   # unweighted pool
        return x, w

    t0 = time.perf_counter()
    if args.method == "asmc":                               # identity-flow annealed-SMC baseline (no flow trained)
        stages, Y, complete = run_asmc(
            u0, u, n_valid=NV, n_pool=NP, mc_step=MC_STEP, mc_iters=MC_ITERS,
            smc_rungs=SMC_RUNGS, smc_rung_iters=SMC_RUNG_ITERS, adaptive_tau=ADAPTIVE_TAU,
            validation_tau=VALIDATION_TAU, shrink_factor=SHRINK, enlarge_factor=ENLARGE, wrap=wrap,
            device=DEV, status=log, max_stages=MAX_STAGES, max_retry=MAX_RETRY, t_safe=T_SAFE,
            e_min=E_MIN, e_max=E_MAX, r_min=R_MIN, r_max=R_MAX, drop=DROP, anneal_mode=ANNEAL_MODE)
        flow = None
    else:
        stages, Y, complete, flow, F_inv = run_boltzmann(
            u0, u, flow_factory, n_valid=NV, n_pool=NP, n_batch=NB, steps=ST, lr=LR, lam=LAMBDA,
            mc_step=MC_STEP, mc_iters=MC_ITERS, smc_rungs=SMC_RUNGS, smc_rung_iters=SMC_RUNG_ITERS,
            adaptive_tau=ADAPTIVE_TAU, validation_tau=VALIDATION_TAU, shrink_factor=SHRINK,
            enlarge_factor=ENLARGE, wrap=wrap, qt_fn=qt_fn, device=DEV, status=log,
            max_stages=MAX_STAGES, max_retry=MAX_RETRY, t_safe=T_SAFE,
            method=args.method, grad_clip=GRAD_CLIP, taming=0.0, lr_warmup=LR_WARMUP,
            e_min=E_MIN, e_max=E_MAX, r_min=R_MIN, r_max=R_MAX, drop=DROP, compile_inv=COMPILE_INV,
            ess_gate=ESS_GATE, release_cache=RELEASE_CACHE, gate_snapshot=GATE_SNAPSHOT,
            anneal_mode=ANNEAL_MODE)

    wall = time.perf_counter() - t0
    val_ess = [round(s["val_ess"], 4) for s in stages]
    sharpen_ess = [None if s.get("sharpen_ess") is None else round(s["sharpen_ess"], 4) for s in stages]
    log(f"##### DONE complete={complete}  K={len(stages)}  val_ess={val_ess}  "
        f"sharpen_ess={sharpen_ess}  wall={wall:.0f}s #####")
    if args.smoke:                                           # smoke verifies execution only -> never write into the molecule folder
        log(f"smoke OK ({TAG}): run completed, data_<TAG>.pth NOT saved")
    else:
        torch.save(dict(name=P["molecule"], d=D, method=TAG, loss=args.method, stages=stages,
                        ladder=[s["t"] for s in stages], complete=complete, val_ess=val_ess,
                        sharpen_ess=sharpen_ess, wall_s=wall, config=P),
                   os.path.join(HERE, f"data_{TAG}.pth"))
        log(f"saved -> data_{TAG}.pth")


if __name__ == "__main__":
    main()
