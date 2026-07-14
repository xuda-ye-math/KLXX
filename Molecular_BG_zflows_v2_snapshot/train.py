#!/usr/bin/env python
"""Per-molecule Boltzmann-generator training on the IMPLICIT-SOLVENT alanine-dipeptide
target, driven by the local config.json.

Faithful port of the proven vacuum driver (Molecular_BG/adp_60d/train.py): the ladder /
sharpening / loss pipeline is UNCHANGED. The only differences are the physics setup, per
the rebuild plan (Phase 0/1):
  * force field: amber96 + amber96_obc (GBSAOBCForce, OBC2) built from the ADP PDB
    topology -- NOT the openff small-molecule prmtop -- so the torch Amber_Force_Field
    (with its validated OBC2 GB term) targets the same energy as the FAB Fig.19 MD
    reference (see validate_obc.py: torch vs OpenMM total < 1e-2 kJ/mol).
  * chirality: L-alanine start (position_min_energy.pt), so whitening stats are L.
  * whitening short MD is implicit-solvent (consistent with the target).

`asmc` is the classical annealed-SMC reference (identity flow, no flow trained).

Usage::

    python train.py [--method asmc|klxx|kl] [--raw] [--no-delta] [--smoke]
                    [--n_valid N --n_pool N --n_batch N --steps N]
"""
import os, sys, json, time, argparse
from datetime import datetime
import numpy as np
import torch
import openmm as mm
import openmm.app as app
from openmm import unit, LangevinMiddleIntegrator, Platform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import zflows_md
from zflows_md.flow import NCSF
from zflows_md.utils import (compute_ESS_log, resample, suppress_warnings, lbfgs, langevin,
                             set_cache_size_limit, set_ess_metric)
from zflows_md.boltzmann import build, run_boltzmann, run_asmc
from zflows_md.potential import linear_combination

suppress_warnings()
set_cache_size_limit(128)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# External fixed assets (same as verify_md.py / validate_obc.py): amber96 ADP topology +
# L-alanine min-energy structure (nm, 22-atom ordering matching the PDB). Resolved against a
# candidate list so a move of fab-torch / zflows_md_backup under /mnt/projects doesn't break it.
def _first_existing(cands, what):
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{what} not found in any of: {cands}")

PDB = _first_existing([
    "/mnt/projects/zflows_md_backup/tests/data/alanine_dipeptide.pdb",
    "/home/xuda/zflows_md_backup/tests/data/alanine_dipeptide.pdb",
], "ADP PDB")
LSTART = _first_existing([
    "/mnt/projects/fab-torch/experiments/aldp/data/position_min_energy.pt",
    "/home/xuda/fab-torch/experiments/aldp/data/position_min_energy.pt",
], "L-alanine position_min_energy.pt")


def _l_positions():
    return torch.load(LSTART, weights_only=False).numpy().reshape(-1, 3).astype(np.float64)


def build_implicit():
    """amber96 + amber96_obc on the ADP PDB topology -> (system, bonds, posL[nm])."""
    pdb = app.PDBFile(PDB)
    ff = app.ForceField("amber96.xml", "amber96_obc.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    bonds = [(b[0].index, b[1].index) for b in pdb.topology.bonds()]
    return pdb, system, bonds, _l_positions()


def short_md(pdb, n_frames, stride=200):
    """Short IMPLICIT-solvent Langevin MD from the L-alanine min config -> frames for the
    whitening (bond/angle) statistics."""
    ff = app.ForceField("amber96.xml", "amber96_obc.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    integ = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
    try:
        sim = app.Simulation(pdb.topology, system, integ, Platform.getPlatformByName("CUDA"))
    except Exception:
        sim = app.Simulation(pdb.topology, system, integ)
    sim.context.setPositions(_l_positions() * unit.nanometer)
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
    ap.add_argument("--method", choices=["klxx", "kl", "asmc"], default="asmc")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--no-delta", dest="no_delta", action="store_true")
    ap.add_argument("--n_valid", type=int); ap.add_argument("--n_pool", type=int)
    ap.add_argument("--n_batch", type=int); ap.add_argument("--steps", type=int)
    args = ap.parse_args()
    if args.no_delta and args.method != "klxx":
        raise SystemExit("--no-delta applies only to --method klxx (kl/asmc carry no delta-QT pool)")

    P = json.load(open(os.path.join(HERE, "config.json")))
    D = int(P.get("dim", P.get("d")))
    N_POOL, N_BATCH, N_VALID = P["n_pool"], P["n_batch"], P["n_valid"]
    STEPS, LR, LAMBDA = P["steps"], P["lr"], P.get("lambda", P.get("lam"))
    BINS, TRANSFORMS, HIDDEN = P["bins"], P["transforms"], tuple(P["hidden"])
    OPT_STEP, OPT_ITERS = P["opt_step"], P["opt_iters"]
    MC_STEP, MC_ITERS = P["mc_step"], P["mc_iters"]
    SMC_RUNGS, SMC_RUNG_ITERS = P["smc_rungs"], P["smc_rung_iters"]
    QT_CHUNK = P["qt_chunk"]
    ADAPTIVE_TAU, VALIDATION_TAU, T_SAFE = P["adaptive_tau"], P["validation_tau"], P["t_safe"]
    SHRINK, ENLARGE = P["shrink_factor"], P["enlarge_factor"]
    MAX_STAGES, MAX_RETRY = P.get("max_stages", 25), P.get("max_retry", 8)
    ESS_GATE = {int(k): float(v) for k, v in P["ess_gate"].items()}
    RELEASE_CACHE, GRAD_CLIP, LR_WARMUP = P.get("release_cache", True), P["grad_clip"], P["lr_warmup"]
    R_FLOOR = P.get("r_floor", P.get("r_max", 0.2))
    E_MIN, E_MAX = P["e_min"], (P.get("e_max") or P.get("e_cap"))
    R_MIN, R_MAX = P.get("r_min"), P.get("r_max")
    DROP = P.get("drop", 0.0)
    GATE_SNAPSHOT = bool(P.get("gate_snapshot", False))
    ANNEAL_MODE = P.get("anneal_mode", "geometric")
    DELTA = (0.0 if args.no_delta else float(P.get("delta", 0.0))) if args.method == "klxx" else 0.0
    set_ess_metric(P.get("ess_metric", "raw"), c=P.get("ess_metric_c", 2.5))

    HIGHD = D > 60
    COMPILE_INV = not HIGHD
    CH = QT_CHUNK * 4 if HIGHD else QT_CHUNK
    NV = args.n_valid or (4000 if args.smoke else N_VALID)
    NP = args.n_pool or (2000 if args.smoke else N_POOL)
    NB = args.n_batch or (500 if args.smoke else N_BATCH)
    ST = args.steps or (60 if args.smoke else STEPS)
    TAG = (("asmc_raw" if args.raw else "asmc") if args.method == "asmc"
           else args.method + ("_delta" if DELTA > 0 else "") + ("_raw" if args.raw else "_sharpen"))
    # no-clobber: if data_<TAG>.pth already exists (e.g. the legacy soft-core asmc), suffix this
    # run's log AND data so the legacy result and its log are preserved (never overwritten).
    RUN = TAG + ("_fab_nosharpen" if (not args.smoke and os.path.exists(os.path.join(HERE, f"data_{TAG}.pth"))) else "")

    LOG = os.path.join(HERE, f"status_{RUN}{'.smoke' if args.smoke else ''}.log"); open(LOG, "w").close()
    def log(m):
        line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")

    log(f"##### {P['molecule'].upper()} IMPLICIT (amber96+OBC) BG  d={D}  method={TAG}  "
        f"pool={NP} batch={NB} valid={NV} steps={ST} delta={DELTA} drop={DROP} device={DEV} #####")
    e_status = ("FAB linlog singular potential, fixed (no anneal/sharpen)" if args.method == "asmc"
                else f"RAW fixed-cap={E_MAX} (no anneal/sharpen)" if args.raw
                else f"e_anneal={E_MIN}->{E_MAX} ({ANNEAL_MODE})  r_anneal={R_MAX}->{R_MIN}")
    log(f"  force field=amber96+amber96_obc (L-alanine start)  {e_status}  grad_clip={GRAD_CLIP}  gate_snapshot={GATE_SNAPSHOT}")

    # ASMC targets FAB's accurate singular potential: the TRUE LJ/Coulomb energy regularized
    # only by boltzgen's barely-active linlog cut (reduced units), NOT the aggressive soft-core
    # (r_floor clamp + softcap). r_floor stays a tiny overflow guard. KL/KLXX keep the soft-core
    # (their per-stage reg anneal relies on it).
    if args.method == "asmc":
        REG_MODE, BUILD_RFLOOR = "linlog", 1.0e-3
        ENERGY_CUT, ENERGY_MAX = float(P.get("energy_cut", 1.0e8)), float(P.get("energy_max", 1.0e20))
    else:
        REG_MODE, BUILD_RFLOOR = "softcap", R_FLOOR
        ENERGY_CUT, ENERGY_MAX = 1.0e8, 1.0e20

    log("  build implicit system + short implicit MD for whitening stats ...")
    pdb, system, bonds, _ = build_implicit()
    frames = short_md(pdb, 200 if args.smoke else 2000)
    B = build(md_frames=frames, T=300.0, device=DEV, dtype=torch.float32, system=system, bonds=bonds,
              r_floor=BUILD_RFLOOR, e_cap=E_MAX, reg_mode=REG_MODE, energy_cut=ENERGY_CUT, energy_max=ENERGY_MAX)
    u, u0 = B["u"], B["u0"]; a, b, wrap = B["a"], B["b"], B["wrap"]
    assert B["n_internal"] == D, (B["n_internal"], D)
    log(f"  torch force field has_gb={u.ff.has_gb}  reg_mode={REG_MODE}"
        f"{f' (energy_cut={ENERGY_CUT:g} energy_max={ENERGY_MAX:g}, r_floor={BUILD_RFLOOR:g})' if REG_MODE=='linlog' else ''}"
        f"  (implicit-solvent target active)")
    for pot in (u0, u):
        pot.enable_grad(mode="default"); pot.enable_eval(mode="default")

    if args.raw:
        E_MIN = E_MAX = R_MIN = R_MAX = None

    def flow_factory():
        f = NCSF(a=a.tolist(), b=b.tolist(), bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN).to(DEV); f.zeros(); return f

    def qt_fn(target):
        x = u0.samples(NP)
        if DELTA > 0.0:
            soft = linear_combination([target], [1.0 - DELTA])
            x = lbfgs(x, soft, step=OPT_STEP, iters=OPT_ITERS, armijo=True, chunk=CH)
            x = wrap(langevin(x, soft, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))
            logw = -DELTA * target.eval(x)
            logw = logw.masked_fill(~torch.isfinite(logw), float("-inf"))
            m = logw.max()
            w_imp = (logw - m).exp() if torch.isfinite(m) else torch.ones_like(logw)
            x = resample(x, w_imp)
            x = wrap(langevin(x, target, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))
        else:
            x = lbfgs(x, target, step=OPT_STEP, iters=OPT_ITERS, armijo=True, chunk=CH)
            x = wrap(langevin(x, target, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))
        w = torch.full((x.shape[0],), 1.0 / x.shape[0], device=x.device)
        return x, w

    t0 = time.perf_counter()
    if args.method == "asmc":
        # pure temperature-annealed SMC on the FIXED FAB-accurate target: no reg anneal, no
        # sharpening -- the classical reference is just GPU compute, no training. The singular
        # (soft-core-free) potential needs tamed ULA drift (taming>0) so the exploding clash
        # forces cannot blow up the Langevin, plus clash screening (drop particles whose reduced
        # energy exceeds min+screen_window and resample the survivors).
        TAMING = float(P.get("taming", 0.01))
        SCREEN_WINDOW = float(P.get("screen_window", 300.0))
        log(f"  ASMC on singular potential: taming={TAMING}  screen_window={SCREEN_WINDOW} kT")
        stages, Y, complete = run_asmc(
            u0, u, n_valid=NV, n_pool=NP, mc_step=MC_STEP, mc_iters=MC_ITERS,
            smc_rungs=SMC_RUNGS, smc_rung_iters=SMC_RUNG_ITERS, adaptive_tau=ADAPTIVE_TAU,
            validation_tau=VALIDATION_TAU, shrink_factor=SHRINK, enlarge_factor=ENLARGE, wrap=wrap,
            device=DEV, status=log, max_stages=MAX_STAGES, max_retry=MAX_RETRY, t_safe=T_SAFE,
            drop=DROP, taming=TAMING, screen_window=SCREEN_WINDOW)
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
    log(f"##### DONE complete={complete}  K={len(stages)}  val_ess={val_ess}  wall={wall:.0f}s #####")

    # phi/psi (deg) of the final SMC ensemble Y -> the Ramachandran; reconstruct full Cartesian.
    from zflows_md.dihedral import dihedral
    PHI, PSI = (4, 6, 8, 14), (6, 8, 14, 16)
    with torch.no_grad():
        cart = u.xi_to_cartesian(Y).detach().cpu().numpy()
    phi, psi = dihedral(cart, PHI), dihedral(cart, PSI)
    log(f"  Ramachandran: {len(phi)} samples  frac(phi>0)={float((phi > 0).mean()):.3f}")

    if args.smoke:
        log(f"smoke OK ({TAG}): run completed, data_<TAG>.pth NOT saved")
    else:
        out = os.path.join(HERE, f"data_{RUN}.pth")   # RUN carries the no-clobber suffix if a legacy run exists
        torch.save(dict(name=P["molecule"], d=D, method=TAG, loss=args.method, stages=stages,
                        ladder=[s["t"] for s in stages], complete=complete, val_ess=val_ess,
                        phi=phi, psi=psi, wall_s=wall, reg_mode=REG_MODE,
                        energy_cut=ENERGY_CUT, energy_max=ENERGY_MAX, config=P), out)
        rama = os.path.join(HERE, "ramachandran_asmc.npz")
        np.savez_compressed(rama, phi=phi, psi=psi,
                            t=[s["t"] for s in stages], val_ess=val_ess,
                            smc_ess=[s["smc_ess"] for s in stages])
        log(f"saved -> {os.path.basename(out)}  and  {os.path.basename(rama)}")


if __name__ == "__main__":
    main()
