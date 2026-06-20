# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""General prmtop-molecule full-internal Boltzmann generator (KLXX vs forward KL).

For molecules WITH HETEROATOMS (N/O/Cl/S) supplied as an AMBER prmtop + rst7 —
e.g. N-methylacetamide (`nma`, the minimal amide / peptide-bond model, d=30).
Same engine + output format as alkane_bg.py (so make_summary.py picks it up).

Run: python hetero_bg.py --prmtop tests/data/nma.prmtop --crd tests/data/nma.rst7 \
                        --name nma --method klxx
"""
import argparse, os, sys, time, json, shutil
from datetime import datetime
import numpy as np
import torch
import parmed as pmd
from openmm import app, unit, LangevinMiddleIntegrator, Platform

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))           # zflows-md root
from zflows_md.flow import NCSF
from zflows_md.utils import (compute_ESS_log, resample, suppress_warnings, lbfgs, langevin,
                             set_cache_size_limit, set_ess_metric, get_ess_khat)
from zflows_md.boltzmann import build, run_boltzmann, run_asmc, compose_pushforward
from zflows_md.potential import linear_combination
from zflows_md.bg.bgconfig import load_config
from zflows_md.bg.mode_coverage import build_mode_checker            # mode-coverage acceptance gate (opt-in)

suppress_warnings()
set_cache_size_limit(128)                               # raise dynamo cache (default 8) so the
                                                        # compiled inverse/loss don't fall back to eager
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- ALL parameters come from config.json via bgconfig.load_config(d) (single source of
#      truth; edit config.json, not this file). Resolved per-molecule in main(). ----


def short_md(prmtop, crd, n_frames, stride=200):
    s = pmd.load_file(prmtop, xyz=crd)
    system = s.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
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
    ap.add_argument("--prmtop", required=True); ap.add_argument("--crd", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", choices=["klxx", "kl", "asmc"], default="klxx")  # asmc = identity-flow annealed SMC baseline
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--raw", action="store_true")    # ablation: fixed e_cap=e_max throughout, no cap anneal, no sharpening
    ap.add_argument("--delta", action="store_true")  # klxx only: delta-weighted QT (temper on (1-delta)U, reweight e^{-delta U})
    ap.add_argument("--n_valid", type=int); ap.add_argument("--n_pool", type=int)
    ap.add_argument("--n_batch", type=int); ap.add_argument("--steps", type=int)
    ap.add_argument("--grad_clip", type=float, default=None)   # override config (e.g. nma re-run at 2.5)
    ap.add_argument("--resume_keep", type=int, default=0)      # resume: replay first N saved stages of data_<TAG>.pth, retrain from N+1
    ap.add_argument("--e_min", type=float, default=None)       # override config cap-anneal range (e.g. glycerol config.json has null)
    ap.add_argument("--e_max", type=float, default=None)
    ap.add_argument("--ess_metric", choices=["raw", "smooth", "psis"], default=None)  # override config; opt-in robust ESS
    args = ap.parse_args()
    D = 3 * len(pmd.load_file(args.prmtop).atoms) - 6       # internal DOF (single molecule)
    P = load_config(D, args.name)                           # authoritative per-molecule <name>_<D>d/config.json
    N_POOL, N_BATCH, N_VALID = P["n_pool"], P["n_batch"], P["n_valid"]
    STEPS, LR, LAMBDA = P["steps"], P["lr"], P["lambda"]
    BINS, TRANSFORMS, HIDDEN = P["bins"], P["transforms"], tuple(P["hidden"])
    OPT_STEP, OPT_ITERS = P["opt_step"], P["opt_iters"]
    MC_STEP, MC_ITERS = P["mc_step"], P["mc_iters"]
    SMC_RUNGS, SMC_RUNG_ITERS = P["smc_rungs"], P["smc_rung_iters"]
    QT_CHUNK = P["qt_chunk"]
    ADAPIVE_TAU, VALIDATION_TAU, T_SAFE = P["adaptive_tau"], P["validation_tau"], P["t_safe"]
    SHRINK, MAX_STAGES, MAX_RETRY = P["shrink_factor"], P["max_stages"], P["max_retry"]
    MAX_REFRESH = P.get("max_refresh", 0)              # refresh-before-reject retries (0 = off; resample+Langevin Y_{k-1} on low val ESS)
    ENLARGE_FACTOR = P["enlarge_factor"]               # next t_init extrapolation on SUCCESS (default 2)
    ESS_GATE = P["ess_gate"]; RELEASE_CACHE = P["release_cache"]
    GRAD_CLIP = args.grad_clip if args.grad_clip is not None else P["grad_clip"]
    LR_WARMUP = P["lr_warmup"]
    R_FLOOR = P.get("r_floor", P.get("r_max", 0.2))    # build base; if only the r-anneal (r_min/r_max) is set, start at r_max (soft)
    E_MIN = args.e_min if args.e_min is not None else P["e_min"]    # CLI override (glycerol config.json has null e_min/e_max)
    E_MAX = args.e_max if args.e_max is not None else P["e_max"]    # cap-anneal range (design.tex Sec.3)
    RAW = args.raw
    MODE_GATE = bool(P.get("mode_gate", False))        # mode-coverage gate on the POST-SHARPEN deliverable (off by default -> 48d/36d unchanged)
    GATE_SNAPSHOT = bool(P.get("gate_snapshot", False)) # opt-in: snapshot at each ess_gate, accept the BEST-passing (no fail-refresh)
    ANNEAL_MODE = P.get("anneal_mode", "geometric")    # e/r cap schedule: "geometric" (default, exp in t) or "arithmetic" (linear in t)
    # delta-weighted QT softens the wide-coverage pool's temper to (1-delta)U and reweights by e^{-delta U};
    # it only touches qt_fn, which is klxx-only (kl carries no wide-coverage pool -> --delta is meaningless).
    if args.delta and args.method != "klxx":
        raise SystemExit("--delta applies only to --method klxx (kl has no wide-coverage QT pool)")
    DELTA = float(P.get("delta", 0.1)) if args.delta else 0.0
    # output id (data_<TAG>.pth): {loss}{_delta?}{_sharpen|_raw} -- sharpening runs carry the MC sharpening step
    TAG = "asmc" if args.method == "asmc" else args.method + ("_delta" if args.delta else "") + ("_raw" if RAW else "_sharpen")
    # raw ablation: e_min=e_max=None -> run_boltzmann e_anneal=False -> set_regularization never called,
    # sharpening skipped; the cap stays at the build value E_MAX (fixed e_cap=e_max). NOT e_min=e_max=200.
    EA_MIN, EA_MAX = (None, None) if RAW else (E_MIN, E_MAX)
    R_MIN, R_MAX = (None, None) if RAW else (P.get("r_min"), P.get("r_max"))   # nonbonded soft-core anneal r_max->r_min (off if unset)
    DROP = P.get("drop", 0.0)              # discard the worst `drop` fraction in the ESS calc + resample (0 = off)
    ESS_METRIC = args.ess_metric if args.ess_metric is not None else P.get("ess_metric", "raw")  # raw/smooth/psis (opt-in)
    ESS_METRIC_C = P.get("ess_metric_c", 2.5)   # smooth-ESS weight-bound size (larger=gentler; c=6 -> raw 0.4 ~ smooth 0.5)
    set_ess_metric(ESS_METRIC, c=ESS_METRIC_C)  # set ONCE before any compute_ESS_log; default "raw" = unchanged (48d/36d)
    NV = args.n_valid or (4000 if args.smoke else N_VALID)
    NP = args.n_pool or (2000 if args.smoke else N_POOL)
    NB = args.n_batch or (500 if args.smoke else N_BATCH)
    ST = args.steps or (60 if args.smoke else STEPS)

    frames = short_md(args.prmtop, args.crd, 200 if args.smoke else 2000)
    B = build(args.prmtop, args.crd, md_frames=frames, T=300.0, device=DEV, dtype=torch.float32,
              r_floor=R_FLOOR, e_cap=E_MAX)                 # initial cap = e_max; run_boltzmann anneals e_min->e_max
    u, u0 = B["u"], B["u0"]; D_b = B["n_internal"]; a, b, wrap = B["a"], B["b"], B["wrap"]
    assert D_b == D, (D_b, D)
    HIGHD = D > 60                # compile the inverse up to d=60 (full 16GB VRAM); raw only beyond

    OD = (os.path.join(REPO, "tests", f"smoke_{args.name}_{D}d") if args.smoke
          else os.path.join(REPO, f"{args.name}_{D}d"))
    os.makedirs(OD, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(OD, "run.py"))
    LOG = os.path.join(OD, f"status_{TAG}.log"); open(LOG, "w").close()   # per-method log: never clobber another method's run record

    def log(m):
        line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")

    log(f"##### {args.name.upper()} (prmtop) FULL-INTERNAL d={D} method={TAG} "
        f"pool={NP} batch={NB} valid={NV} drop={DROP} max_refresh={MAX_REFRESH} ess_metric={ESS_METRIC}(c={ESS_METRIC_C}) #####")
    for pot in (u0, u):
        pot.enable_grad(mode="default"); pot.enable_eval(mode="default")

    def flow_factory():
        f = NCSF(a=a.tolist(), b=b.tolist(), bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN).to(DEV); f.zeros(); return f

    CH = QT_CHUNK * 4 if HIGHD else QT_CHUNK

    def qt_fn(target):                                     # target = U_k (the stage target)
        # Algorithm 2/3 (design.tex): quench (L-BFGS) then temper (Langevin) -> a wide-coverage
        # UNWEIGHTED pool of mu_k samples; train_stage resamples the mini-batch from it.
        x = u0.samples(NP)
        if DELTA > 0.0:
            # delta-weighted QT: quench+temper on the FLATTENED (1-delta)*U_k for wider barrier crossing,
            # then reweight by e^{-delta U_k} and rejuvenate on U_k -> unbiased UNWEIGHTED pool of mu_k.
            soft = linear_combination([target], [1.0 - DELTA])                              # (1-delta)*U_k
            x = lbfgs(x, soft, step=OPT_STEP, iters=OPT_ITERS, armijo=True, chunk=CH)       # quench on (1-d)U_k
            x = wrap(langevin(x, soft, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH)) # temper on (1-d)U_k
            logw = -DELTA * target.eval(x)                                                  # w propto e^{-delta U_k}
            logw = logw.masked_fill(~torch.isfinite(logw), float("-inf"))                   # the softened temper can wander to degenerate geometries (NaN/Inf energy); drop them (0 weight)
            m = logw.max()
            w_imp = (logw - m).exp() if torch.isfinite(m) else torch.ones_like(logw)        # finite guard for the resample multinomial
            x = resample(x, w_imp)                                                          # reweight to U_k (wandered configs resampled out)
            x = wrap(langevin(x, target, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))  # rejuvenate on U_k
        else:
            x = lbfgs(x, target, step=OPT_STEP, iters=OPT_ITERS, armijo=True, chunk=CH)        # quench on U_k
            x = wrap(langevin(x, target, step=MC_STEP, iters=MC_ITERS, taming=0.0, chunk=CH))  # temper on U_k
        w = torch.full((x.shape[0],), 1.0 / x.shape[0], device=x.device)                   # unweighted pool
        return x, w

    compile_inv = not HIGHD
    config = dict(molecule=args.name, d=D, n_pool=NP, n_batch=NB, n_valid=NV, steps=ST,
                  lr=LR, lam=LAMBDA, bins=BINS, transforms=TRANSFORMS, hidden=list(HIDDEN),
                  opt_step=OPT_STEP, opt_iters=OPT_ITERS, mc_step=MC_STEP, mc_iters=MC_ITERS,
                  smc_rungs=SMC_RUNGS, smc_rung_iters=SMC_RUNG_ITERS, adaptive_tau=ADAPIVE_TAU,
                  validation_tau=VALIDATION_TAU, t_safe=T_SAFE, qt_chunk=CH, compile_inv=compile_inv,
                  high_d_raw_inverse=HIGHD, temperature_K=300.0, prmtop=os.path.basename(args.prmtop),
                  r_floor=R_FLOOR, r_min=R_MIN, r_max=R_MAX, drop=DROP, e_min=EA_MIN, e_max=EA_MAX, e_cap=E_MAX, raw=RAW,
                  grad_clip=GRAD_CLIP, delta=DELTA, lr_warmup=LR_WARMUP, max_refresh=MAX_REFRESH, ess_metric=ESS_METRIC, ess_metric_c=ESS_METRIC_C,
                  mode_gate=MODE_GATE, mode_cover_tau=P.get("mode_cover_tau", 0.25), gate_snapshot=GATE_SNAPSHOT,
                  anneal_mode=ANNEAL_MODE)
    # config.json is the SOLE, read-only runtime INPUT and is never written here. No extra config
    # file is created in OD; the resolved run config is kept only inside the torch.save data record
    # below (config=config) for analysis.
    log(f"{'RAW fixed-cap (no anneal/sharpen)' if RAW else 'sharpening'} (d={D}): r_floor={R_FLOOR}, "
        f"e_min={EA_MIN}, e_max={EA_MAX}, anneal={ANNEAL_MODE}, fixed_cap={E_MAX if RAW else '-'}, grad_clip={GRAD_CLIP}, "
        f"lr_warmup={LR_WARMUP}; NO skip-abort (gate ESS is the only standard)")
    resume_stages = None
    if args.resume_keep > 0:                                # warm-start from a prior run's saved stages
        _prev = torch.load(os.path.join(OD, f"data_{TAG}.pth"), weights_only=False)
        resume_stages = _prev["stages"][:args.resume_keep]
        log(f"[resume] keeping first {len(resume_stages)} stages of data_{TAG}.pth "
            f"(t={[round(s['t'], 3) for s in resume_stages]}); retraining from stage {len(resume_stages)+1}")
    t0 = time.perf_counter()
    def _checkpoint(_stages, _complete):                     # crash-safe: persist resumable data after each stage
        torch.save(dict(name=args.name, d=D, method=TAG, loss=args.method, raw=RAW, stages=_stages,
                        ladder=[s["t"] for s in _stages], complete=_complete,
                        val_ess=[round(s["val_ess"], 4) for s in _stages],
                        sharpen_ess=[s.get("sharpen_ess") for s in _stages],
                        full_target_ess=None, bg_frac={}, wall_s=time.perf_counter() - t0, config=config),
                   os.path.join(OD, f"data_{TAG}.pth"))
        log(f"[checkpoint] {len(_stages)} stage(s) -> data_{TAG}.pth (complete={_complete}); "
            f"resume via --resume_keep {len(_stages)}")
    def _fail_checkpoint(_k, _attempt, _t_k, _val_ess, _ok, _flow, _ess_hist):  # debug: overwrite latest FAILED attempt
        torch.save(dict(name=args.name, d=D, method=TAG, loss=args.method, raw=RAW, failed=True,
                        stage=_k, attempt=_attempt, t_k=_t_k, val_ess=_val_ess, train_ok=_ok,
                        ess_hist=_ess_hist, wall_s=time.perf_counter() - t0, config=config,
                        state_dict={key: v.cpu().clone() for key, v in _flow.state_dict().items()}),
                   os.path.join(OD, f"data_{TAG}_failed.pth"))
        log(f"[failed-checkpoint] stage {_k} attempt {_attempt} (t_k={_t_k:.4f}, val_ess={_val_ess:.4f}) "
            f"-> data_{TAG}_failed.pth (overwritten)")
    mode_check_fn = None
    if MODE_GATE:                                          # build the INDEPENDENT mode template ONCE (uniform-torsion Langevin @ soft cap)
        def _set_soft():                                   # soft cap (e_min, + r_max if r-anneal on) -- torsion modes are cap-independent
            if EA_MIN is not None: u.set_regularization(EA_MIN)
            if R_MAX is not None: u.set_r_floor(R_MAX)
        log("[mode-gate] building independent mode template (uniform-torsion Langevin @ soft cap)...")
        mode_check_fn = build_mode_checker(
            B, u, u0, wrap, set_soft_cap=_set_soft, mc_step=MC_STEP, prm=args.prmtop, device=DEV,
            n_walk=P.get("mode_walk", 50000), iters=P.get("mode_iters", 4000),
            top=P.get("mode_top", 6), tau_cov=P.get("mode_cover_tau", 0.25),
            p_min=P.get("mode_p_min", 0.10), status=log)
    if args.method == "asmc":                                # lean pure annealed-SMC baseline: no flow/inverse/compile/snapshot
        stages, Y, complete = run_asmc(
            u0, u, n_valid=NV, n_pool=NP, mc_step=MC_STEP, mc_iters=MC_ITERS,
            smc_rungs=SMC_RUNGS, smc_rung_iters=SMC_RUNG_ITERS,
            adaptive_tau=ADAPIVE_TAU, validation_tau=VALIDATION_TAU, shrink_factor=SHRINK,
            enlarge_factor=ENLARGE_FACTOR, wrap=wrap, device=DEV, status=log,
            max_stages=MAX_STAGES, max_retry=MAX_RETRY, t_safe=T_SAFE,
            e_min=EA_MIN, e_max=EA_MAX, r_min=R_MIN, r_max=R_MAX, drop=DROP,
            anneal_mode=ANNEAL_MODE, resume_stages=resume_stages, checkpoint_fn=_checkpoint)
        flow = F_inv = None
    else:
        stages, Y, complete, flow, F_inv = run_boltzmann(
            u0, u, flow_factory, n_valid=NV, n_pool=NP, n_batch=NB, steps=ST, lr=LR, lam=LAMBDA,
            mc_step=MC_STEP, mc_iters=MC_ITERS, smc_rungs=SMC_RUNGS, smc_rung_iters=SMC_RUNG_ITERS,
            adaptive_tau=ADAPIVE_TAU, validation_tau=VALIDATION_TAU, shrink_factor=SHRINK,
            enlarge_factor=ENLARGE_FACTOR, wrap=wrap,
            qt_fn=qt_fn, device=DEV, status=log, max_stages=MAX_STAGES, max_retry=MAX_RETRY,
            max_refresh=MAX_REFRESH,
            t_safe=T_SAFE, method=args.method, grad_clip=GRAD_CLIP,
            taming=0.0, lr_warmup=LR_WARMUP, e_min=EA_MIN, e_max=EA_MAX, r_min=R_MIN, r_max=R_MAX, drop=DROP, compile_inv=compile_inv,
            ess_gate=ESS_GATE, release_cache=RELEASE_CACHE, resume_stages=resume_stages,
            checkpoint_fn=_checkpoint, fail_checkpoint_fn=_fail_checkpoint, mode_check_fn=mode_check_fn,
            gate_snapshot=GATE_SNAPSHOT, anneal_mode=ANNEAL_MODE)
    if complete and stages and flow is not None:              # (vi) end-to-end ESS of the COMPOSED generator vs the TRUE target (flow runs only)
        _ftn = min(NV, 200000)                                # u is left at the t=1 sharp reg after the run; raw (no drop) = honest
        _, _ftlw = compose_pushforward(flow, F_inv, [s["state_dict"] for s in stages], u0, u, _ftn, DEV, chunk=NB)
        stages[-1]["full_target_ess"] = compute_ESS_log(_ftlw).item()
        log(f"[full-target] composed end-to-end ESS={stages[-1]['full_target_ess']:.4f} (n={_ftn})")
    wall = time.perf_counter() - t0
    val_ess = [round(s["val_ess"], 4) for s in stages]
    fte = stages[-1].get("full_target_ess") if stages else None
    log(f"##### {args.name} DONE complete={complete} K={len(stages)} val_ess={val_ess} "
        f"full_target_ess={fte} wall={wall:.0f}s #####")

    sharpen_ess = [None if s.get("sharpen_ess") is None else round(s["sharpen_ess"], 4) for s in stages]
    log(f"  val_ess={val_ess}  sharpen_ess={sharpen_ess}")
    torch.save(dict(name=args.name, d=D, method=TAG, loss=args.method, raw=RAW, stages=stages,
                    ladder=[s["t"] for s in stages], complete=complete, val_ess=val_ess,
                    sharpen_ess=sharpen_ess, full_target_ess=fte, bg_frac={}, wall_s=wall, config=config),
               os.path.join(OD, f"data_{TAG}.pth"))

    # combined KLXX vs KL table (same shape as alkane_bg)
    def _load(m):
        p = os.path.join(OD, f"data_{m}.pth")
        return torch.load(p, weights_only=False) if os.path.exists(p) else None
    METHODS = ("klxx_sharpen", "kl_sharpen", "klxx_raw", "kl_raw", "klxx_delta_sharpen", "klxx_delta_raw", "asmc")  # sharpen/raw + delta-QT + identity-flow ASMC baseline
    def _sh(s):
        v = s.get("sharpen_ess"); return "nan" if v is None else f"{v:.4f}"
    with open(os.path.join(OD, "results_table.csv"), "w") as f:
        f.write("method,stage,t_k,smc_ess,validation_ess,sharpen_ess,n_shrink\n")
        for m in METHODS:
            dd = _load(m)
            if dd:
                for i, s in enumerate(dd["stages"]):
                    f.write(f"{m},{i+1},{s['t']:.4f},{s.get('smc_ess', float('nan')):.4f},"
                            f"{s['val_ess']:.4f},{_sh(s)},{s.get('n_shrink', 0)}\n")
    with open(os.path.join(OD, "results_table.md"), "w") as f:
        f.write(f"# {args.name} (d={D}, heteroatoms via {os.path.basename(args.prmtop)}) — KLXX vs forward KL\n\n")
        f.write(f"pool={NP} batch={NB} valid={NV} | NCSF bins={BINS} transforms={TRANSFORMS} "
                f"hidden={HIDDEN} | compiled_inverse={compile_inv}\n")
        for m, lab in [("klxx_sharpen", "KL+X sharpening (`klxx_sharpen`)"), ("kl_sharpen", "forward-KL sharpening (`kl_sharpen`)"),
                       ("klxx_raw", "KL+X fixed-cap (`klxx_raw`)"), ("kl_raw", "forward-KL fixed-cap (`kl_raw`)"),
                       ("klxx_delta_sharpen", "KL+X sharpening + delta-QT (`klxx_delta_sharpen`)"),
                       ("klxx_delta_raw", "KL+X fixed-cap + delta-QT (`klxx_delta_raw`)"),
                       ("asmc", "annealed-SMC baseline, identity flow (`asmc`)")]:
            dd = _load(m)
            if not dd:
                f.write(f"\n*{lab}: not run yet.*\n"); continue
            F = 1.0
            for v, sh in zip(dd["val_ess"], dd["sharpen_ess"]):
                F *= 1.0 / max(v, 1e-6)
                if sh is not None:                      # sharpening: include the MC sharpening ESS (corrected-F rule)
                    F *= 1.0 / max(sh, 1e-6)
            f.write(f"\n## {lab} — complete={dd['complete']}, K={len(dd['stages'])}, **F={F:.2f}**"
                    + (f", full-target ESS={dd['full_target_ess']:.3f}" if dd.get('full_target_ess') is not None else "") + "\n\n")
            f.write("| stage | t_k | SMC ESS | validation ESS | sharpen ESS |\n|---|---|---|---|---|\n")
            for i, s in enumerate(dd["stages"]):
                sh = s.get("sharpen_ess")
                f.write(f"| {i+1} | {s['t']:.3f} | {s.get('smc_ess', float('nan')):.3f} | "
                        f"**{s['val_ess']:.3f}** | {'-' if sh is None else f'{sh:.3f}'} |\n")
    log(f"wrote {OD}/ : data_{TAG}.pth, results_table.{{md,csv}}")


if __name__ == "__main__":
    main()
