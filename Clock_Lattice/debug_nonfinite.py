# Deterministic repro + instrumentation of the stage-1 non-finite abort at
# t_init = 0.25 (L=8, B=100k, seed 0; crashed at train-step index 12 at both
# lr=1e-3 and lr=5e-4).  Mirrors train.py's construction order exactly so the
# RNG stream matches, monkeypatches core.boltzmann.train_stage with a copy
# that checks, per step: loss finiteness, per-term z statistics (eager
# recompute), gradient norm/finiteness after backward, and parameter
# finiteness after opt.step().  Exits at the first non-finite occurrence with
# the culprit named.  Writes findings to debug_nonfinite.log.
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zflows.flow import NCSF
from zflows.potential import Uniform
from zflows.utils import compute_ESS_log, resample, langevin, suppress_warnings
import core.boltzmann as BZ
from core import wrap_torus, quench_and_temper_torus
import parameters as PRM
from potential import Clock

suppress_warnings()
device = 'cuda'
LOG = HERE / 'debug_nonfinite.log'


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    with open(LOG, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


def stats(name, t):
    finite = torch.isfinite(t).all().item()
    return (f"{name}: finite={finite} min={t.min().item():.3e} "
            f"max={t.max().item():.3e} absmean={t.abs().mean().item():.3e}")


def train_stage_debug(flow, F_inv, closs, y_valid, u_prev, u_next, hat_pool, *,
                      steps, batch, lr, mc_step, rungs, rung_iters, wrap,
                      method='balance', status=print, report_every=1):
    """Copy of core.boltzmann.train_stage with per-step finite checks."""
    dev = y_valid.device
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    Nv, half = y_valid.shape[0], batch // 2
    Ph = hat_pool.shape[0] if hat_pool is not None else 0
    ess_hist = []
    for step in range(min(steps, 20)):                  # crash is at idx 12
        with torch.no_grad():
            G_now = flow.t()
            xb = y_valid[torch.randint(0, Nv, (batch,), device=dev)]
            y, ladj = F_inv.inv_ladj(xb)
            y, ladj = y.clone(), ladj.clone()
            logw = u_prev.eval(xb) - u_next.eval(y) + ladj
            ess_hist.append(compute_ESS_log(logw).item())
            x_bar = y[:half]
            for m in range(rungs):
                inc = logw / rungs
                y = resample(y, (inc - inc.max()).exp())
                y = wrap(langevin(y, u_next, step=mc_step, iters=rung_iters))
                if m < rungs - 1:
                    xf, ladj_f = G_now.call_and_ladj(y)
                    logw = u_prev.eval(xf) - u_next.eval(y) - ladj_f
            x_mu = y
            x_hat = hat_pool[torch.randint(0, Ph, (half,), device=dev)]
            x_hat = wrap(langevin(x_hat, u_next, step=mc_step, iters=rung_iters))
            x_mix = torch.cat([x_hat, x_bar], dim=0)
            x_mix = x_mix[torch.randperm(batch, device=dev)]
            yp = torch.cat([x_mu, x_mix], dim=0)

            # ---- eager z diagnostics on the packed batch ----
            x_eag, ladj_eag = G_now.call_and_ladj(yp)
            log(f"step {step:>2}  " + stats("ladj(yp)", ladj_eag))
            if not torch.isfinite(yp).all():
                log(f"step {step}  !! NON-FINITE INPUT BATCH yp (sampling chain)")
                log("  " + stats("x_mu", x_mu) + " | " + stats("x_mix", x_mix))
                sys.exit(1)

        loss = closs(yp)
        if not torch.isfinite(loss):
            log(f"step {step}  !! NON-FINITE LOSS (params were finite at entry)")
            sys.exit(1)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), float('inf'))
        bad_grads = [n for n, p in flow.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        opt.step()
        bad_params = [n for n, p in flow.named_parameters()
                      if not torch.isfinite(p).all()]
        log(f"step {step:>2}  loss={loss.item():.4e}  ESS={ess_hist[-1]:.3f}  "
            f"grad_norm={gn.item():.4e}  bad_grads={len(bad_grads)}  "
            f"bad_params={len(bad_params)}")
        if bad_grads:
            log(f"step {step}  !! NON-FINITE GRADS in: {bad_grads[:8]}")
        if bad_params:
            log(f"step {step}  !! NON-FINITE PARAMS after opt.step(): "
                f"{bad_params[:8]}")
            log(">> DIAGNOSIS: gradient blow-up -> Adam stepped params to "
                "non-finite; loss guard fires one step late.")
            sys.exit(0)
    log("reached 20 steps with everything finite -- no repro (?)")
    return ess_hist, True


BZ.train_stage = train_stage_debug                      # monkeypatch

# ---- mirror train.py __main__ construction exactly (seed stream!) ----
L, method = 8, 'balance'
D = L * L
NV, NP, NB, ST = PRM.N_VALID, PRM.N_POOL, PRM.N_BATCH, PRM.STEPS
MC, RUNG, OPTI = PRM.MC_ITERS, PRM.SMC_RUNG_ITERS, PRM.OPT_ITERS

torch.manual_seed(0)
lim = PRM.NSF_LIM
u0 = Uniform([-lim] * D, [lim] * D, device=device)
u = Clock(L, PRM.P, PRM.J, PRM.H).to(device)
for pot in (u0, u):
    pot.enable_grad(mode="reduce-overhead")
    pot.enable_eval(mode="default")


def wrap(x):
    return wrap_torus(x, lim)


def flow_factory():
    flow = NCSF(a=[-lim] * D, b=[lim] * D, bins=PRM.BINS,
                transforms=PRM.TRANSFORMS,
                hidden_features=PRM.HIDDEN).to(device)
    flow.zeros()
    return flow


def qt_fn(u_next):
    return quench_and_temper_torus(u_next, NP, D, lim, PRM.OPT_STEP, OPTI,
                                   PRM.MC_STEP, MC, device)


log(f"##### DEBUG REPRO START t_safe=0.25 lr=1e-3 B={NB} #####")
BZ.run_boltzmann(
    u0, u, flow_factory, n_valid=NV, n_pool=NP, n_batch=NB, steps=ST,
    lr=1e-3, lam=PRM.LAMBDA,
    mc_step=PRM.MC_STEP, mc_iters=MC,
    smc_rungs=PRM.SMC_RUNGS, smc_rung_iters=RUNG,
    adaptive_tau=PRM.ADAPIVE_TAU,
    validation_tau=PRM.VALIDATION_TAU, shrink=PRM.SHRINK_FACTOR,
    wrap=wrap, qt_fn=qt_fn, device=device, status=log,
    max_stages=1, max_retry=1, t_safe=0.25, method=method)
log("##### DEBUG REPRO END #####")
