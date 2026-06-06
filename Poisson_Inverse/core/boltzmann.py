# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Adaptive-temperature Boltzmann generator training (Algorithm 4, Paper/main.tex).

Potential-invariant: targets enter only as zflows ``Potential`` objects, the
flow enters as a factory returning a fresh identity-initialized zflows ``Flow``,
and the domain enters through a ``wrap`` map (``identity_wrap`` on R^d,
``wrap_torus`` on the torus -- the potential must be periodic so wrapping is
exact and Jacobian-free).

Algorithm 4 wiring, stage k (verbatim step labels):
  (i)   training samples: batches drawn at run time from the validation set
        Y_{k-1}, rejuvenated by Langevin on U_{k-1}
  (ii)  adaptive temperature selection (Algorithm 3): classical SMC from
        mu_{k-1} to mu_k with M rungs, accept the largest t_k whose smallest
        per-rung ESS stays above ADAPIVE_TAU, else shrink by SHRINK_FACTOR
  (iii) wide-coverage set: quench and temper on the stage target U_k
  (iv)  X-regularized forward KL at balanced hyperparameters; the mu_k batch
        is the Algorithm-1 (M=1, G=identity) surrogate: reweight the mu_{k-1}
        batch by exp(U_{k-1}-U_k), resample, short Langevin rejuvenation on U_k
  (v)   validation set update by importance sampling, gated by
        ESS(w) >= VALIDATION_TAU (abort the stage and shrink t_k on failure).

zflows interfaces used directly: linear_combination (bridge potentials),
sequential_monte_carlo (per-rung ESS list for Algorithm 3), langevin, lbfgs,
resample, compute_ESS_log.
"""
import time

import torch
from zflows.potential import Potential, linear_combination
from zflows.loss import loss_compile
from zflows.utils import (langevin, lbfgs, resample, compute_ESS_log,
                          sequential_monte_carlo)


# ---------------------------------------------------------------------------
# domain wrap
# ---------------------------------------------------------------------------
def wrap_torus(x: torch.Tensor, lim: float) -> torch.Tensor:
    """Wrap each coordinate into [-lim, lim). Exact on a 2*lim-periodic
    potential; volume-preserving, so no Jacobian correction anywhere."""
    return torch.remainder(x + lim, 2.0 * lim) - lim


def identity_wrap(x: torch.Tensor) -> torch.Tensor:
    return x


# ---------------------------------------------------------------------------
# losses (conventions copied from the frozen HD_Product_Ladder/core.py).
# loss_KL / loss_X are the EAGER REFERENCE implementations -- training uses
# the compiled fused_stage_loss below; keep these for baselines and checks.
# ---------------------------------------------------------------------------
def loss_KL(y, source: Potential, target: Potential, G):
    x, ladj = G.call_and_ladj(y)            # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    return z.mean()


def loss_X(y, source: Potential, target: Potential, G):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return (z - z[perm]).abs().mean()


def fused_stage_loss(yp, u0: Potential, u: Potential, tp, tn, G, lam, split):
    """KL + lam*X_mu + X_mix on one packed batch, loss_compile-friendly.

    yp = cat([x_mu, x_mix]) with x_mu = yp[:split] (mu_k surrogate batch) and
    x_mix = yp[split:] (PRE-SHUFFLED equal mixture of hat and detached bar
    halves -- shuffling outside the graph makes the cyclic roll(1) pairing
    equivalent to a random permutation). The stage bridges are built inside
    the graph from the 0-d tensor buffers tp = t_{k-1}, tn = t_k (mutated
    in place per stage with .fill_(), so ONE compile serves the whole run):
        U_{k-1}(x) = (1-tp) U_0(x) + tp U(x),   U_k likewise with tn.
    """
    x, ladj = G.call_and_ladj(yp)
    z = ((1.0 - tp) * u0(x) + tp * u(x)
         - (1.0 - tn) * u0(yp) - tn * u(yp) - ladj)
    z_mu, z_mix = z[:split], z[split:]
    return (z_mu.mean() + lam * (z_mu - z_mu.roll(1)).abs().mean()
            + (z_mix - z_mix.roll(1)).abs().mean())


def fused_kl_loss(y, u0: Potential, u: Potential, tp, tn, G):
    """Forward KL ONLY (the baseline method 'kl'): mean of the stage
    log-ratio over the mu_k surrogate batch; no X terms, no mixture.
    Same in-graph bridge construction from the tp/tn buffers as
    fused_stage_loss, so one compile serves all temperature stages."""
    x, ladj = G.call_and_ladj(y)
    z = ((1.0 - tp) * u0(x) + tp * u(x)
         - (1.0 - tn) * u0(y) - tn * u(y) - ladj)
    return z.mean()


# ---------------------------------------------------------------------------
# bridge potentials  U_t = (1 - t) U_0 + t U
# ---------------------------------------------------------------------------
def bridge(u0: Potential, u: Potential, t: float):
    return linear_combination([u, u0], [float(t), 1.0 - float(t)])


# ---------------------------------------------------------------------------
# (iii) wide-coverage set on the torus: uniform melt -> L-BFGS quench ->
#       Langevin temper -> wrap. (Paper Algorithm 2 melts mu_0 samples with a
#       Gaussian scatter; on the torus the maximum-entropy melt IS a fresh
#       uniform draw -- distributionally the same role, no scatter needed.)
# ---------------------------------------------------------------------------
def quench_and_temper_torus(target: Potential, n: int, d: int, lim: float,
                            opt_step: float, opt_iters: int,
                            mc_step: float, mc_iters: int,
                            device) -> torch.Tensor:
    x = (torch.rand(n, d, device=device) * 2.0 - 1.0) * lim       # melt
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)  # quench
    x = langevin(x, target, step=mc_step, iters=mc_iters)        # temper
    return wrap_torus(x, lim)


# ---------------------------------------------------------------------------
# (ii) adaptive temperature selection (Algorithm 3)
# ---------------------------------------------------------------------------
def adaptive_step(pool: torch.Tensor, u0: Potential, u: Potential, t_prev: float,
                  *, tau: float, shrink: float, rungs: int, rung_iters: int,
                  mc_step: float, wrap, status=print, max_shrinks: int = 60,
                  t_init: float = 1.0):
    """Largest admissible t_k below t_init (the safe start t_safe on stage 1,
    thereafter the enlarge-factor extrapolation min(3 t_{k-1} - 2 t_{k-2}, 1),
    Gamma = 2): run M-rung classical SMC from mu_{k-1} to mu_k, accept when
    min per-rung ESS >= tau, else shrink t_k toward t_prev. Returns
    (t_k, ess_min, n_shrinks)."""
    t_k = min(max(t_init, t_prev + 1e-4), 1.0)
    for s in range(max_shrinks):
        u_prev, u_next = bridge(u0, u, t_prev), bridge(u0, u, t_k)
        _, ess = sequential_monte_carlo(wrap(pool.clone()), u_prev, u_next,
                                        ladder=rungs, step=mc_step,
                                        iters=rung_iters)
        ess_min, n_low = min(ess), sum(1 for e in ess if e < tau)
        status(f"    [select] t_k={t_k:.4f}  SMC ESS_min={ess_min:.3f} "
               f"rungs<{tau}: {n_low}/{len(ess)} "
               f"({'accept' if ess_min >= tau else 'shrink'})")
        if ess_min >= tau:
            return t_k, ess_min, s
        t_k = t_prev + shrink * (t_k - t_prev)
    return t_k, ess_min, max_shrinks                              # safety exit


# ---------------------------------------------------------------------------
# (iv) train the stage flow G_k on the X-regularized forward KL
# ---------------------------------------------------------------------------
def train_stage(flow, F_inv, closs, y_valid: torch.Tensor, u_prev: Potential,
                u_next: Potential, hat_pool, *, steps: int,
                batch: int, lr: float, mc_step: float,
                rungs: int, rung_iters: int, wrap, method: str = 'balance',
                grad_clip: float = 1e3, max_skip: int = 10,
                status=print, report_every: int = 1):
    """Per gradient step (Algorithm 4 step (iv); batches per step (i) drawn at
    run time from the validation set):
      X       ~ Y_{k-1}                              (size `batch`)
      Y, ladj = F_inv.inv_ladj(X)                    ONE compiled inverse:
      logw    = U_{k-1}(X) - U_k(Y) + ladj           direct log(mu_k/nu_k),
                its ESS is the logged per-step diagnostic
      Xbar    = Y[:half]                             detached nu_bar half (free)
      X_mu    = M-rung AIS from nu_k to mu_k         (paper Algorithm 1 with
                G the stage flow): per rung, reweight by w^{1/M}, resample,
                Langevin on U_k; the per-rung weight at moved particles is
                recomputed with a cheap FORWARD pass. SMC (Algorithm 3) and
                this AIS are the SAME algorithm (G = identity vs G = flow),
                so they share exactly the same parameters (rungs, rung_iters,
                mc_step).
      Xhat    = langevin(hat_subset, U_k)            (size half)
      loss    = closs(cat[X_mu, shuffle(cat[Xhat, Xbar])])  [compiled fused]
    `F_inv` and `closs` are captured ONCE per run: flow.zeros() and optimizer
    steps mutate parameters in place, which both compiled closures track
    without recompiling (zeros_sanity.py); inverse outputs are cloned out of
    the CUDA-graph static buffers. The VALIDATION update is NOT AIS -- it
    stays a single importance-sampling push (Algorithm 4 step (v)).
    """
    device = y_valid.device
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    Nv, half = y_valid.shape[0], batch // 2
    Ph = hat_pool.shape[0] if hat_pool is not None else 0
    ess_hist, t0, n_skip = [], time.perf_counter(), 0
    for step in range(steps):
        with torch.no_grad():
            G_now = flow.t()                                      # forward (cheap, eager)
            xb = y_valid[torch.randint(0, Nv, (batch,), device=device)]
            y, ladj = F_inv.inv_ladj(xb)
            y, ladj = y.clone(), ladj.clone()                     # out of static buffers
            logw = u_prev.eval(xb) - u_next.eval(y) + ladj        # log mu_k/nu_k
            ess_hist.append(compute_ESS_log(logw).item())
            x_bar = y[:half]                                      # detached nu_bar half
            # M-rung AIS nu_k -> mu_k along the geometric path
            for m in range(rungs):
                inc = logw / rungs
                y = resample(y, (inc - inc.max()).exp())
                y = wrap(langevin(y, u_next, step=mc_step, iters=rung_iters))
                if m < rungs - 1:                                 # forward-only refresh
                    xf, ladj_f = G_now.call_and_ladj(y)
                    logw = u_prev.eval(xf) - u_next.eval(y) - ladj_f
            x_mu = y
            if method == 'balance':
                x_hat = hat_pool[torch.randint(0, Ph, (half,), device=device)]
                x_hat = wrap(langevin(x_hat, u_next, step=mc_step, iters=rung_iters))
                x_mix = torch.cat([x_hat, x_bar], dim=0)
                x_mix = x_mix[torch.randperm(batch, device=device)]  # pre-shuffle (roll)
        loss = (closs(torch.cat([x_mu, x_mix], dim=0)) if method == 'balance'
                else closs(x_mu))                                 # 'kl': forward KL only
        if not torch.isfinite(loss):                              # pathological batch:
            n_skip += 1                                           # params untouched; fresh
            params_ok = all(torch.isfinite(p).all().item()        # batch next step.
                            for p in flow.parameters())           # params_ok=False would
            status(f"    [train] non-finite loss at step {step} "  # mean grad poisoning
                   f"(params finite: {params_ok}); step skipped "  # slipped through.
                   f"({n_skip}/{max_skip})")
            if n_skip >= max_skip or not params_ok:
                status(f"    [train] stage aborted (skips={n_skip}, "
                       f"params finite: {params_ok})")
                return ess_hist, False
            continue
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
        if torch.isfinite(gnorm):                                 # spike -> clipped step
            opt.step()
        else:                                                     # inf/NaN grads: skip the
            n_skip += 1                                           # update BEFORE Adam eats
            status(f"    [train] non-finite grad at step {step} (loss was "
                   f"{loss.item():.3e}); step skipped ({n_skip}/{max_skip})")
            if n_skip >= max_skip:
                status(f"    [train] stage aborted (skips={n_skip})")
                return ess_hist, False
        if (step + 1) % report_every == 0 or step == 0:
            ms = 1000.0 * (time.perf_counter() - t0) / (step + 1)
            status(f"    [train] step {step+1:>5}/{steps}  loss={loss.item():.3e}  "
                   f"direct ESS={ess_hist[-1]:.3f}  {ms:6.1f} ms/step")
    return ess_hist, True


# ---------------------------------------------------------------------------
# (v) validation set update (importance weights of the trained inverse)
# ---------------------------------------------------------------------------
def validation_update(F_inv, y_valid: torch.Tensor, u_prev: Potential,
                      u_next: Potential, *, chunk: int = 500000):
    """Push Y_{k-1} through G_k^{-1} (compiled fused inverse); w =
    exp(U_{k-1}(Y_{k-1}) - U_k(Ytilde) + ladj_inv) (zflows convention:
    ladj_inv = -log|det J_{G_k}(Ytilde)|). Returns (y_tilde, logw, ESS(w))."""
    outs, lws = [], []
    with torch.no_grad():
        for xb in y_valid.split(chunk):
            yt, ladj = F_inv.inv_ladj(xb)
            yt, ladj = yt.clone(), ladj.clone()    # out of the static buffers
            lws.append(u_prev.eval(xb) - u_next.eval(yt) + ladj)
            outs.append(yt)
    y_tilde, logw = torch.cat(outs), torch.cat(lws)
    return y_tilde, logw, compute_ESS_log(logw).item()


# ---------------------------------------------------------------------------
# Algorithm 4 main loop
# ---------------------------------------------------------------------------
def run_boltzmann(u0: Potential, u: Potential, flow_factory, *, n_valid: int,
                  n_pool: int, n_batch: int, steps: int, lr: float, lam: float,
                  mc_step: float, mc_iters: int,
                  smc_rungs: int, smc_rung_iters: int,
                  adaptive_tau: float, validation_tau: float, shrink: float,
                  wrap, qt_fn, device, status=print,
                  max_stages: int = 30, max_retry: int = 6, t_tol: float = 1e-3,
                  t_safe: float = 0.25, method: str = 'balance',
                  grad_clip: float = 1e3, max_skip: int = 10):
    """Returns (stages, Y, complete, flow, F_inv) -- per-stage records (t_k,
    diagnostics, state_dict), the final validation set, whether the ladder
    reached t = 1 (False = INCOMPLETE: a stage failed its gate or max_stages
    hit; final metrics are then NOT target-faithful), and the run's shared
    flow + compiled inverse (for compose_pushforward). `qt_fn(u_next)` must
    return the wide-coverage set on the stage target (step (iii))."""
    Y = u0.samples(n_valid).to(device)
    t_prev, stages, d = 0.0, [], Y.shape[1]
    # Compile-once discipline (verified by zeros_sanity.py): ONE flow for the
    # whole run (reset to identity with flow.zeros() before every attempt --
    # in place, so the captures stay valid), ONE fused-inverse capture, and
    # ONE compiled fused loss serving ALL temperature stages: the stage
    # temperatures live in 0-d buffers mutated with .fill_(), and the batch
    # shape is fixed, so no recompile ever occurs after the first step.
    flow = flow_factory()
    F_inv = flow.t().enable_inv_ladj()
    tp_buf = torch.zeros((), device=device)
    tn_buf = torch.zeros((), device=device)
    if method == 'balance':
        closs = loss_compile(fused_stage_loss, u0, u, tp_buf, tn_buf, flow.t(),
                             float(lam), n_batch)
    else:                                                         # 'kl' baseline
        closs = loss_compile(fused_kl_loss, u0, u, tp_buf, tn_buf, flow.t())
    while t_prev < 1.0 and len(stages) < max_stages:
        k = len(stages) + 1
        u_prev = bridge(u0, u, t_prev)
        # (i) selection pool: draw from Y_{k-1}, rejuvenate on U_{k-1}
        pool = Y[torch.randint(0, n_valid, (n_pool,), device=device)]
        if t_prev > 0.0:
            pool = wrap(langevin(pool, u_prev, step=mc_step, iters=mc_iters))
        # (ii) adaptive temperature selection (Algorithm 4). Initial guess:
        # the safe start t_safe on stage 1 (the leading increment faces the
        # largest deformation; an over-aggressive start is exposed only
        # after a full training), thereafter the enlarge-factor (Gamma = 2)
        # extrapolation min(1, t_{k-1} + 2 (t_{k-1} - t_{k-2})) -- step
        # sizes may grow, only the first few stages are essentially hard.
        ts = [0.0] + [s['t'] for s in stages]
        t_init = min(3.0 * ts[-1] - 2.0 * ts[-2], 1.0) if len(ts) >= 2 else t_safe
        t_k, smc_ess, n_shrink = adaptive_step(
            pool, u0, u, t_prev, tau=adaptive_tau, shrink=shrink,
            rungs=smc_rungs, rung_iters=smc_rung_iters, mc_step=mc_step,
            wrap=wrap, status=status, t_init=t_init)
        if 1.0 - t_k < t_tol:
            t_k = 1.0
        status(f"[stage {k}] t_{{k-1}}={t_prev:.4f} -> t_k={t_k:.4f} "
               f"(SMC ESS_min={smc_ess:.3f}, {n_shrink} shrinks)")
        # (iii)-(v) with the validation-ESS acceptance gate. A stage is
        # accepted ONLY when the gate passes; the accepted (t_k, flow,
        # y_tilde, logw, u_next) are captured at the moment of acceptance.
        attempts, accepted = [], None
        for attempt in range(max_retry):
            u_next = bridge(u0, u, t_k)
            hat = qt_fn(u_next) if method == 'balance' else None  # (iii)
            flow.zeros()                                          # identity reset, in place
            tp_buf.fill_(t_prev)
            tn_buf.fill_(t_k)
            ess_hist, ok = train_stage(                           # (iv)
                flow, F_inv, closs, Y, u_prev, u_next, hat, steps=steps,
                batch=n_batch, lr=lr, mc_step=mc_step,
                rungs=smc_rungs, rung_iters=smc_rung_iters,
                wrap=wrap, method=method, grad_clip=grad_clip,
                max_skip=max_skip, status=status)
            y_tilde, logw, val_ess = validation_update(F_inv, Y, u_prev, u_next)
            attempts.append(dict(t_k=t_k, val_ess=val_ess, train_ok=ok,
                                 accepted=bool(ok and val_ess >= validation_tau),
                                 ess_hist=list(ess_hist)))   # full per-step ESS,
                                                             # rejected attempts too
            status(f"[stage {k}] attempt {attempt+1}: validation ESS={val_ess:.3f} "
                   f"(floor {validation_tau})")
            if ok and val_ess >= validation_tau:
                accepted = dict(t_k=t_k, flow=flow, y_tilde=y_tilde, logw=logw,
                                u_next=u_next, val_ess=val_ess, ess_hist=ess_hist)
                break
            t_k = t_prev + shrink * (t_k - t_prev)                # abort & shrink
            status(f"[stage {k}] abort -> shrink to t_k={t_k:.4f}")
        if accepted is None:
            status(f"[stage {k}] STAGE FAILED: validation ESS never reached "
                   f"{validation_tau} in {max_retry} attempts; stopping with an "
                   f"INCOMPLETE ladder (reached t={t_prev:.4f})")
            return stages, Y, False, flow, F_inv
        # (v) accept: resample + rejuvenate the validation set
        t_k, u_next = accepted['t_k'], accepted['u_next']
        logw = accepted['logw']
        Y = resample(accepted['y_tilde'], (logw - logw.max()).exp())
        Y = wrap(langevin(Y, u_next, step=mc_step, iters=mc_iters))
        stages.append(dict(
            t=t_k, smc_ess=smc_ess, n_shrink=n_shrink,
            val_ess=accepted['val_ess'], attempts=attempts,
            train_ess_hist=accepted['ess_hist'],
            state_dict={key: v.cpu().clone()
                        for key, v in accepted['flow'].state_dict().items()}))
        t_prev = t_k
    return stages, Y, (t_prev >= 1.0), flow, F_inv


# ---------------------------------------------------------------------------
# final evaluation: compose the stage inverses and accumulate the weights
# ---------------------------------------------------------------------------
def compose_pushforward(flow, F_inv, state_dicts, u0: Potential, u: Potential,
                        n: int, device, chunk: int = 500000):
    """Generate y = G_K^{-1}(... G_1^{-1}(x)) for x ~ mu_0 and return
    (y, logw) with logw = u0(x) - u(y) + sum_k ladj_inv_k (the direct
    importance weight of the composed generator against the target).
    Reuses the run's single flow + compiled inverse: each stage's state_dict
    is loaded in place (the captured F_inv keeps tracking the parameters)."""
    x = u0.samples(n).to(device)
    y = x
    ladj_sum = torch.zeros(n, device=device)
    with torch.no_grad():
        for sd in state_dicts:                                    # stages 1..K
            flow.load_state_dict({k: v.to(device) for k, v in sd.items()})
            outs, lds = [], []
            for yb in y.split(chunk):
                yt, ladj = F_inv.inv_ladj(yb)
                outs.append(yt.clone())            # out of the static buffers
                lds.append(ladj.clone())
            y = torch.cat(outs)
            ladj_sum = ladj_sum + torch.cat(lds)
        logw = u0.eval(x) - u.eval(y) + ladj_sum
    return y, logw
