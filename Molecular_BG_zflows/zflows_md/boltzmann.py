# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Adaptive-temperature Boltzmann generator training.

Potential-invariant: targets enter only as zflows ``Potential`` objects, the
flow enters as a factory returning a fresh identity-initialized zflows ``Flow``,
and the domain enters through a ``wrap`` map (``identity_wrap`` on R^d,
``wrap_torus`` on the torus -- the potential must be periodic so wrapping is
exact and Jacobian-free).

Training wiring, stage k (verbatim step labels):
  (i)   training samples: batches drawn at run time from the validation set
        Y_{k-1}, rejuvenated by Langevin on U_{k-1}
  (ii)  adaptive temperature selection: classical SMC from
        mu_{k-1} to mu_k with M rungs, accept the largest t_k whose smallest
        per-rung ESS stays above ADAPIVE_TAU, else shrink by SHRINK_FACTOR
  (iii) wide-coverage set: quench and temper on the stage target U_k
  (iv)  X-regularized forward KL at balanced hyperparameters; the mu_k batch
        is the M=1, G=identity surrogate: reweight the mu_{k-1}
        batch by exp(U_{k-1}-U_k), resample, short Langevin rejuvenation on U_k
  (v)   validation set update by importance sampling, gated by
        ESS(w) >= VALIDATION_TAU (abort the stage and shrink t_k on failure).

zflows interfaces used directly: linear_combination (bridge potentials),
sequential_monte_carlo (per-rung ESS list), langevin, lbfgs,
resample, compute_ESS_log.
"""
import time

import torch
from .potential import Potential, linear_combination
from .loss import loss_compile
from .utils import (langevin, lbfgs, resample, compute_ESS_log,
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
# X-regularized stage losses -- the compiled training path (the only loss used).
# ---------------------------------------------------------------------------


def fused_stage_loss(yp, u0: Potential, u: Potential, tp, tn, G, lam, split):
    """forward KL + lam*X_mu + X_mix on one packed batch -- all UNIFORM weights.

    yp = cat([x_mu, x_mix]): x_mu = yp[:split] (mu_k AIS surrogate batch), x_mix =
    yp[split:] (PRE-SHUFFLED equal mixture of hat and detached bar halves -- shuffling
    outside the graph makes the cyclic roll(1) pairing a random permutation sigma).
    The mixture X term is the plain unweighted pairwise mean |z - z[roll(1)]|. Stage bridges
    are built in-graph from tp = t_{k-1}, tn = t_k (mutated by .fill_()).
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
#       Langevin temper -> wrap. (The melt scatters mu_0 samples with a Gaussian
#       in the Cartesian formulation; on the torus the maximum-entropy melt IS a
#       fresh uniform draw -- the same role, no scatter needed.)
# ---------------------------------------------------------------------------
def quench_and_temper_torus(target: Potential, n: int, d: int, lim: float,
                            opt_step: float, opt_iters: int,
                            mc_step: float, mc_iters: int,
                            device, taming: float = 0.0) -> torch.Tensor:
    x = (torch.rand(n, d, device=device) * 2.0 - 1.0) * lim       # melt
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)  # quench
    x = langevin(x, target, step=mc_step, iters=mc_iters, taming=taming)  # temper
    return wrap_torus(x, lim)


# ---------------------------------------------------------------------------
# (ii) adaptive temperature selection
# ---------------------------------------------------------------------------
def adaptive_step(pool: torch.Tensor, u0: Potential, u: Potential, t_prev: float,
                  *, tau: float, shrink_factor: float, rungs: int, rung_iters: int,
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
        t_k = t_prev + shrink_factor * (t_k - t_prev)
    return t_k, ess_min, max_shrinks                              # safety exit


# ---------------------------------------------------------------------------
# (iv) train the stage flow G_k on the X-regularized forward KL
# ---------------------------------------------------------------------------
# Staged ESS gate: {training step -> minimum VALIDATION ESS}. At each checkpoint the
# held-out validation ESS of the CURRENT flow must clear the rising bar, else the stage is
# rejected and the ladder shrinks t_k (fail fast). It ALWAYS uses the validation ESS, never
# the per-step DIRECT (training-batch) ESS: the direct ESS is noisy AND lags the still-
# improving flow (its trailing mean ran ~0.15 below the validation ESS), so a direct-ESS
# bar wrongly rejected KLXX stages whose validation ESS was actually high -- adding spurious
# stages and inflating F = prod(1/ESS_k) relative to the KL baseline.
_ESS_GATE = {250: 0.05, 500: 0.10, 750: 0.25, 1000: 0.50}


def _gate_val_ess(F_inv, y_valid, u_prev, u_next, n, chunk):
    """Reliable staged-gate metric: the importance-sampling ESS of the CURRENT flow over a
    fixed n-sample slice of the held-out validation set (chunked so the inverse push fits)."""
    lws = []
    with torch.no_grad():
        for sub in y_valid[:n].split(chunk):
            yt, lj = F_inv.inv_ladj(sub)
            lws.append(u_prev.eval(sub) - u_next.eval(yt.clone()) + lj.clone())
    return compute_ESS_log(torch.cat(lws)).item()


def train_stage(flow, F_inv, closs, y_valid: torch.Tensor, u_prev: Potential,
                u_next: Potential, hat_pool, *, steps: int,
                batch: int, lr: float, mc_step: float, n_pool: int = None,
                mc_iters: int = 0, drop: float = 0.0,
                rungs: int, rung_iters: int, wrap, method: str = 'klxx',
                grad_clip: float = 1e3, taming: float = 0.0,
                warmup: int = 0, ess_gate=None, gate_snapshot: bool = False,
                release_cache: bool = False, status=print, report_every: int = 1):
    """Per gradient step (step (iv); batches per step (i) drawn at
    run time from the validation set):
      X       ~ Y_{k-1}                              (size `batch`)
      Y, ladj = F_inv.inv_ladj(X)                    ONE compiled inverse:
      logw    = U_{k-1}(X) - U_k(Y) + ladj           direct log(mu_k/nu_k),
                its ESS is the logged per-step diagnostic
      Xbar    = Y[:half]                             detached nu_bar half (free)
      X_mu    = M-rung AIS from nu_k to mu_k         (with
                G the stage flow): per rung, reweight by w^{1/M}, resample,
                Langevin on U_k; the per-rung weight at moved particles is
                recomputed with a cheap FORWARD pass. SMC and
                this AIS are the SAME algorithm (G = identity vs G = flow),
                so they share exactly the same parameters (rungs, rung_iters,
                mc_step).
      Xhat    = langevin(hat_subset, U_k)            (size half)
      loss    = closs(cat[X_mu, shuffle(cat[Xhat, Xbar])])  [compiled fused]
    `F_inv` and `closs` are captured ONCE per run: flow.zeros() and optimizer
    steps mutate parameters in place, which both compiled closures track
    without recompiling (zeros_sanity.py); inverse outputs are cloned out of
    the CUDA-graph static buffers. The VALIDATION update is NOT AIS -- it
    stays a single importance-sampling push (step (v)).
    """
    device = y_valid.device
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    Nv, half = y_valid.shape[0], batch // 2
    # hat mu is the QT wide-coverage pool: each step RESAMPLES the hat mini-batch from it and
    # refreshes with a short Langevin chain on U_k; the loss then runs with UNIFORM weights.
    if method == 'klxx' and hat_pool is not None:
        hat_x, hat_w = hat_pool                                  # (samples, normalized weights)
    else:
        hat_x, hat_w = None, None
    ess_hist, snapshots, t0, n_skip = [], [], time.perf_counter(), 0
    # `drop`: oversample each loss input by (1+drop) and discard the worst (highest-energy / singular)
    # particles so the gradient batch is clean. drop=0 -> b_os=batch, h_os=half, _clean is a no-op
    # -> behaviour is IDENTICAL to before (36d/48d, which never set drop, are unaffected).
    b_os = batch + int(round(drop * batch))
    h_os = half + int(round(drop * half))
    def _clean(x, n, pot=None):                     # keep the n LOWEST-energy samples (drop the singular tail)
        return x if x.shape[0] <= n else x[torch.topk((pot or u_next).eval(x), n, largest=False).indices]
    if gate_snapshot:                               # IDENTITY (step-0) snapshot = SMC fallback for extreme-singular stages:
        n_gate0 = n_pool if n_pool is not None else min(Nv, 48000)    # the flow ENTERS as identity (run_boltzmann flow.zeros()); if
        _, _, id_vess = validation_update(F_inv, y_valid[:n_gate0], u_prev, u_next, chunk=batch, drop=drop)  # training degrades it BELOW the bare bridge,
        snapshots.append((0, {key: v.detach().cpu().clone() for key, v in flow.state_dict().items()}, id_vess))  # this candidate wins -> the stage is a PURE SMC step
        status(f"    [train] identity snapshot (step 0, SMC fallback): validation ESS={id_vess:.3f}")
    for step in range(steps):
        if warmup > 0:                                         # early-stage stabilizer: linear lr
            for g in opt.param_groups:                         # warmup tames Adam's first-step
                g['lr'] = lr * min(1.0, (step + 1) / warmup)   # magnitude (~lr regardless of grad)
                                                               # on the steep soft-core, which else
                                                               # crashes the ESS of a near-converged
                                                               # (near-identity / late-t) stage --
                                                               # see tests/smoke_identity_target.py
        with torch.no_grad():
            G_now = flow.t()                                      # forward (cheap, eager)
            xb = y_valid[torch.randint(0, Nv, (b_os,), device=device)]   # oversample the inverse input by (1+drop)
            y, ladj = F_inv.inv_ladj(xb)
            y, ladj = y.clone(), ladj.clone()                     # out of static buffers
            logw = u_prev.eval(xb) - u_next.eval(y) + ladj        # AIS weight nu_k -> mu_k
            ess_hist.append(compute_ESS_log(logw, drop=drop).item())  # direct ESS (drops the worst `drop` frac)
            x_bar = _clean(y[:h_os], half, u_prev)                # nu_bar (hat-nu): drop singular tail by u_prev (its OWN dist -- no sharp-mode coverage bias; mu/mu_hat use u_next)
            # M-rung AIS nu_k -> mu_k, rejuvenating on U_k (runs at the oversampled b_os)
            for m in range(rungs):
                inc = logw / rungs
                y = resample(y, (inc - inc.max()).exp())
                y = wrap(langevin(y, u_next, step=mc_step, iters=rung_iters, taming=taming))
                if m < rungs - 1:                                 # forward-only refresh
                    xf, ladj_f = G_now.call_and_ladj(y)
                    logw = u_prev.eval(xf) - u_next.eval(y) - ladj_f
            x_mu = _clean(y, batch)                               # AIS mu_k, singular tail dropped -> clean batch
            if method == 'klxx':
                idx = torch.multinomial(hat_w, h_os, replacement=True)   # oversampled QT (hat-mu) draw
                x_hat = hat_x[idx]
                x_hat = wrap(langevin(x_hat, u_next, step=mc_step, iters=rung_iters, taming=taming))
                x_hat = _clean(x_hat, half)                       # hat-mu, singular tail dropped -> clean half
                perm = torch.randperm(batch, device=device)
                x_mix = torch.cat([x_hat, x_bar], dim=0)[perm]           # pre-shuffle (roll)
        if method == 'klxx':
            loss = closs(torch.cat([x_mu, x_mix], dim=0))
        else:
            loss = closs(x_mu)
        if not torch.isfinite(loss):                              # pathological batch:
            n_skip += 1                                           # skip it (params left
            status(f"    [train] non-finite loss at step {step}; "  # untouched), NEVER
                   f"step skipped (n_skip={n_skip}; gate ESS is the only standard)")
            continue                                              # no abort -- the
        opt.zero_grad(); loss.backward()                          # validation-ESS gate is
        gnorm = torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)  # the only judge
        if torch.isfinite(gnorm):                                 # spike -> clipped step
            opt.step()
        else:                                                     # inf/NaN grads: skip the
            n_skip += 1                                           # update (no abort)
            status(f"    [train] non-finite grad at step {step}; step skipped (n_skip={n_skip})")
        if (step + 1) % report_every == 0 or step == 0:
            ms = 1000.0 * (time.perf_counter() - t0) / (step + 1)
            status(f"    [train] step {step+1:>5}/{steps}  loss={loss.item():.3e}  "
                   f"direct ESS={ess_hist[-1]:.3f}  {ms:6.1f} ms/step")
        # STAGED ESS gate (rising bar) on the held-out validation ESS (N_POOL slice; reliable, unlike
        # the noisy direct ESS). DEFAULT: a miss REJECTS the stage -> train_stage returns ok=False ->
        # run_boltzmann shrinks t_k. With `gate_snapshot` (opt-in): never mid-reject -- SNAPSHOT the flow
        # (state_dict + gate val_ess) at each gate step and keep training; run_boltzmann then picks the
        # BEST-passing snapshot (robust to a late-training ESS downgrade). The fail-refresh is removed;
        # `drop` only filters bad particles inside compute_ESS_log / resample.
        bar = (ess_gate or _ESS_GATE).get(step + 1)
        if bar is not None:
            n_gate = n_pool if n_pool is not None else min(Nv, 48000)  # staged-gate slice = N_POOL (config.json)
            _, _, vess = validation_update(F_inv, y_valid[:n_gate], u_prev, u_next, chunk=batch, drop=drop)
            if gate_snapshot:                                     # opt-in: collect a snapshot, never mid-reject
                snapshots.append((step + 1, {key: v.detach().cpu().clone() for key, v in flow.state_dict().items()}, vess))
                status(f"    [train] gate step {step+1}: validation ESS={vess:.3f} "
                       f"(snapshot {len(snapshots)} saved; bar {bar:.2f})")
            elif vess < bar:                                      # DEFAULT: reject the stage on a gate miss
                status(f"    [train] staged ESS gate step {step+1}: validation ESS={vess:.3f}"
                       f" < {bar:.2f} -> REJECT stage (shrink t_k)")
                return ess_hist, False, None
            else:
                status(f"    [train] staged ESS gate step {step+1}: validation ESS={vess:.3f}"
                       f" >= {bar:.2f} OK")
            if release_cache and torch.cuda.is_available():   # high-d VRAM: free the
                torch.cuda.empty_cache()                      # validation-push pool
    return ess_hist, True, (snapshots if gate_snapshot else None)


# ---------------------------------------------------------------------------
# (v) validation set update (importance weights of the trained inverse)
# ---------------------------------------------------------------------------
def validation_update(F_inv, y_valid: torch.Tensor, u_prev: Potential,
                      u_next: Potential, *, chunk: int = 100000, drop: float = 0.0):
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
    return y_tilde, logw, compute_ESS_log(logw, drop=drop).item()


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------
def run_boltzmann(u0: Potential, u: Potential, flow_factory, *, n_valid: int,
                  n_pool: int, n_batch: int, steps: int, lr: float, lam: float,
                  mc_step: float, mc_iters: int,
                  smc_rungs: int, smc_rung_iters: int,
                  adaptive_tau: float, validation_tau: float, shrink_factor: float,
                  wrap, qt_fn, device, status=print, enlarge_factor: float = 2.0,
                  max_stages: int = 30, max_retry: int = 6,
                  t_tol: float = 0.1,
                  t_safe: float = 0.25, method: str = 'klxx',
                  grad_clip: float = 1e3, taming: float = 0.0,
                  lr_warmup: int = 0, e_min=None, e_max=None, r_min=None, r_max=None, drop: float = 0.0,
                  compile_inv: bool = True, ess_gate=None, release_cache: bool = False,
                  resume_stages=None, checkpoint_fn=None, fail_checkpoint_fn=None,
                  mode_check_fn=None, gate_snapshot: bool = False, anneal_mode: str = "geometric"):
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
    # mode='default', NOT the zflows reduce-overhead default: the NCSF circular
    # bisection inverse has a data-dependent graph break, so reduce-overhead's
    # CUDA-graph capture either OOMs or runs ~20x SLOWER than raw (measured at
    # d=19..60 on RTX 5070 Ti). mode='default' gives a clean ~3.3x speedup over
    # raw with modest memory.
    F_inv = flow.t()
    if compile_inv:
        F_inv.enable_inv_ladj(mode='default')   # compiled (small d, e.g. alkanes)
    else:
        # d too large for the autoregressive bisection inverse to torch.compile
        # (OOM at d~60 in both modes); use the RAW eager inverse via the same
        # .inv_ladj interface. Slower (~0.9 s/batch at d=60) but no OOM.
        F_inv._inv_ladj_fn = F_inv.inv.call_and_ladj
    tp_buf = torch.zeros((), device=device)
    tn_buf = torch.zeros((), device=device)
    if method == 'klxx':
        closs = loss_compile(fused_stage_loss, u0, u, tp_buf, tn_buf, flow.t(),
                             float(lam), n_batch)
    else:                                                         # 'kl' baseline
        closs = loss_compile(fused_kl_loss, u0, u, tp_buf, tn_buf, flow.t())
    # regularization anneal (design.tex Sec.3): the target soft-core cap anneals geometrically
    # e_min -> e_max along t; the flow trains at the PREVIOUS (soft) e-level U[t_{k-1}] and Monte
    # Carlo sharpens to the current one U[t_k]. e_anneal=False reproduces the fixed-cap BG.
    e_anneal = e_min is not None and e_max is not None and float(e_max) > float(e_min)
    def _e_of(t):                                           # cap schedule: "geometric" (default, exp in t) or "arithmetic" (linear in t)
        return (float(e_min) + (float(e_max) - float(e_min)) * float(t)) if anneal_mode == "arithmetic" \
            else float(e_min) * (float(e_max) / float(e_min)) ** float(t)
    # r_floor anneal (mirrors e exactly): soft LARGE floor r_max at t=0 -> sharp r_min at t=1,
    # geometric in t. A larger floor bounds the LJ clash wall, so the flow stops packing atoms into
    # r->0 overlaps in the early stages; sharpened by the SAME per-stage MC step as e. Off if unset.
    r_anneal = r_min is not None and r_max is not None and float(r_max) > float(r_min)
    def _r_of(t):                                           # r_floor schedule: matches anneal_mode (arithmetic = linear r_max->r_min)
        return (float(r_max) + (float(r_min) - float(r_max)) * float(t)) if anneal_mode == "arithmetic" \
            else float(r_max) * (float(r_min) / float(r_max)) ** float(t)
    def _set_reg(t):                                         # advance BOTH regularizations to ladder level t
        if e_anneal: u.set_regularization(_e_of(t))
        if r_anneal: u.set_r_floor(_r_of(t))
    anneal = e_anneal or r_anneal
    if resume_stages:                                        # warm-start: replay pre-trained stages to
        status(f"[resume] replaying {len(resume_stages)} saved stages to rebuild Y "  # rebuild Y, then
               f"(t={[round(s['t'], 3) for s in resume_stages]})")                     # train from the next stage
        for rs in resume_stages:                             # mirrors flow_proper_smc + the accept block (iv)-(vi)
            tk = rs['t']
            _set_reg(t_prev)                                   # validation + Langevin at the SOFT cap/floor reg(t_prev)
            u_prev, u_next = bridge(u0, u, t_prev), bridge(u0, u, tk)
            flow.load_state_dict({key: v.to(device) for key, v in rs['state_dict'].items()})
            y_tilde, logw, rep_ess = validation_update(F_inv, Y, u_prev, u_next, drop=drop)
            Y = resample(y_tilde, (logw - logw.max()).exp(), drop=drop)
            Y = wrap(langevin(Y, u_next, step=mc_step, iters=mc_iters, taming=taming))
            if anneal:                                         # MC sharpen reg(t_prev) -> reg(tk), as in (vi)
                ua = u.eval(Y); _set_reg(tk); ub = u.eval(Y)
                logw_s = -tk * (ub - ua)
                Y = resample(Y, (logw_s - logw_s.max()).exp(), drop=drop)
                Y = wrap(langevin(Y, bridge(u0, u, tk), step=mc_step, iters=mc_iters, taming=taming))
            status(f"[resume] replayed stage t={tk:.3f}: val_ess={rep_ess:.3f} "  # faithfulness check vs the
                   f"(orig {rs.get('val_ess', float('nan')):.3f}); Y advanced")     # original recorded val_ess
            stages.append(rs)                                 # keep the ORIGINAL stage record verbatim
            t_prev = tk
    while t_prev < 1.0 and len(stages) < max_stages:
        k = len(stages) + 1
        _set_reg(t_prev)                                          # flow trains on U[t_{k-1}] (soft cap + soft r_floor)
        u_prev = bridge(u0, u, t_prev)
        # (i) selection pool: draw from Y_{k-1}, rejuvenate on U_{k-1}
        pool = Y[torch.randint(0, n_valid, (n_pool,), device=device)]
        if t_prev > 0.0:
            pool = wrap(langevin(pool, u_prev, step=mc_step, iters=mc_iters, taming=taming))
        # (ii) adaptive temperature selection. Initial guess:
        # the safe start t_safe on stage 1 (the leading increment faces the
        # largest deformation; an over-aggressive start is exposed only
        # after a full training), thereafter the enlarge-factor extrapolation
        # min(1, t_{k-1} + enlarge_factor (t_{k-1} - t_{k-2})) -- step sizes may
        # grow on success (enlarge_factor, default 2), only the first few stages are hard.
        ts = [0.0] + [s['t'] for s in stages]
        t_init = min(ts[-1] + enlarge_factor * (ts[-1] - ts[-2]), 1.0) if len(ts) >= 2 else t_safe
        t_k, smc_ess, n_shrink = adaptive_step(
            pool, u0, u, t_prev, tau=adaptive_tau, shrink_factor=shrink_factor,
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
            hat = qt_fn(u_next) if method == 'klxx' else None   # (iii) QT wide-coverage on U_k
            flow.zeros()                                          # identity reset, in place
            tp_buf.fill_(t_prev)
            tn_buf.fill_(t_k)
            ess_hist, ok, snapshots = train_stage(                # (iv)
                flow, F_inv, closs, Y, u_prev, u_next, hat, steps=steps,
                batch=n_batch, lr=lr, mc_step=mc_step, n_pool=n_pool,
                mc_iters=mc_iters, drop=drop,
                rungs=smc_rungs, rung_iters=smc_rung_iters,
                wrap=wrap, method=method, grad_clip=grad_clip,
                taming=taming, warmup=lr_warmup, status=status,
                ess_gate=ess_gate, gate_snapshot=gate_snapshot, release_cache=release_cache)
            if release_cache and torch.cuda.is_available():   # defrag BEFORE the big validation push
                torch.cuda.empty_cache()
            # ACCEPTANCE: pick the flow + apply BOTH gates. With gate_snapshot, evaluate the gate
            # snapshots BEST-first (by gate val_ess) and accept the best that passes the ESS floor AND
            # the post-sharpen mode-gate; reject (shrink) only if ALL fail. Default -> the single final
            # flow. The fail-refresh is removed; drop only filters bad particles inside resample.
            if gate_snapshot and snapshots:
                cand = sorted(snapshots, key=lambda s: -s[2])     # (step, state_dict, gate_val_ess), best first
            else:
                cand = [(steps, None, None)]                       # single final flow (default behaviour)
            chosen, last_val, last_cov = None, float('nan'), True
            for c_step, c_sd, c_gate in cand:
                if c_sd is not None:                               # load this snapshot's flow into the shared flow
                    flow.load_state_dict({key: v.to(device) for key, v in c_sd.items()})
                if release_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                y_tilde, logw, val_ess = validation_update(F_inv, Y, u_prev, u_next, drop=drop)
                last_val = val_ess
                if release_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not (ok and val_ess >= validation_tau):         # candidate fails the ESS floor -> next
                    continue
                covered = True                                     # post-sharpen MODE-coverage gate on THIS candidate
                if mode_check_fn is not None:
                    Yf = resample(y_tilde, (logw - logw.max()).exp(), drop=drop)
                    Ys = Yf[torch.randint(0, Yf.shape[0], (min(Yf.shape[0], 30000),), device=Yf.device)]
                    Ys = wrap(langevin(Ys, u_next, step=mc_step, iters=mc_iters, taming=taming))  # at reg(t_{k-1})
                    if anneal:                                     # MC sharpen reg(t_{k-1}) -> reg(t_k), as in (vi)
                        ua = u.eval(Ys); _set_reg(t_k); ub = u.eval(Ys)
                        lws = -t_k * (ub - ua)
                        Ys = resample(Ys, (lws - lws.max()).exp(), drop=drop)
                        Ys = wrap(langevin(Ys, bridge(u0, u, t_k), step=mc_step, iters=mc_iters, taming=taming))
                        _set_reg(t_prev)                           # restore the soft cap for the next candidate / retry
                    covered = mode_check_fn(Ys)
                last_cov = covered
                if covered:                                        # passes ESS floor AND mode-gate -> best-passing
                    chosen = dict(t_k=t_k, flow=flow, y_tilde=y_tilde, logw=logw, u_next=u_next,
                                  val_ess=val_ess, ess_hist=ess_hist, gate_step=c_step)
                    break
            attempts.append(dict(t_k=t_k, train_ok=ok, n_cand=len(cand), mode_covered=last_cov,
                                 val_ess=(chosen['val_ess'] if chosen else last_val),
                                 accepted=chosen is not None, ess_hist=list(ess_hist)))
            if chosen is not None:
                accepted = chosen
                status(f"[stage {k}] attempt {attempt+1}: ACCEPT snapshot@step {chosen['gate_step']} "
                       f"val_ess={chosen['val_ess']:.3f} (best of {len(cand)} candidate(s), floor {validation_tau})")
                break
            status(f"[stage {k}] attempt {attempt+1}: NO candidate passed (ESS floor {validation_tau} "
                   f"+ mode-gate) among {len(cand)} snapshot(s) -> shrink t_k")
            if fail_checkpoint_fn is not None:                   # debug: overwrite the single FAILED snapshot
                fail_checkpoint_fn(k, attempt + 1, t_k, last_val, ok, flow, list(ess_hist))
            t_k = t_prev + shrink_factor * (t_k - t_prev)        # abort & shrink
            status(f"[stage {k}] abort -> shrink to t_k={t_k:.4f}")
        if accepted is None:
            status(f"[stage {k}] STAGE FAILED: validation ESS never reached "
                   f"{validation_tau} in {max_retry} attempts; stopping with an "
                   f"INCOMPLETE ladder (reached t={t_prev:.4f})")
            return stages, Y, False, flow, F_inv
        # (v) accept: resample + rejuvenate the validation set
        t_k, u_next = accepted['t_k'], accepted['u_next']
        logw = accepted['logw']
        Y = resample(accepted['y_tilde'], (logw - logw.max()).exp(), drop=drop)
        Y = wrap(langevin(Y, u_next, step=mc_step, iters=mc_iters, taming=taming))
        sharpen_ess = None
        if anneal:                                               # (vi) MC sharpening reg(t_prev) -> reg(t_k) (e and/or r_floor)
            if release_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()                         # defrag before the 2x full-Y sharpening evals
            ua = u.eval(Y)                                        # U[t_{k-1}](Y) at reg(t_prev)
            _set_reg(t_k)                                         # advance cap + r_floor (no torch retrace)
            ub = u.eval(Y)                                        # U[t_k](Y)   at reg(t_k)
            logw_s = -t_k * (ub - ua)
            sharpen_ess = compute_ESS_log(logw_s, drop=drop).item()
            Y = resample(Y, (logw_s - logw_s.max()).exp(), drop=drop)
            Y = wrap(langevin(Y, bridge(u0, u, t_k), step=mc_step, iters=mc_iters, taming=taming))
            emsg = f"e {_e_of(t_prev):.1f}->{_e_of(t_k):.1f}  " if e_anneal else ""
            rmsg = f"r {_r_of(t_prev):.3f}->{_r_of(t_k):.3f}  " if r_anneal else ""
            status(f"[stage {k}] sharpen {emsg}{rmsg}sharpen ESS={sharpen_ess:.3f}")
        stages.append(dict(
            t=t_k, smc_ess=smc_ess, n_shrink=n_shrink, sharpen_ess=sharpen_ess,
            val_ess=accepted['val_ess'], attempts=attempts,
            train_ess_hist=accepted['ess_hist'],
            state_dict={key: v.cpu().clone()
                        for key, v in accepted['flow'].state_dict().items()}))
        t_prev = t_k
        if checkpoint_fn is not None:                        # crash-safe: persist completed stages after each one
            checkpoint_fn(stages, t_prev >= 1.0)
    # NO early-stop: the ladder ALWAYS runs to t=1 explicitly. The final stage
    # trains the flow at t=1 and resamples Y at t=1 (above), so the returned
    # samples are uniform-weighted samples of the TRUE target.
    return stages, Y, (t_prev >= 1.0), flow, F_inv


# ---------------------------------------------------------------------------
# ASMC baseline: pure adaptive annealed SMC (NO flow, NO inverse, NO compile,
# NO training, NO snapshot). Per stage: adaptive_step picks t_k, the DIRECT
# bridge reweight ESS on Y decides accept/reject immediately, then resample +
# Langevin + e/r-anneal sharpen. Same standard (taus, ess_metric, schedule)
# as run_boltzmann -- the ONLY difference is there is no transport map, so it
# isolates what the annealing process alone achieves. General over molecules.
# ---------------------------------------------------------------------------
def run_asmc(u0: Potential, u: Potential, *, n_valid: int, n_pool: int,
             mc_step: float, mc_iters: int, smc_rungs: int, smc_rung_iters: int,
             adaptive_tau: float, validation_tau: float, shrink_factor: float, wrap,
             device, status=print, enlarge_factor: float = 2.0, max_stages: int = 30,
             max_retry: int = 6, t_tol: float = 0.1, t_safe: float = 0.25,
             e_min=None, e_max=None, r_min=None, r_max=None, drop: float = 0.0,
             anneal_mode: str = "geometric", resume_stages=None, checkpoint_fn=None):
    """Returns (stages, Y, complete). Each stage record: t, smc_ess, n_shrink,
    val_ess (the DIRECT bridge reweight ESS = the accept standard), sharpen_ess,
    attempts. NO state_dict (there is no flow)."""
    Y = u0.samples(n_valid).to(device)
    t_prev, stages = 0.0, []
    e_anneal = e_min is not None and e_max is not None and float(e_max) > float(e_min)
    r_anneal = r_min is not None and r_max is not None and float(r_max) > float(r_min)
    def _e_of(t):
        return (float(e_min) + (float(e_max) - float(e_min)) * float(t)) if anneal_mode == "arithmetic" \
            else float(e_min) * (float(e_max) / float(e_min)) ** float(t)
    def _r_of(t):
        return (float(r_max) + (float(r_min) - float(r_max)) * float(t)) if anneal_mode == "arithmetic" \
            else float(r_max) * (float(r_min) / float(r_max)) ** float(t)
    def _set_reg(t):
        if e_anneal: u.set_regularization(_e_of(t))
        if r_anneal: u.set_r_floor(_r_of(t))
    anneal = e_anneal or r_anneal

    def _smc_step(tk):
        """One accepted SMC move to t_k on Y: resample by the bridge reweight + Langevin, then sharpen."""
        nonlocal Y
        up, un = bridge(u0, u, t_prev), bridge(u0, u, tk)
        logw = up.eval(Y) - un.eval(Y)                            # DIRECT bridge reweight (no flow, no inverse)
        Y = resample(Y, (logw - logw.max()).exp(), drop=drop)
        Y = wrap(langevin(Y, un, step=mc_step, iters=mc_iters, taming=0.0))
        sharpen_ess = None
        if anneal:                                                # MC sharpen reg(t_prev) -> reg(tk) (e and/or r_floor)
            ua = u.eval(Y); _set_reg(tk); ub = u.eval(Y)
            logw_s = -tk * (ub - ua)
            sharpen_ess = compute_ESS_log(logw_s, drop=drop).item()
            Y = resample(Y, (logw_s - logw_s.max()).exp(), drop=drop)
            Y = wrap(langevin(Y, bridge(u0, u, tk), step=mc_step, iters=mc_iters, taming=0.0))
        return sharpen_ess

    if resume_stages:                                             # replay saved SMC stages to rebuild Y
        status(f"[resume] replaying {len(resume_stages)} saved SMC stages "
               f"(t={[round(s['t'], 3) for s in resume_stages]})")
        for rs in resume_stages:
            _set_reg(t_prev); _smc_step(rs['t']); stages.append(rs); t_prev = rs['t']
    while t_prev < 1.0 and len(stages) < max_stages:
        k = len(stages) + 1
        _set_reg(t_prev)                                          # soft cap/floor at t_prev
        pool = Y[torch.randint(0, n_valid, (n_pool,), device=device)]
        if t_prev > 0.0:
            pool = wrap(langevin(pool, bridge(u0, u, t_prev), step=mc_step, iters=mc_iters, taming=0.0))
        ts = [0.0] + [s['t'] for s in stages]
        t_init = min(ts[-1] + enlarge_factor * (ts[-1] - ts[-2]), 1.0) if len(ts) >= 2 else t_safe
        t_k, smc_ess, n_shrink = adaptive_step(                   # adaptive ladder (pure SMC, no flow)
            pool, u0, u, t_prev, tau=adaptive_tau, shrink_factor=shrink_factor,
            rungs=smc_rungs, rung_iters=smc_rung_iters, mc_step=mc_step,
            wrap=wrap, status=status, t_init=t_init)
        if 1.0 - t_k < t_tol:
            t_k = 1.0
        status(f"[stage {k}] t_{{k-1}}={t_prev:.4f} -> t_k={t_k:.4f} (SMC ESS_min={smc_ess:.3f}, {n_shrink} shrinks)")
        # ACCEPT on the DIRECT bridge reweight ESS, immediately (no snapshot, no flow); else shrink t_k.
        attempts, val_ess = [], None
        for attempt in range(max_retry):
            up, un = bridge(u0, u, t_prev), bridge(u0, u, t_k)
            ess = compute_ESS_log(up.eval(Y) - un.eval(Y), drop=drop).item()
            attempts.append(dict(t_k=t_k, val_ess=ess, accepted=ess >= validation_tau))
            if ess >= validation_tau:
                status(f"[stage {k}] attempt {attempt+1}: ACCEPT direct ESS={ess:.3f} (floor {validation_tau})")
                val_ess = ess; break
            status(f"[stage {k}] attempt {attempt+1}: direct ESS={ess:.3f} < {validation_tau} -> shrink t_k")
            t_k = t_prev + shrink_factor * (t_k - t_prev)
        if val_ess is None:
            status(f"[stage {k}] STAGE FAILED: direct ESS never reached {validation_tau} in "
                   f"{max_retry} attempts; stopping INCOMPLETE (reached t={t_prev:.4f})")
            return stages, Y, False
        sharpen_ess = _smc_step(t_k)                              # accept: resample + Langevin + sharpen
        emsg = f"e {_e_of(t_prev):.1f}->{_e_of(t_k):.1f}  " if e_anneal else ""
        rmsg = f"r {_r_of(t_prev):.3f}->{_r_of(t_k):.3f}  " if r_anneal else ""
        status(f"[stage {k}] sharpen {emsg}{rmsg}sharpen ESS={'%.3f' % sharpen_ess if sharpen_ess is not None else 'n/a'}")
        stages.append(dict(t=t_k, smc_ess=smc_ess, n_shrink=n_shrink, sharpen_ess=sharpen_ess,
                           val_ess=val_ess, attempts=attempts))
        t_prev = t_k
        if checkpoint_fn is not None:
            checkpoint_fn(stages, t_prev >= 1.0)
    return stages, Y, (t_prev >= 1.0)


# ---------------------------------------------------------------------------
# final evaluation: compose the stage inverses and accumulate the weights
# ---------------------------------------------------------------------------
def compose_pushforward(flow, F_inv, state_dicts, u0: Potential, u: Potential,
                        n: int, device, chunk: int = 100000):
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


# ── assembly: build the BG problem from the package modules (was pdb_potential.build) ──
import math
import numpy as np
from .potential import Uniform
from .forcefield import build_system, Amber_Force_Field, R_FLOOR, E_CAP0, E_CAP_SCALE
from .coords import Internal_Coordinates
from .potential import Source, PDB_Potential, Torsion_Target
# ──────────────────────────────────────────────────────────────────────
# domain wrap for the mixed (Euclidean bond/angle + periodic torsion) box
# ──────────────────────────────────────────────────────────────────────
def make_wrap(tor_start: int, box_be: float):
    """Domain map for the BG core: clamp Euclidean (bond/angle) dims into the
    flow box [-box_be, box_be] and wrap torsion dims into [-pi, pi]. The clamp
    keeps Langevin-rejuvenated particles inside the NCSF spline domain; the
    torsion wrap is exact and volume-preserving on the periodic block."""
    lo, hi = -box_be + 1e-4, box_be - 1e-4

    def wrap(x: torch.Tensor) -> torch.Tensor:
        eu = x[..., :tor_start].clamp(lo, hi)
        to = torch.remainder(x[..., tor_start:] + math.pi, 2.0 * math.pi) - math.pi
        return torch.cat([eu, to], dim=-1)

    return wrap


# ──────────────────────────────────────────────────────────────────────
# build() — single entry point assembling the whole ADP problem
# ──────────────────────────────────────────────────────────────────────
def build(prmtop: str = None, crd: str = None, md_frames: np.ndarray = None,
          T: float = 300.0, box_be: float = 8.0, device="cpu",
          dtype: torch.dtype = torch.float32, system=None, bonds=None,
          r_floor: float = R_FLOOR, e_cap: float = E_CAP0,
          e_cap_scale: float = E_CAP_SCALE):
    """Assemble a molecule's BG problem in whitened internal coordinates.

    Returns a dict with: ff, ic, u (target PDB_Potential), u0 (Source), the flow
    box (a, b), torsion-dim start index, the domain wrap, whitening stats, and
    sizes. Whitening (mu, sigma for bonds/angles) is estimated from md_frames.
    """
    # Molecule source: either a prmtop/crd pair (ADP) or a pre-built OpenMM
    # System + bond list (e.g. butane built in molecules.py). Both feed the same
    # validated Amber_Force_Field / BAT / Torsion_Target machinery.
    if system is None:
        struct, system = build_system(prmtop, crd)
        bonds = [(b.atom1.idx, b.atom2.idx) for b in struct.bonds]
    M = system.getNumParticles()
    ff = Amber_Force_Field(system, dtype=dtype, r_floor=r_floor).to(device)
    ic = Internal_Coordinates(bonds, M).to(device)

    # whitening stats from MD frames (bonds/angles only; torsions stay raw)
    x = torch.tensor(np.asarray(md_frames), dtype=dtype, device=device)
    with torch.no_grad():
        z, _ = ic.to_internal(x)
    nb_, na_ = M - 1, M - 2
    bd, ang = z[:, :nb_], z[:, nb_:nb_ + na_]
    mu_b, sig_b = bd.mean(0), bd.std(0).clamp_min(1e-6)
    mu_a, sig_a = ang.mean(0), ang.std(0).clamp_min(1e-6)

    # full-dimensional pieces (used for the FINE reweight against the full target)
    u = PDB_Potential(ff, ic, mu_b, sig_b, mu_a, sig_a, T=T,
                      e_cap=e_cap, e_cap_scale=e_cap_scale).to(device)
    u0 = Source(n_white=nb_ + na_, n_tor=M - 3, box_be=box_be).to(device)

    tor_start = nb_ + na_
    n_internal = 3 * M - 6
    a = torch.cat([torch.full((tor_start,), -box_be),
                   torch.full((M - 3,), -math.pi)]).to(device)
    b = torch.cat([torch.full((tor_start,), box_be),
                   torch.full((M - 3,), math.pi)]).to(device)
    wrap = make_wrap(tor_start, box_be)

    # torsion-only TORUS BG problem (the soft block the flow is trained on)
    n_tor = M - 3
    u_L = Torsion_Target(ff, ic, mu_b, mu_a, T=T,
                        e_cap=e_cap, e_cap_scale=e_cap_scale).to(device)
    u0_tor = Uniform([-math.pi] * n_tor, [math.pi] * n_tor, device=device)
    a_tor = torch.full((n_tor,), -math.pi).to(device)
    b_tor = torch.full((n_tor,), math.pi).to(device)

    return dict(ff=ff, ic=ic, u=u, u0=u0, a=a, b=b, wrap=wrap,
                tor_start=tor_start, M=M, n_internal=n_internal, T=T,
                # torus BG problem
                u_L=u_L, u0_tor=u0_tor, a_tor=a_tor, b_tor=b_tor,
                n_tor=n_tor, lim=math.pi,
                whitening=dict(mu_b=mu_b, sig_b=sig_b, mu_a=mu_a, sig_a=sig_a))
