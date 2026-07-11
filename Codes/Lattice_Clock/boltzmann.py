"""Monitored twins of the jflows Boltzmann drivers for the clock benchmark.

`boltzmann_forward_KL_G` (adaptive ladder) and
`boltzmann_forward_KLXX_G_fixed` (fixed t_list) are copies of their
`jflows.boltzmann` namesakes — identical signatures, ladder machinery,
identity check, records, and PRNG streams — plus per-stage sector-occupancy
monitoring. After each (accepted) stage the driver prints, over the full
particle set:

    occ(push)    sector occupancy of the pushforward through the stage map
    E[w|s]       mean normalized stage weight per sector (fair = 1.000)
    occ(weight)  weighted occupancy — what the importance correction wants
    occ(final)   occupancy after resample + Langevin rejuvenation

each with the occupancy bias err = (1/P) sum_s |p_s - 1/P|. This makes any
sector-selective weight distortion (e.g. the theta = pi seam sector of the
clock model) visible live in the training log, stage by stage.

Import from this folder (train.py):
    from boltzmann import boltzmann_forward_KL_G, boltzmann_forward_KLXX_G_fixed
"""

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jflows.boltzmann import (
    _BG_DEFAULTS,
    _iw_log_identity,
    _iw_log_jit,
    _langevin_jit,
    _smc_jit,
)
from jflows.potential import Potential, linear_combination
from jflows.train import Monitor, train_forward_KL_G, train_forward_KLXX_G
from jflows.utils import compute_ESS_log, langevin, resample

sys.path.insert(0, str(Path(__file__).resolve().parent))
from potential import magnetization

P = 6   # clock sectors (matches potential.Clock in this folder)


# ── sector-occupancy monitoring ──────────────────────────────────────

def _sectors(y: np.ndarray) -> np.ndarray:
    m = magnetization(y)
    return np.round(np.angle(m) * P / (2.0 * math.pi)).astype(np.int64) % P


def _occ(y: np.ndarray) -> np.ndarray:
    s = _sectors(y)
    return np.bincount(s, minlength=P).astype(np.float64) / s.shape[0]


def _bias(p: np.ndarray) -> float:
    return float(np.abs(p - 1.0 / P).sum() / P)


def _fmt(p: np.ndarray) -> str:
    return "[" + " ".join(f"{q:.3f}" for q in p) + "]"


def _occ_report(status, k: int, y_push, log_w, y_final) -> None:
    """Per-stage occupancy monitor: pushforward / per-sector weights /
    weighted / final occupancy of the full particle set."""
    yp = np.asarray(y_push)
    s = _sectors(yp)
    w = np.asarray(log_w, dtype=np.float64)
    w = np.exp(w - w.max())
    wn = w / w.mean()
    occ_p = np.bincount(s, minlength=P).astype(np.float64) / s.shape[0]
    ew = np.array([wn[s == j].mean() if (s == j).any() else np.nan
                   for j in range(P)])
    occ_w = np.array([w[s == j].sum() for j in range(P)]) / w.sum()
    occ_f = _occ(np.asarray(y_final))
    status(f"[stage {k}] occ(push)   = {_fmt(occ_p)}  bias={_bias(occ_p):.4f}")
    status(f"[stage {k}] E[w|s]      = {_fmt(ew)}  (fair 1.000)")
    status(f"[stage {k}] occ(weight) = {_fmt(occ_w)}  bias={_bias(occ_w):.4f}")
    status(f"[stage {k}] occ(final)  = {_fmt(occ_f)}  bias={_bias(occ_f):.4f}")


def _bg_advance_mon(key_res, key_mc, y, log_w, flow, u_k,
                    mc_step, mc_iters, mc_adjust, chunk):
    """Stage advance (push -> reweight/resample -> Langevin), as
    `jflows.boltzmann._bg_advance` with type="G", but also returns the
    pushforward so the occupancy monitor reuses it at no extra cost."""
    y_push = jnp.concatenate(
        [flow.inv(c) for c in jnp.array_split(y, chunk, axis=0)], axis=0
    )
    y_new = resample(key_res, y_push, jnp.exp(log_w - log_w.max()))
    y_new = langevin(key_mc, y_new, u_k, step=mc_step, iters=mc_iters,
                     adjust=mc_adjust, chunk=chunk)
    return y_new, y_push


# ── monitored drivers (copies of jflows.boltzmann + _occ_report) ─────

def boltzmann_forward_KL_G(
    x_valid,
    source: Potential,
    target: Potential,
    flow,
    n_pool: int,
    n_batch: int,
    steps: int,
    lr: float,
    ladder: int,
    mc_step: float,
    mc_iters: int,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunk: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
):
    """`jflows.boltzmann.boltzmann_forward_KL_G` + per-stage occupancy
    monitoring (see the module docstring). Identical ladder, identity
    check, records, and PRNG streams."""
    p = dict(_BG_DEFAULTS)
    if bg_param:
        unknown = set(bg_param) - set(p)
        if unknown:
            raise ValueError(f"boltzmann_forward_KL_G: unknown bg_param keys {sorted(unknown)}")
        p.update(bg_param)
    if not (0.0 < p["shrink_factor"] < 1.0 and 0.0 < p["t_safe"] <= 1.0 and p["max_retry"] >= 1):
        raise ValueError(f"boltzmann_forward_KL_G: invalid bg_param {p!r}")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(4)  # this ladder's own base stream

    y_valid = x_valid                  # validation/particle set at t = 0 (= mu_0)
    identity_flow = flow.zeros()       # identity map — the SMC fallback, shared by all stages
    stages: list[dict] = []
    t_prev = 0.0
    while t_prev < 1.0 and len(stages) < p["max_stages"]:
        k = len(stages) + 1
        hist = [0.0] + [s["t"] for s in stages]
        t_k = p["t_safe"] if not stages else min(
            hist[-1] + p["enlarge_factor"] * (hist[-1] - hist[-2]), 1.0
        )
        if 1.0 - t_k < p["t_tol"]:
            t_k = 1.0
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        # (1) SMC pre-selection of t_k on an n_pool-sized selection pool
        if p["tau_smc"] > 0.0:
            key_pool, key_mc_pool = jax.random.split(jax.random.fold_in(key, 20_000 + k))
            pool = y_valid[jax.random.randint(key_pool, (n_pool,), 0, y_valid.shape[0])]
            if t_prev > 0.0:
                pool = _langevin_jit(key_mc_pool, pool, u_prev, step=mc_step,
                                     iters=mc_iters, adjust=mc_adjust, chunk=chunk)
            smc_base = jax.random.fold_in(key, 10_000 + k)  # disjoint from the advance keys
            for s_i in range(60):                           # max_shrinks, as in the reference
                u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
                _, smc_ess = _smc_jit(jax.random.fold_in(smc_base, s_i), pool,
                                      u_prev, u_k, ladder=ladder, step=mc_step,
                                      iters=mc_iters, adjust=mc_adjust, chunk=chunk)
                ess_smc = float(smc_ess.min())
                ok_smc = ess_smc >= p["tau_smc"]
                status(f"[stage {k}] [select] t_k={t_k:.4f}  SMC ESS = {ess_smc:.3f} "
                       f"({'accept' if ok_smc else 'shrink'})")
                if ok_smc:
                    break
                t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
        accepted = False
        for attempt in range(1, p["max_retry"] + 1):
            u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
            status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} "
                   f"(attempt {attempt}/{p['max_retry']}) training the increment ...")
            seed = jnp.uint32(k * p["max_retry"] + attempt)  # fresh stream per attempt
            cand, ess_hist = train_forward_KL_G(y_valid, u_prev, u_k, flow,
                                                n_batch, steps, lr, ladder, mc_step,
                                                mc_iters, mc_adjust, monitor,
                                                seed=seed, checkpoint=checkpoint,
                                                e_clip=e_clip, g_clip=g_clip)
            log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunk=chunk)
            ess_tr = float(compute_ESS_log(log_w_tr))
            log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunk=chunk)
            ess_id = float(compute_ESS_log(log_w_id))
            jax.effects_barrier()  # keep monitor lines ahead of the stage status
            # identity check: keep the better of the trained flow and the
            # identity map (pure SMC), so a stage is never worse than SMC
            if ess_tr >= ess_id:
                cand_flow, log_w, ess_k = cand, log_w_tr, ess_tr
            else:
                cand_flow, log_w, ess_k = identity_flow, log_w_id, ess_id
            imp = ess_k - ess_id  # improvement over identity, always >= 0
            status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
                   f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f}; "
                   f"tau_ess = {p['tau_ess']:.2f})")
            if ess_k >= p["tau_ess"]:
                accepted = True
                break
            status(f"[stage {k}] t={t_k:.4f} REJECTED -> shrink")
            t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
        if not accepted:
            status(f"[stage {k}] gave up after {p['max_retry']} attempts "
                   f"(last t={t_k:.4f}, ESS = {ess_k:.3f}) — ladder INCOMPLETE")
            break
        flow = cand_flow               # warm start (identity if the fallback won)
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_new, y_push = _bg_advance_mon(key_res, key_mc, y_valid, log_w, flow,
                                        u_k, mc_step, mc_iters, mc_adjust, chunk)
        y_new = jax.block_until_ready(y_new)  # errors surface at THIS stage, not the next
        _occ_report(status, k, y_push, log_w, y_new)
        y_valid = y_new
        stages.append({"t": t_k, "ess": ess_k, "flow": flow,
                       "ess_history": ess_hist, "imp_history": imp})
        status(f"[stage {k}] t={t_k:.4f} ACCEPTED (attempt {attempt}): stage flow saved")
        t_prev = t_k
    if t_prev < 1.0:
        status(f"boltzmann_forward_KL_G: ladder INCOMPLETE at t = {t_prev:.4f} "
               f"({len(stages)} accepted stages); particle set at the last accepted bridge")
    else:
        status(f"boltzmann_forward_KL_G: ladder COMPLETE ({len(stages)} stages); "
               f"particle set at the target")
    return y_valid, stages


def boltzmann_forward_KLXX_G_fixed(
    x_valid,
    source: Potential,
    target: Potential,
    flow,
    n_pool: int,
    n_batch: int,
    steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_step: float,
    opt_iters: int,
    mc_step: float,
    mc_iters: int,
    t_list,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunk: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
):
    """`jflows.boltzmann.boltzmann_forward_KLXX_G_fixed` + per-stage
    occupancy monitoring (see the module docstring). Identical schedule,
    identity check, records, and PRNG streams."""
    t_list = [float(t) for t in t_list]
    if t_list and t_list[0] == 0.0:
        t_list = t_list[1:]
    if not t_list:
        raise ValueError("boltzmann_forward_KLXX_G_fixed: t_list is empty")
    if any(a >= b for a, b in zip(t_list, t_list[1:])):
        raise ValueError("boltzmann_forward_KLXX_G_fixed: t_list must be strictly increasing")
    if not (0.0 < t_list[0] and t_list[-1] <= 1.0):
        raise ValueError("boltzmann_forward_KLXX_G_fixed: t_list must lie in (0, 1]")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(8)  # same base stream as boltzmann_forward_KLXX_G
    if t_list[-1] != 1.0:
        status(f"boltzmann_forward_KLXX_G_fixed: t_list ends at {t_list[-1]:.4f} < 1 "
               f"— incomplete fixed ladder; particle set stops at the last bridge")

    y_valid = x_valid
    identity_flow = flow.zeros()
    stages: list[dict] = []
    t_prev = 0.0
    for k, t_k in enumerate(t_list, start=1):
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
        status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} training the increment (fixed) ...")
        seed = jnp.uint32(k)
        cand, ess_hist = train_forward_KLXX_G(y_valid, u_prev, u_k, flow,
                                              n_pool, n_batch, steps, lr, ladder,
                                              melt, opt_step, opt_iters, mc_step,
                                              mc_iters, coeff_lambda, coeff_alpha,
                                              coeff_beta, mc_adjust, monitor,
                                              seed=seed, checkpoint=checkpoint,
                                              e_clip=e_clip, g_clip=g_clip)
        log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunk=chunk)
        ess_tr = float(compute_ESS_log(log_w_tr))
        log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunk=chunk)
        ess_id = float(compute_ESS_log(log_w_id))
        jax.effects_barrier()  # keep monitor lines ahead of the stage status
        # identity check: keep the better of the trained flow and the
        # identity map (pure SMC), so a stage is never worse than SMC
        if ess_tr >= ess_id:
            cand_flow, log_w, ess_k = cand, log_w_tr, ess_tr
        else:
            cand_flow, log_w, ess_k = identity_flow, log_w_id, ess_id
        imp = ess_k - ess_id  # improvement over identity, always >= 0
        status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
               f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f})")
        flow = cand_flow               # warm start (identity if the fallback won)
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_new, y_push = _bg_advance_mon(key_res, key_mc, y_valid, log_w, flow,
                                        u_k, mc_step, mc_iters, mc_adjust, chunk)
        y_new = jax.block_until_ready(y_new)  # errors surface at THIS stage
        _occ_report(status, k, y_push, log_w, y_new)
        y_valid = y_new
        stages.append({"t": t_k, "ess": ess_k, "flow": flow,
                       "ess_history": ess_hist, "imp_history": imp})
        status(f"[stage {k}] t={t_k:.4f} DONE: stage flow saved; particle set advanced")
        t_prev = t_k
    status(f"boltzmann_forward_KLXX_G_fixed: fixed ladder DONE "
           f"({len(stages)} stages, {'COMPLETE' if t_list[-1] == 1.0 else 'INCOMPLETE'})")
    return y_valid, stages
