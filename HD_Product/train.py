# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""High-dimensional product multi-well sweep. One-step IS only (no AIS ladder).
Trains the 4 X-functional losses for each k in K_LIST (highest d first) and saves
data_k{K}.pth. Progress (percent + status + timing) is written to train_status.log;
no tqdm. See vd_plan.md."""

import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import (
    importance_weights, compute_ESS, compute_ESS_log, resample, langevin,
    suppress_warnings, set_cache_size_limit,
)

from core import MultiWell, loss_KL, loss_X, quench_and_temper, coverage, mode_coverage
import parameters as P

suppress_warnings()
set_cache_size_limit(64)

HERE = Path(__file__).resolve().parent
device = 'cuda' if torch.cuda.is_available() else 'cpu'

STATUS = HERE / 'train_status.log'
SWEEP  = HERE / 'sweep_status.log'


def log(path, msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(path, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


def eval_flow(flow, u0, u1, k, n_valid, chunk=50000):
    """Final ESS on a fresh N_VALID source pool (chunked) + pushforward samples."""
    d = P.dim(k)
    Gi = flow.t().inv
    ys, logws = [], []
    with torch.no_grad():
        done = 0
        while done < n_valid:
            b = min(chunk, n_valid - done)
            x = u0.samples(b)
            y, ladj = Gi.call_and_ladj(x)
            logw = -u1(y) + u0(x) + ladj
            ys.append(y.cpu()); logws.append(logw)
            done += b
    logw = torch.cat(logws)
    ess = compute_ESS_log(logw).item()
    y = torch.cat(ys)
    return ess, y


def train_method(method, k, u0, u1, x_pool, hat_pool, steps, t_budget):
    d = P.dim(k)
    terms = set(method.split('+'))
    use_x_mu     = 'X_mu' in terms
    use_x_hat_mu = 'X_hat_mu' in terms
    use_x_mix    = 'X_mix' in terms
    needs_hat    = use_x_hat_mu or use_x_mix

    torch.manual_seed(0)
    flow = NSF(a=[-P.NSF_LIM] * d, b=[P.NSF_LIM] * d,
               bins=P.bins(k), transforms=P.transforms(k),
               hidden_features=P.hidden(k)).to(device)
    flow.zeros()
    optimizer = torch.optim.Adam(flow.parameters(), lr=P.LR)

    N = x_pool.shape[0]
    Phat = hat_pool.shape[0] if hat_pool is not None else 0
    half = P.BATCH // 2
    ess_history = []
    report_every = max(1, steps // 100)
    t0 = time.perf_counter()
    aborted = None

    for step in range(steps):
        idx = torch.randperm(N, device=device)[:P.BATCH]
        x = x_pool[idx]

        with torch.no_grad():
            G_now = flow.t()
            # single autoregressive inverse: y = G^{-1}(x) AND its ladj in one pass,
            # then derive the IS weights from that ladj (avoids a 2nd inverse/step).
            y, ladj = G_now.inv.call_and_ladj(x)             # bar_nu pushforward
            logw = -u1(y) + u0(x) + ladj                     # log mu/nu at y
            ess = compute_ESS_log(logw).item()
            ess_history.append(ess)
            w = (logw - logw.max()).exp()

            y_mu = resample(y, w)                            # one-step IS surrogate for mu
            y_mu = langevin(y_mu, u1, step=P.IS_MC_STEP, iters=P.IS_MC_ITERS)

            if needs_hat:
                hidx = torch.randint(0, Phat, (P.BATCH,), device=device)
                y_hat = hat_pool[hidx]
                y_hat = langevin(y_hat, u1, step=P.IS_MC_STEP, iters=P.IS_MC_ITERS)

        G = flow.t()
        loss = loss_KL(y_mu, u0, u1, G)
        if use_x_mu:
            loss = loss + P.LAMBDA * loss_X(y_mu, u0, u1, G)
        if use_x_hat_mu:
            loss = loss + loss_X(y_hat, u0, u1, G)
        if use_x_mix:
            y_mix = torch.cat([y_hat[:half], y[:half]], dim=0)
            loss = loss + loss_X(y_mix, u0, u1, G)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not torch.isfinite(loss):
            aborted = f"non-finite loss at step {step}"
            break

        if (step + 1) % report_every == 0 or step == 0:
            sync()
            elapsed = time.perf_counter() - t0
            ms = 1000.0 * elapsed / (step + 1)
            eta = ms * (steps - step - 1) / 1000.0
            pct = 100.0 * (step + 1) / steps
            log(STATUS, f"k={k} d={d}  {method:<18} step {step+1:>5}/{steps}  "
                        f"{pct:5.1f}%  loss={loss.item():.3e}  ess={ess:.3f}  "
                        f"elapsed={elapsed:6.1f}s  {ms:5.2f}ms/step  eta={eta:5.0f}s")
            # kill-early: projected wall-clock blows the budget
            if t_budget is not None and ms * steps / 1000.0 > 2.0 * t_budget:
                aborted = (f"projected {ms*steps/1000.0:.0f}s > 2x budget "
                           f"{t_budget:.0f}s at step {step}")
                break

    sync()
    wall = time.perf_counter() - t0
    if aborted:
        log(STATUS, f"k={k} d={d}  {method:<18} ABORTED: {aborted}")
    return flow, ess_history, wall, aborted


def run_k(k, steps, t_budget):
    d = P.dim(k)
    log(SWEEP, f"=== k={k} d={d} START  (steps={steps}) ===")
    data_path = HERE / f"data_k{k}.pth"
    if data_path.exists():
        log(SWEEP, f"k={k}: {data_path.name} exists, skipping")
        return

    u0 = Gaussian(mean=[0.0] * d, variance=[P.SIGMA ** 2] * d).to(device)
    u1 = MultiWell(k).to(device)
    u1.enable_grad(); u1.enable_eval()

    torch.manual_seed(0)
    x_pool = u0.samples(P.n_train(k))

    # QT pool for hat_mu (built once)
    torch.manual_seed(1)
    t_qt = time.perf_counter()
    x_qt = u0.samples(P.p_qt(k))
    hat_pool = quench_and_temper(x_qt, u1, sigma=P.QT_SIGMA,
                                 opt_step=P.QT_OPT_STEP, opt_iters=P.QT_OPT_ITERS,
                                 mc_step=P.QT_MC_STEP, mc_iters=P.QT_MC_ITERS)
    sync()
    cov_qt, _ = mode_coverage(hat_pool, k)
    log(SWEEP, f"k={k}: QT pool {hat_pool.shape[0]} samples in "
               f"{time.perf_counter()-t_qt:.1f}s, mode_cov={cov_qt:.3f}")

    # ESS estimate needs far fewer than N_VALID samples; the autoregressive
    # inverse is compute-bound, so cap the eval pool (20k -> ESS stable to ~1e-2).
    eval_n = min(P.n_valid(k), 20000)

    runs = {}
    for m in P.METHODS:
        flow, ess_hist, wall, aborted = train_method(
            m, k, u0, u1, x_pool, hat_pool, steps, t_budget)
        ess, y = eval_flow(flow, u0, u1, k, eval_n)
        mcov, mbal = mode_coverage(y.to(device), k)
        kcov = coverage(y[:20000].to(device), hat_pool[:min(2000, hat_pool.shape[0])], k=5)
        runs[m] = {
            'ess_history': ess_hist,
            'final_ess': ess,
            'mode_coverage': mcov,
            'mode_balance': mbal,
            'knn_coverage': kcov,
            'wall_s': wall,
            'aborted': aborted,
            'samples': y[:20000].clone(),
        }
        log(SWEEP, f"k={k} d={d}  {m:<18} final_ess={ess:.4f}  mode_cov={mcov:.3f}  "
                   f"knn_cov={kcov:.3f}  wall={wall:.1f}s"
                   + ("  [ABORTED]" if aborted else ""))

    torch.save({
        'k': k, 'd': d, 'steps': steps,
        'config': {kk: getattr(P, kk) for kk in
                   ('SIGMA', 'NSF_LIM', 'BATCH', 'LR', 'LAMBDA', 'ALPHA', 'BETA',
                    'IS_MC_STEP', 'IS_MC_ITERS', 'QT_SIGMA')},
        'arch': {'bins': P.bins(k), 'transforms': P.transforms(k), 'hidden': P.hidden(k)},
        'qt_mode_coverage': cov_qt,
        'runs': runs,
    }, data_path)
    log(SWEEP, f"=== k={k} d={d} DONE -> {data_path.name} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--klist', type=str, default=None,
                    help='comma-separated k values (default parameters.K_LIST)')
    ap.add_argument('--steps', type=int, default=None,
                    help='override STEPS')
    ap.add_argument('--budget', type=float, default=None,
                    help='per-method wall-clock soft budget in seconds (kill-early)')
    args = ap.parse_args()

    klist = ([int(s) for s in args.klist.split(',')] if args.klist
             else list(P.K_LIST))

    log(SWEEP, f"##### SWEEP START klist={klist} device={device} #####")
    t_all = time.perf_counter()
    for k in klist:
        steps = args.steps if args.steps is not None else P.steps(k)
        run_k(k, steps, args.budget)
    log(SWEEP, f"##### SWEEP DONE in {time.perf_counter()-t_all:.1f}s #####")


if __name__ == '__main__':
    main()
