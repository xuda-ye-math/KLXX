# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""Sensor-array source-localization Bayesian inverse problem. Trains the 4
X-functional losses (KL, KL+X_mu, KL+X_mu+X_hat_mu, KL+X_mu+X_mix) with one-step
importance sampling (M=1) on a permutation-symmetric n!-mode posterior, and
evaluates all four methods on the SAME target for an honest ESS comparison.
Progress is written to train_status.log (no tqdm). Saves data.pth.

Usage:
  python train.py             # full run (parameters.STEPS)
  python train.py --steps 200 # sanity run
"""
import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
torch.set_num_threads(32)
from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import (
    importance_weights, compute_ESS_log, resample, langevin,
    suppress_warnings, set_cache_size_limit,
)

import core
from core import (
    SensorArrayPosterior, loss_KL, loss_X, quench_and_temper,
    coverage, mode_coverage_nearest,
)
import parameters as P

suppress_warnings()
set_cache_size_limit(64)

HERE = Path(__file__).resolve().parent
device = 'cuda' if torch.cuda.is_available() else 'cpu'

STATUS = HERE / 'train_status.log'
NMODES = 1
for _i in range(2, P.N_SRC + 1):
    NMODES *= _i


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(STATUS, 'a') as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


def build_targets():
    """Returns (u0 source, u1 target, centers, theta_star)."""
    theta_star = core.true_theta(P.THETA_STAR)
    sensors = core.sensor_positions(P.N_SENSORS, P.SENSOR_LIM)
    data = core.make_data(P.THETA_STAR, sensors, P.ELL, P.SIGMA_OBS, seed=P.DATA_SEED)

    u0 = Gaussian(mean=[0.0] * P.N_SRC, variance=[P.SIGMA_PRIOR ** 2] * P.N_SRC).to(device)
    u1 = SensorArrayPosterior(sensors, data, P.ELL, P.SIGMA_OBS, P.SIGMA_PRIOR).to(device)
    u1.enable_grad(); u1.enable_eval()

    centers = core.mode_centers(P.THETA_STAR).to(device)
    return u0, u1, centers, theta_star


def new_flow():
    torch.manual_seed(0)
    flow = NSF(a=[-P.NSF_LIM] * P.N_SRC, b=[P.NSF_LIM] * P.N_SRC,
               bins=P.BINS, transforms=P.TRANSFORMS,
               hidden_features=P.HIDDEN).to(device)
    flow.zeros()
    return flow


def train_method(method, u0, u1, x_pool, hat_pool, steps):
    terms = set(method.split('+'))
    use_x_mu     = 'X_mu' in terms
    use_x_hat_mu = 'X_hat_mu' in terms
    use_x_mix    = 'X_mix' in terms
    needs_hat    = use_x_hat_mu or use_x_mix

    flow = new_flow()
    optimizer = torch.optim.Adam(flow.parameters(), lr=P.LR)

    N = x_pool.shape[0]
    Phat = hat_pool.shape[0] if hat_pool is not None else 0
    half = P.BATCH // 2
    ess_history = []
    report_every = max(1, steps // 100)
    t0 = time.perf_counter()
    y_hat = None
    aborted = None

    for step in range(steps):
        idx = torch.randperm(N, device=device)[:P.BATCH]
        x = x_pool[idx]

        with torch.no_grad():
            G_now = flow.t()
            # y = G^{-1}(x) (the pushforward bar_nu) AND ladj in one inverse pass;
            # IS weights derived from that ladj.
            y, ladj = G_now.inv.call_and_ladj(x)
            logw = -u1(y) + u0(x) + ladj
            ess = compute_ESS_log(logw).item()
            ess_history.append(ess)
            w = (logw - logw.max()).exp()

            # one-step IS surrogate for mu: resample then short Langevin.
            y_mu = resample(y, w)
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
            log(f"{method:<18} ABORTED: {aborted}")
            break

        if (step + 1) % report_every == 0 or step == 0:
            sync()
            elapsed = time.perf_counter() - t0
            pct = 100.0 * (step + 1) / steps
            log(f"{method:<18} step {step+1:>5}/{steps}  {pct:5.1f}%  "
                f"loss={loss.item():.3e}  ess={ess:.3f}  elapsed={elapsed:6.1f}s")

    return flow, ess_history, aborted


def eval_flow(flow, u0, u1, n_valid, chunk=20000):
    """Honest eval: ESS on the target U with a fresh source pool, no SNIS tricks.
    Returns (ess, samples)."""
    Gi = flow.t().inv
    ys, logws = [], []
    with torch.no_grad():
        done = 0
        while done < n_valid:
            b = min(chunk, n_valid - done)
            x = u0.samples(b)
            y, ladj = Gi.call_and_ladj(x)
            logw = -u1(y) + u0(x) + ladj
            ys.append(y); logws.append(logw)
            done += b
    ess = compute_ESS_log(torch.cat(logws)).item()
    y = torch.cat(ys)
    return ess, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    steps = args.steps if args.steps is not None else P.STEPS

    data_path = HERE / (f"data_{args.tag}.pth" if args.tag else "data.pth")

    log(f"##### RUN START steps={steps} device={device} tag={args.tag or '(full)'} #####")
    t_all = time.perf_counter()

    u0, u1, centers, theta_star = build_targets()
    log(f"targets built: n={P.N_SRC} nmodes={NMODES}  ||theta*||={theta_star.norm():.3f}  "
        f"U(modes)={[round(v,3) for v in u1(centers).tolist()]}")

    # source pool (shared across all runs)
    torch.manual_seed(0)
    x_pool = u0.samples(P.N_TRAIN)

    # QT pool for hat_mu, built ONCE.
    torch.manual_seed(1)
    t_qt = time.perf_counter()
    x_qt = u0.samples(P.P_QT)
    hat_pool = quench_and_temper(x_qt, u1, sigma=P.QT_SIGMA,
                                 opt_step=P.QT_OPT_STEP, opt_iters=P.QT_OPT_ITERS,
                                 mc_step=P.QT_MC_STEP, mc_iters=P.QT_MC_ITERS)
    assert torch.isfinite(hat_pool).all(), "QT produced non-finite output (LBFGS step too large?)"
    sync()
    qt_cov, qt_occ, qt_counts = mode_coverage_nearest(hat_pool, centers, frac=P.MODE_FRAC)
    log(f"QT pool {hat_pool.shape[0]} samples in {time.perf_counter()-t_qt:.1f}s  "
        f"modes_found={qt_cov*NMODES:.0f}/{NMODES}  occ={[round(v,3) for v in qt_occ.tolist()]}")

    runs = {}
    for m in P.METHODS:
        log(f"=== train {m} ===")
        flow, ess_hist, aborted = train_method(m, u0, u1, x_pool, hat_pool, steps)

        torch.manual_seed(123)
        ess, y = eval_flow(flow, u0, u1, P.N_VALID)
        mcov, occ, counts = mode_coverage_nearest(y, centers, frac=P.MODE_FRAC)
        kcov = coverage(y[:20000], hat_pool[:min(2000, hat_pool.shape[0])], k=P.KNN_K)

        runs[m] = {
            'ess_history': ess_hist,
            'final_ess': ess,
            'mode_coverage': mcov,
            'modes_found': int(round(mcov * NMODES)),
            'occupancy': occ,
            'counts': counts,
            'knn_coverage': kcov,
            'aborted': aborted,
            'samples': y[:20000].cpu().clone(),
            'state_dict': {k: v.cpu().clone() for k, v in flow.state_dict().items()},
        }
        log(f"{m:<18} ess={ess:.4f}  modes_found={runs[m]['modes_found']}/{NMODES}  "
            f"knn_cov={kcov:.3f}  occ={[round(v,3) for v in occ.tolist()]}"
            + ("  [ABORTED]" if aborted else ""))

    torch.save({
        'steps': steps,
        'n_src': P.N_SRC, 'nmodes': NMODES,
        'theta_star': theta_star.cpu(),
        'centers': centers.cpu(),
        'config': {kk: getattr(P, kk) for kk in
                   ('N_SRC', 'N_SENSORS', 'SENSOR_LIM', 'ELL', 'SIGMA_OBS', 'SIGMA_PRIOR',
                    'THETA_STAR', 'NSF_LIM', 'BINS', 'TRANSFORMS', 'BATCH', 'LR', 'LAMBDA',
                    'IS_MC_STEP', 'IS_MC_ITERS', 'QT_SIGMA', 'QT_OPT_STEP', 'MODE_FRAC',
                    'KNN_K', 'STEPS')},
        'qt_mode_coverage': qt_cov,
        'qt_occupancy': qt_occ,
        'runs': runs,
    }, data_path)
    log(f"##### RUN DONE in {time.perf_counter()-t_all:.1f}s -> {data_path.name} #####")


if __name__ == '__main__':
    main()
