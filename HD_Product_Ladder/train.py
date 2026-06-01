# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""AIS-ladder study on the product multi-well at d = 2**K, hidden HIDDEN.

CONSISTENT WITH HD_Product: pure eager (no torch.compile), exactly ONE
autoregressive inverse per step -- y = G^{-1}(x) is computed once and reused for
the ESS diagnostic, the mu-surrogate, and the X_mix bar_nu half, identical to
HD_Product/train.py. The ONLY change vs HD_Product is the mu-surrogate: instead
of the single reweight+resample+Langevin hop (M=1), we run an M-rung annealed
ladder that REUSES that one inverse -- each extra rung recomputes the IS weight
with a cheap FORWARD pass G(y) (no second inverse) before resample+Langevin. At
M=1 this reduces exactly to HD_Product's one-step surrogate.

One flow, one ladder length M per run; the M sweep is separate invocations:
  for M in 1 2 4 8 16 32; do python train.py --M $M; done

Usage:
  python train.py --M 8
  python train.py --M 8 --steps 50    # quick sanity
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import compute_ESS_log, resample, langevin, suppress_warnings

import core
from core import MultiWell, loss_KL, loss_X, quench_and_temper, coverage, mode_coverage
import parameters as P

suppress_warnings()

HERE = Path(__file__).resolve().parent
device = 'cuda' if torch.cuda.is_available() else 'cpu'
STATUS = HERE / 'train_status.log'
SWEEP = HERE / 'sweep_status.log'


def log(path, msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(path, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


def new_flow(d):
    torch.manual_seed(0)
    flow = NSF(a=[-P.NSF_LIM] * d, b=[P.NSF_LIM] * d,
               bins=P.BINS, transforms=P.TRANSFORMS,
               hidden_features=P.HIDDEN).to(device)
    flow.zeros()
    return flow


def ais_ladder(y, logw, x, u0, u1, G, M):
    """M-rung annealed surrogate for mu that REUSES the single inverse.
    Input y ~ nu (= G^{-1}(x)) and logw = log w(y) already computed from that
    inverse pass. Each rung reweights by w^{1/M}, resamples, and Langevin-
    rejuvenates toward mu; the per-rung weight at a moved y is recomputed with a
    cheap FORWARD G(y) (no extra inverse). M=1 == HD_Product's one-step surrogate."""
    for k in range(M):
        inc = logw / M
        w = (inc - inc.max()).exp()
        y = resample(y, w)
        y = langevin(y, u1, step=P.IS_MC_STEP, iters=P.IS_MC_ITERS)
        if k < M - 1:                       # recompute logw at the moved y (forward only)
            xf, ladj_f = G.call_and_ladj(y)  # xf = G(y), ladj_f = log|det J_G(y)|
            logw = u0(xf) - u1(y) - ladj_f
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, required=True, help='AIS ladder length (single value)')
    ap.add_argument('--steps', type=int, default=None, help='override STEPS')
    args = ap.parse_args()

    M = args.M
    steps = args.steps if args.steps is not None else P.STEPS
    k, d = P.K, P.dim()

    log(SWEEP, f"##### LADDER RUN START M={M} d={d} steps={steps} device={device} #####")
    t_all = time.perf_counter()

    # targets (identical to HD_Product)
    u0 = Gaussian(mean=[0.0] * d, variance=[P.SIGMA ** 2] * d).to(device)
    u1 = MultiWell(k).to(device)
    u1.enable_grad(); u1.enable_eval()                 # Langevin / QT fast paths

    torch.manual_seed(0)
    x_pool = u0.samples(P.n_train())

    # QT pool for hat_mu (X_mix's QT half)
    torch.manual_seed(1)
    t_qt = time.perf_counter()
    hat_pool = quench_and_temper(u0.samples(P.p_qt()), u1, sigma=P.QT_SIGMA,
                                 opt_step=P.QT_OPT_STEP, opt_iters=P.QT_OPT_ITERS,
                                 mc_step=P.QT_MC_STEP, mc_iters=P.QT_MC_ITERS)
    assert torch.isfinite(hat_pool).all(), "QT produced non-finite output"
    sync()
    qt_cov, _ = mode_coverage(hat_pool, k, frac=P.MODE_FRAC)
    log(SWEEP, f"QT pool {hat_pool.shape[0]} in {time.perf_counter()-t_qt:.1f}s mode_cov={qt_cov:.3f}")

    # ONE flow, pure eager (no compile) -- exactly HD_Product.
    flow = new_flow(d)
    optimizer = torch.optim.Adam(flow.parameters(), lr=P.LR)

    N = x_pool.shape[0]
    Phat = hat_pool.shape[0]
    half = P.BATCH // 2
    ess_history = []                                   # direct mu/nu ESS, one per step
    report_every = max(1, steps // 100)
    t0 = time.perf_counter()
    t_last, step_last = t0, 0
    aborted = None

    for step in range(steps):
        x = x_pool[torch.randperm(N, device=device)[:P.BATCH]]

        with torch.no_grad():
            G_now = flow.t()
            # THE single inverse: y = G^{-1}(x) and its ladj in one pass (HD_Product).
            y, ladj = G_now.inv.call_and_ladj(x)
            logw = -u1(y) + u0(x) + ladj               # log w(y) = log mu/nu  (direct ESS)
            ess = compute_ESS_log(logw).item()
            ess_history.append(ess)

            # mu-surrogate: M-rung ladder reusing y (M=1 == HD_Product one-step).
            y_mu = ais_ladder(y, logw, x, u0, u1, G_now, M)

            # X_mix hat half: fresh QT draw + short Langevin (HD_Product).
            y_hat = langevin(hat_pool[torch.randint(0, Phat, (P.BATCH,), device=device)],
                             u1, step=P.IS_MC_STEP, iters=P.IS_MC_ITERS)
            y_mix = torch.cat([y_hat[:half], y[:half]], dim=0)

        G = flow.t()
        loss = (loss_KL(y_mu, u0, u1, G)
                + P.LAMBDA * loss_X(y_mu, u0, u1, G)
                + loss_X(y_mix, u0, u1, G))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if not torch.isfinite(loss):
            aborted = f"non-finite loss at step {step}"
            log(STATUS, f"M={M:<2} ABORTED: {aborted}"); break

        if (step + 1) % report_every == 0 or step == 0:
            sync()
            now = time.perf_counter()
            ms = 1000.0 * (now - t_last) / max(1, (step + 1) - step_last)   # PER-STEP, not cumulative
            t_last, step_last = now, step + 1
            log(STATUS, f"M={M:<2} step {step+1:>5}/{steps}  {100.0*(step+1)/steps:5.1f}%  "
                        f"loss={loss.item():.3e}  ess={ess:.3f}  {ms:6.1f}ms/step")

    sync()
    wall = time.perf_counter() - t0

    # final eval: direct ESS + mode coverage on a fresh source pool (one more inverse).
    torch.manual_seed(123)
    with torch.no_grad():
        xv = u0.samples(min(P.n_valid(), 20000))
        yv, ladjv = flow.t().inv.call_and_ladj(xv)
        logwv = -u1(yv) + u0(xv) + ladjv
        final_ess = compute_ESS_log(logwv).item()
    mcov, mbal = mode_coverage(yv, k, frac=P.MODE_FRAC)
    kcov = coverage(yv[:20000], hat_pool[:min(2000, hat_pool.shape[0])], k=5)

    torch.save({
        'M': M, 'k': k, 'd': d, 'steps': steps, 'method': P.METHOD,
        'config': {kk: getattr(P, kk) for kk in
                   ('SIGMA', 'NSF_LIM', 'BINS', 'TRANSFORMS', 'HIDDEN', 'BATCH', 'LR',
                    'LAMBDA', 'ALPHA', 'BETA', 'IS_MC_STEP', 'IS_MC_ITERS', 'QT_SIGMA', 'STEPS')},
        'qt_mode_coverage': qt_cov,
        'ess_history': ess_history,               # flat list, one direct ESS per step
        'final_ess': final_ess,
        'mode_coverage': mcov, 'mode_balance': mbal, 'knn_coverage': kcov,
        'wall_s': wall, 'aborted': aborted,
        'samples': yv[:20000].cpu().clone(),
        'state_dict': {kk: v.cpu().clone() for kk, v in flow.state_dict().items()},
    }, HERE / f"data_M{M}.pth")

    log(SWEEP, f"##### M={M} DONE ess={final_ess:.4f} mode_cov={mcov:.3f} knn={kcov:.3f} "
               f"wall={wall:.1f}s in {time.perf_counter()-t_all:.1f}s -> data_M{M}.pth #####"
               + ("  [ABORTED]" if aborted else ""))


if __name__ == '__main__':
    main()
