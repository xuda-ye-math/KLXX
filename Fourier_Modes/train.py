# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""Fourier-field Bayesian inverse problem (idea_C). Trains the 4 X-functional
losses (KL, KL+X_mu, KL+X_mu+X_hat_mu, KL+X_mu+X_mix) with one-step importance
sampling (M=1) on a 4-mode sign-flip posterior, and evaluates ALL methods on the
SAME fine-grid target U_fine for an honest ESS comparison. Progress is written to
train_status.log (no tqdm). Saves data.pth.

Usage:
  python train.py            # full run (parameters.STEPS)
  python train.py --steps 200  # sanity run
"""
import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import (
    importance_weights, compute_ESS_log, resample, langevin,
    suppress_warnings, set_cache_size_limit,
)

import core
from core import (
    FourierFieldPosterior, loss_KL, loss_X, quench_and_temper,
    coverage, mode_coverage_nearest,
)
import parameters as P

suppress_warnings()
set_cache_size_limit(64)

HERE = Path(__file__).resolve().parent
device = 'cuda' if torch.cuda.is_available() else 'cpu'

STATUS = HERE / 'train_status.log'


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
    """Returns (u0 source, u1_train coarse target, u1_fine eval target, centers,
    theta_star). u1_train defines the training potential / IS weights; u1_fine is
    the honest eval potential on the finer grid."""
    pv = core.prior_var().to(device)
    theta_star = core.true_theta(seed=P.DATA_SEED, amp=P.AMP)

    # Fine grid: data generation + post-training eval.
    fine_pts = core.cell_grid(P.FINE_N)
    Phi_g_fine = core.group_basis(fine_pts)
    data_fine = core.make_data(theta_star, Phi_g_fine, P.SIGMA_OBS, seed=P.DATA_SEED)

    # Coarse grid: training likelihood. Subsample the SAME true field on the
    # coarse grid (independent observation-noise draw).
    coarse_pts = core.cell_grid(P.COARSE_N)
    Phi_g_coarse = core.group_basis(coarse_pts)
    data_coarse = core.make_data(theta_star, Phi_g_coarse, P.SIGMA_OBS, seed=P.DATA_SEED + 1)

    u0 = Gaussian(mean=[0.0] * P.DIM, variance=core.prior_var().tolist()).to(device)
    u1_train = FourierFieldPosterior(Phi_g_coarse, data_coarse, core.prior_var(), P.SIGMA_OBS).to(device)
    u1_fine = FourierFieldPosterior(Phi_g_fine, data_fine, core.prior_var(), P.SIGMA_OBS).to(device)

    u1_train.enable_grad(); u1_train.enable_eval()
    u1_fine.enable_grad(); u1_fine.enable_eval()

    centers = core.mode_centers(theta_star).to(device)
    return u0, u1_train, u1_fine, centers, theta_star


def new_flow():
    torch.manual_seed(0)
    flow = NSF(a=[-P.NSF_LIM] * P.DIM, b=[P.NSF_LIM] * P.DIM,
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
            # IS weights derived from that ladj. Weights use the TRAINING target.
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


def eval_flow(flow, u0, u1_train, u1_fine, n_valid, chunk=20000):
    """Honest eval: ESS on the training target U_train (where the fake-ESS contrast
    lives) AND on the fine-grid target U_fine (per spec). Same flow, same fresh
    source pool, no self-normalized tricks. The two ESS numbers use IDENTICAL
    pushforward samples, only the target differs.
    Returns (ess_train, ess_fine, samples)."""
    Gi = flow.t().inv
    ys, logws_train, logws_fine, x_chunks = [], [], [], []
    with torch.no_grad():
        done = 0
        while done < n_valid:
            b = min(chunk, n_valid - done)
            x = u0.samples(b)
            y, ladj = Gi.call_and_ladj(x)
            logw_t = -u1_train(y) + u0(x) + ladj
            logw_f = -u1_fine(y) + u0(x) + ladj
            ys.append(y); logws_train.append(logw_t); logws_fine.append(logw_f)
            done += b
    ess_train = compute_ESS_log(torch.cat(logws_train)).item()
    ess_fine = compute_ESS_log(torch.cat(logws_fine)).item()
    y = torch.cat(ys)
    return ess_train, ess_fine, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--tag', type=str, default='')       # suffix for the .pth (e.g. 'sanity')
    args = ap.parse_args()
    steps = args.steps if args.steps is not None else P.STEPS

    data_path = HERE / (f"data_{args.tag}.pth" if args.tag else "data.pth")

    log(f"##### RUN START steps={steps} device={device} tag={args.tag or '(full)'} #####")
    t_all = time.perf_counter()

    u0, u1_train, u1_fine, centers, theta_star = build_targets()
    log(f"targets built: d={P.DIM} G={P.GROUPS} nmodes={core.NMODES}  "
        f"||theta*||={theta_star.norm():.3f}  U_train(modes)="
        f"{[round(v,2) for v in u1_train(centers).tolist()]}")

    # source pool (shared across all runs)
    torch.manual_seed(0)
    x_pool = u0.samples(P.N_TRAIN)

    # QT pool for hat_mu, built ONCE.
    torch.manual_seed(1)
    t_qt = time.perf_counter()
    x_qt = u0.samples(P.P_QT)
    hat_pool = quench_and_temper(x_qt, u1_train, sigma=P.QT_SIGMA,
                                 opt_step=P.QT_OPT_STEP, opt_iters=P.QT_OPT_ITERS,
                                 mc_step=P.QT_MC_STEP, mc_iters=P.QT_MC_ITERS)
    assert torch.isfinite(hat_pool).all(), "QT produced non-finite output (LBFGS step too large?)"
    sync()
    qt_cov, qt_occ, qt_counts = mode_coverage_nearest(hat_pool, centers, frac=P.MODE_FRAC)
    log(f"QT pool {hat_pool.shape[0]} samples in {time.perf_counter()-t_qt:.1f}s  "
        f"modes_found={qt_cov*core.NMODES:.0f}/{core.NMODES}  occ={[round(v,3) for v in qt_occ.tolist()]}")

    runs = {}
    for m in P.METHODS:
        log(f"=== train {m} ===")
        flow, ess_hist, aborted = train_method(m, u0, u1_train, x_pool, hat_pool, steps)

        # Honest eval for ALL 4 methods on the SAME fresh source pool, with the
        # SAME target potentials -- computes BOTH training-target ESS (where the
        # fake-ESS contrast lives) and fine-grid ESS (per spec). No SNIS tricks.
        torch.manual_seed(123)
        ess_train, ess_fine, y = eval_flow(flow, u0, u1_train, u1_fine, P.N_VALID)
        mcov, occ, counts = mode_coverage_nearest(y, centers, frac=P.MODE_FRAC)
        kcov = coverage(y[:20000], hat_pool[:min(2000, hat_pool.shape[0])], k=P.KNN_K)

        runs[m] = {
            'ess_history': ess_hist,
            'ess_train': ess_train,               # training-target (coarse) ESS
            'final_ess': ess_fine,                # fine-grid ESS (per spec)
            'mode_coverage': mcov,                # modes_found / 4
            'modes_found': int(round(mcov * core.NMODES)),
            'occupancy': occ,                     # [4] occupancy fraction
            'counts': counts,                     # [4] raw counts
            'knn_coverage': kcov,                 # Naeem k=5 vs QT pool
            'aborted': aborted,
            'samples': y[:20000].cpu().clone(),
        }
        log(f"{m:<18} train_ess={ess_train:.4f}  fine_ess={ess_fine:.4f}  "
            f"modes_found={runs[m]['modes_found']}/{core.NMODES}  "
            f"knn_cov={kcov:.3f}  occ={[round(v,3) for v in occ.tolist()]}"
            + ("  [ABORTED]" if aborted else ""))

    torch.save({
        'steps': steps,
        'd': P.DIM, 'groups': P.GROUPS, 'nmodes': core.NMODES,
        'theta_star': theta_star.cpu(),
        'centers': centers.cpu(),
        'config': {kk: getattr(P, kk) for kk in
                   ('AMP', 'SIGMA_OBS', 'COARSE_N', 'FINE_N', 'NSF_LIM', 'BINS',
                    'TRANSFORMS', 'BATCH', 'LR', 'LAMBDA', 'IS_MC_STEP', 'IS_MC_ITERS',
                    'QT_SIGMA', 'QT_OPT_STEP', 'MODE_FRAC', 'KNN_K')},
        'qt_mode_coverage': qt_cov,
        'qt_occupancy': qt_occ,
        'runs': runs,
    }, data_path)
    log(f"##### RUN DONE in {time.perf_counter()-t_all:.1f}s -> {data_path.name} #####")


if __name__ == '__main__':
    main()
