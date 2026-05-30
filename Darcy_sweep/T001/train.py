"""Darcy_2D — train 4 X-functional losses on the elliptic source-inverse problem.

Same one-step (M=1) IS contract and method set as Wave_2D / Parabolic_2D / HD_Product / Fourier_Modes.
"""
import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
from zflows.flow import NSF
from zflows.potential import Gaussian, Linear_Combination
from zflows.utils import (
    compute_ESS_log, resample, langevin,
    suppress_warnings, set_cache_size_limit,
)

import core
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
    print(line, flush=True)


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


def _build_one_target(n_grid, pvar, data, enable_compile):
    grid_pts = core.cell_grid(n_grid)
    Phi_flat = core.eval_basis(grid_pts)
    inv_lap  = core.inverse_lap_mult(n_grid, dtype=torch.get_default_dtype(), device='cpu')
    sens = core.sensor_indices(n_grid, P.SENSOR_RING_CENTER, P.SENSOR_RING_RADIUS, P.N_SENSORS)
    target = core.DarcyInverse(
        Phi_flat=Phi_flat, inv_lap=inv_lap, sensor_idx=sens, data=data, prior_var_=pvar,
        n_grid=n_grid, g0=P.G0, delta=P.DELTA, alpha=P.ALPHA, sigma_obs=P.SIGMA_OBS,
    ).to(device)
    if enable_compile:
        target.enable_grad(); target.enable_eval()
    return target


def build_targets_and_data():
    torch.set_default_dtype(torch.float32)
    pvar = core.prior_var()
    gen = torch.Generator().manual_seed(P.SEED_TRUTH)
    theta_star = P.A_TRUTH * torch.sqrt(pvar) * torch.randn(core.DIM, generator=gen)

    fine_grid_pts = core.cell_grid(P.FINE_N_GRID)
    fine_Phi_flat = core.eval_basis(fine_grid_pts)
    fine_inv_lap  = core.inverse_lap_mult(P.FINE_N_GRID, torch.float32, 'cpu')
    fine_sens     = core.sensor_indices(P.FINE_N_GRID, P.SENSOR_RING_CENTER,
                                        P.SENSOR_RING_RADIUS, P.N_SENSORS)
    with torch.no_grad():
        clean = core.forward_observation(
            theta_star.unsqueeze(0), fine_Phi_flat, fine_inv_lap, P.FINE_N_GRID,
            fine_sens, P.G0, P.DELTA, P.ALPHA,
        ).squeeze(0)
    gen_n = torch.Generator().manual_seed(P.SEED_NOISE)
    noise = P.SIGMA_OBS * torch.randn(clean.shape, generator=gen_n)
    data = clean + noise

    target_coarse_raw = _build_one_target(P.N_GRID, pvar, data, enable_compile=True)
    target_fine_raw   = _build_one_target(P.FINE_N_GRID, pvar, data, enable_compile=False)
    # Tempered targets: U_beta(theta) = BETA * U(theta). exp(-U_beta) softens wells.
    target_coarse = Linear_Combination([target_coarse_raw], [P.BETA]).to(device)
    target_fine   = Linear_Combination([target_fine_raw],   [P.BETA]).to(device)
    target_coarse.enable_grad(); target_coarse.enable_eval()
    target_fine.enable_grad();   target_fine.enable_eval()
    source        = Gaussian(mean=[0.0] * core.DIM, variance=pvar.tolist()).to(device)
    log(f"TEMPERED TARGET: U_beta = {P.BETA} * U")
    return target_coarse, target_fine, source, theta_star.to(device)


def eval_flow_coarse_fine(flow, source, target_coarse, target_fine, n_valid, chunk=1000):
    Gi = flow.t().inv
    logws_c, logws_f, ys = [], [], []
    with torch.no_grad():
        done = 0
        while done < n_valid:
            b = min(chunk, n_valid - done)
            x = source.samples(b)
            y, ladj = Gi.call_and_ladj(x)
            base = source(x) + ladj
            logws_c.append(-target_coarse(y) + base)
            logws_f.append(-target_fine(y)   + base)
            ys.append(y.cpu())
            done += b
    ess_c = compute_ESS_log(torch.cat(logws_c)).item()
    ess_f = compute_ESS_log(torch.cat(logws_f)).item()
    return ess_c, ess_f, torch.cat(ys)


def train_method(method, source, target, x_pool, hat_pool, steps, prior_std):
    terms = set(method.split('+'))
    use_x_mu     = 'X_mu' in terms
    use_x_hat_mu = 'X_hat_mu' in terms
    use_x_mix    = 'X_mix' in terms
    needs_hat    = use_x_hat_mu or use_x_mix

    torch.manual_seed(0)
    flow = NSF(a=[-P.NSF_LIM] * core.DIM, b=[P.NSF_LIM] * core.DIM,
               bins=P.BINS, transforms=P.TRANSFORMS,
               hidden_features=P.HIDDEN).to(device)
    flow.zeros()
    opt = torch.optim.Adam(flow.parameters(), lr=P.LR)

    N = x_pool.shape[0]
    Phat = hat_pool.shape[0] if hat_pool is not None else 0
    half = P.BATCH // 2
    ess_hist = []
    report_every = max(1, steps // 100)
    t0 = time.perf_counter()

    for step in range(steps):
        idx = torch.randperm(N, device=device)[:P.BATCH]
        x = x_pool[idx]

        with torch.no_grad():
            G_now = flow.t()
            y, ladj = G_now.inv.call_and_ladj(x)
            logw = -target(y) + source(x) + ladj
            ess  = compute_ESS_log(logw).item()
            ess_hist.append(ess)
            iters_per_rung = max(1, P.IS_MC_ITERS // P.IS_M)
            ps = prior_std if getattr(P, 'USE_PRECONDITIONED', False) else None
            y_mu = core.ais_M_pass(y, source, target, G_now, P.IS_M,
                                   step_size=P.IS_MC_STEP, iters_per_rung=iters_per_rung,
                                   prior_std=ps)
            if needs_hat:
                hidx = torch.randint(0, Phat, (P.BATCH,), device=device)
                y_hat = hat_pool[hidx]
                if ps is None:
                    y_hat = langevin(y_hat, target, step=P.IS_MC_STEP, iters=P.IS_MC_ITERS)
                else:
                    y_hat = core.preconditioned_langevin(y_hat, target, ps, P.IS_MC_STEP, P.IS_MC_ITERS)

        G = flow.t()
        loss = core.loss_KL(y_mu, source, target, G)
        if use_x_mu:
            loss = loss + P.LAMBDA * core.loss_X(y_mu, source, target, G)
        if use_x_hat_mu:
            loss = loss + core.loss_X(y_hat, source, target, G)
        if use_x_mix:
            y_mix = torch.cat([y_hat[:half], y[:half]], dim=0)
            loss = loss + core.loss_X(y_mix, source, target, G)

        opt.zero_grad()
        loss.backward()
        gc = getattr(P, 'GRAD_CLIP', 0.0)
        if gc and gc > 0:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=gc)
        opt.step()
        if not torch.isfinite(loss):
            log(f"{method:<18} ABORTED non-finite loss at step {step}")
            return flow, ess_hist
        if (step + 1) % report_every == 0 or step == 0:
            sync()
            elapsed = time.perf_counter() - t0
            ms = 1000.0 * elapsed / (step + 1)
            pct = 100.0 * (step + 1) / steps
            log(f"{method:<18} step {step+1:>5}/{steps}  {pct:5.1f}%  "
                f"loss={loss.item():.3e}  ess={ess:.3f}  "
                f"elapsed={elapsed:6.1f}s  {ms:5.1f}ms/step")
    return flow, ess_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=None)
    args = ap.parse_args()
    steps = args.steps if args.steps else P.STEPS

    log(f"##### START steps={steps} device={device} #####")
    target_coarse, target_fine, source, theta_star = build_targets_and_data()
    log(f"coarse target: d={core.DIM} grid={P.N_GRID}^2")
    log(f"fine target:   d={core.DIM} grid={P.FINE_N_GRID}^2")

    torch.manual_seed(0)
    x_pool = source.samples(P.SOURCE_POOL)

    torch.manual_seed(1)
    x_qt = source.samples(P.QT_POOL)
    prior_std = torch.sqrt(core.prior_var()).to(device)
    qt_prior_std = prior_std if getattr(P, 'USE_PRECONDITIONED', False) else None
    hat_pool = core.quench_and_temper(
        x_qt, target_coarse, sigma=P.QT_SIGMA, opt_step=P.QT_OPT_STEP,
        opt_iters=P.QT_OPT_ITERS, mc_step=P.QT_MC_STEP, mc_iters=P.QT_MC_ITERS,
        prior_std=qt_prior_std,
    )
    log(f"QT pool: {hat_pool.shape[0]} particles  range_norm={hat_pool.norm(dim=-1).max().item():.3f}")

    centers = core.mode_centers(theta_star.cpu(), P.ALPHA).to(device)
    log(f"mode centers: {core.NMODES} (2 sign-flips x 3 k-shifts of theta_star), "
        f"separation min={torch.cdist(centers, centers).fill_diagonal_(1e9).min().item():.3f}")

    runs = {}
    for m in P.METHODS:
        log(f"=== {m} ===")
        flow, ess_hist = train_method(m, source, target_coarse, x_pool, hat_pool, steps, prior_std)
        ess_c, ess_f, y = eval_flow_coarse_fine(flow, source, target_coarse, target_fine, P.N_VALID)
        kcov = core.coverage(y[:3000].to(device), hat_pool, k=5)
        modes_found, occupancy, counts = core.mode_coverage_nearest(y.to(device), centers)
        runs[m] = {
            'ess_history': ess_hist,
            'final_ess_coarse': ess_c,
            'final_ess_fine':   ess_f,
            'knn_coverage':     kcov,
            'modes_found':      modes_found,
            'occupancy':        occupancy,
            'counts':           counts,
            'samples':          y[:5000].clone(),
        }
        log(f"{m:<18} POST-TRAIN coarse_ess={ess_c:.4f}  fine_ess={ess_f:.4f}  "
            f"modes={modes_found*core.NMODES:.0f}/{core.NMODES}  knn_cov={kcov:.3f}  occ={['%.3f'%o for o in occupancy]}")
        # incremental save after every method so crashes don't lose previous methods
        torch.save({
            'theta_star': theta_star.cpu(),
            'config': {k: getattr(P, k) for k in dir(P) if k.isupper()},
            'runs': runs,
        }, HERE / 'data.pth')
        log(f"snapshot saved -> data.pth ({len(runs)} methods)")

    torch.save({
        'theta_star': theta_star.cpu(),
        'config': {k: getattr(P, k) for k in dir(P) if k.isupper()},
        'runs': runs,
    }, HERE / 'data.pth')
    log(f"##### DONE -> data.pth #####")


if __name__ == '__main__':
    main()
