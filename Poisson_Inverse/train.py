# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Driver: screened-Poisson inverse Boltzmann generator via Algorithm 4.

One ladder per invocation, loss chosen by --method (kl first, then balance --
user protocol: observe the bare KL fail, then the balanced loss repair):

    ~/.envs/torch/bin/python train.py --method kl --m-low 6 --m-full 8
    ~/.envs/torch/bin/python train.py --method balance --m-low 6 --m-full 8

The acceptance gate uses the FINE validation ESS (user rule): extend the stage
inverse by identity on the whitened high modes and reweight against the
t_k-tempered FULL posterior. Early-abort rule: direct ESS < 0.05 at any
200-step checkpoint aborts the stage attempt.

Writes train_status.log, data_<tag>.pth (per-stage state_dicts), results_table
.md/.csv, summary.md, figures/ladder_<tag>.png, figures/wells_<tag>.png.
"""
import sys
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import compute_ESS_log, suppress_warnings
from core import run_boltzmann, compose_pushforward, identity_wrap
import parameters as PRM
import potential as pot

suppress_warnings()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
STATUS = HERE / 'train_status.log'


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(STATUS, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


def well_census(xi_low, std_all, tr_nc, alpha, weights=None):
    """(n, s) symmetry-label occupancy of low-mode samples; optional weights
    give the reweighted occupancy. Returns dict label -> occupancy."""
    th0 = xi_low[:, 0] * std_all[0]
    n_lab = torch.round(th0 * alpha / (2.0 * math.pi)).long()
    s_lab = ((xi_low * tr_nc.unsqueeze(0)).sum(-1) > 0).long()
    pair = n_lab * 2 + s_lab
    w = (torch.ones_like(th0) if weights is None else weights)
    w = w / w.sum()
    out = {}
    for u_ in pair.unique().tolist():
        out[(u_ // 2 if u_ >= 0 else -((-u_ + 1) // 2), u_ % 2)] = \
            w[pair == u_].sum().item()
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['balance', 'kl'], default='kl')
    ap.add_argument('--m-low', type=int, default=4)
    ap.add_argument('--m-full', type=int, default=8)
    ap.add_argument('--gate-tau', type=float, default=PRM.VALIDATION_TAU,
                    help='validation gate floor; 0 disables the gate so a '
                         'collapsing baseline still completes its ladder '
                         '(per-stage ESS remains logged)')
    ap.add_argument('--n-sensors', type=int, default=0,
                    help='0: auto = ceil(1.5 * d_low) (overdetermination rule)')
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    PRM.M_LOW, PRM.M_FULL = args.m_low, args.m_full
    PRM.N_SENSORS = args.n_sensors or int(1.5 * args.m_low ** 2 + 0.5)
    tag = f"{args.method}{args.suffix}"      # full regime recorded in config

    torch.manual_seed(0)
    B = pot.build(PRM, device)
    d, d_full = B['d_low'], B['d_full']
    std_all = B['std_all'].to(device)
    u0 = Gaussian([0.0] * d, [1.0] * d, device=device)
    u = B['u_low']                                      # 0.5|xi|^2 + Phi_low
    for q in (u0, u):
        q.enable_grad(mode="reduce-overhead")
        q.enable_eval(mode="default")
    u_full_base = B['u_full']                           # untempered Phi_full

    def fine_fn(y_tilde, logw_low, t_k):
        """FINE validation ESS at stage temper t_k: extend the CARRIED
        validation pre-images y_tilde ~ nu_k by identity on the whitened high
        modes; logw_fine = logw_low + t_k*(Phi_low(y) - Phi_full([y; xi_h]))."""
        tk = float(t_k)
        lws = []
        with torch.no_grad():
            for yb, lb in zip(y_tilde.split(50000), logw_low.split(50000)):
                xh = torch.randn(yb.shape[0], d_full - d, device=device)
                ye = torch.cat([yb, xh], dim=1)
                lws.append(lb + tk * (u.misfit(yb) - u_full_base.misfit(ye)))
        return compute_ESS_log(torch.cat(lws)).item()

    def flow_factory():
        flow = NSF(a=[-PRM.NSF_LIM] * d, b=[PRM.NSF_LIM] * d, bins=PRM.BINS,
                   transforms=PRM.TRANSFORMS,
                   hidden_features=PRM.HIDDEN).to(device)
        flow.zeros()
        return flow

    def qt_fn(u_next):
        return pot.quench_and_temper(u_next, PRM.N_POOL, d, 2.0, PRM.OPT_STEP,
                                     PRM.OPT_ITERS, PRM.MC_STEP, PRM.MC_ITERS,
                                     device)

    cfg = dict(m_low=args.m_low, m_full=args.m_full, d=d, d_full=d_full,
               method=args.method, n_valid=PRM.N_VALID, n_pool=PRM.N_POOL,
               n_batch=PRM.N_BATCH, steps=PRM.STEPS, lr=PRM.LR,
               gate_tau=args.gate_tau,
               t_safe=PRM.T_SAFE, sigma_obs=PRM.SIGMA_OBS, delta=PRM.DELTA,
               alpha=PRM.ALPHA, prior_s=PRM.PRIOR_S, n_sensors=PRM.N_SENSORS,
               tilt=PRM.EPS_TILT)
    def checkpoint_fn(stages, t_now):
        torch.save(dict(tag=tag, config=cfg, partial=True, t_now=t_now,
                        ladder=[s['t'] for s in stages],
                        stages=[{k_: v for k_, v in s.items() if k_ != 'flow'}
                                for s in stages]),
                   HERE / f'data_{tag}_partial.pth')
        log(f"[checkpoint] data_{tag}_partial.pth updated (K={len(stages)}, "
            f"t={t_now:.4f})")

    log(f"##### POISSON RUN START {tag} d={d} d_full={d_full} "
        f"method={args.method} t_safe={PRM.T_SAFE} device={device} #####")
    t0 = time.perf_counter()
    stages, Y, complete, flow, F_inv = run_boltzmann(
        u0, u, flow_factory, n_valid=PRM.N_VALID, n_pool=PRM.N_POOL,
        n_batch=PRM.N_BATCH, steps=PRM.STEPS, lr=PRM.LR, lam=PRM.LAMBDA,
        mc_step=PRM.MC_STEP, mc_iters=PRM.MC_ITERS,
        smc_rungs=PRM.SMC_RUNGS, smc_rung_iters=PRM.SMC_RUNG_ITERS,
        adaptive_tau=PRM.ADAPIVE_TAU, validation_tau=args.gate_tau,
        shrink=PRM.SHRINK_FACTOR, wrap=identity_wrap, qt_fn=qt_fn,
        device=device, status=log, max_stages=PRM.MAX_STAGES,
        max_retry=PRM.MAX_RETRY, t_safe=PRM.T_SAFE, method=args.method,
        grad_clip=PRM.GRAD_CLIP, max_skip=PRM.MAX_SKIP, fine_fn=fine_fn,
        checkpoint_fn=checkpoint_fn)
    wall = time.perf_counter() - t0

    # ---- final evaluation: compose stages, extend, reweight FULL posterior ----
    n_eval = 100000
    torch.manual_seed(123)
    y_low, logw_low = compose_pushforward(
        flow, F_inv, [s['state_dict'] for s in stages], u0, u, n_eval, device)
    low_ess = compute_ESS_log(logw_low).item()
    u_full = B['u_full']
    with torch.no_grad():
        xh = torch.randn(n_eval, d_full - d, device=device)
        y_ext = torch.cat([y_low, xh], dim=1)
        # logw_full = logw_low + [Phi_low(y_low) - Phi_full(y_ext)] (priors of
        # the high block cancel against their source draw)
        logw_full = logw_low + u.misfit(y_low) - u_full.misfit(y_ext)
    fine_ess = compute_ESS_log(logw_full).item()

    tr_nc = B['xi_truth'][:d].to(device).clone(); tr_nc[0] = 0.0
    occ_push = well_census(y_low, std_all, tr_nc, PRM.ALPHA)
    w_full = (logw_full - logw_full.max()).exp()
    occ_rew = well_census(y_low, std_all, tr_nc, PRM.ALPHA, weights=w_full)
    log(f"##### DONE {tag}: complete={complete} K={len(stages)} "
        f"ladder={[round(s['t'], 3) for s in stages]} low ESS={low_ess:.4f} "
        f"FINE full-d ESS={fine_ess:.4f} wall={wall/60:.1f} min #####")
    log(f"wells pushforward: {occ_push}")
    log(f"wells reweighted:  {occ_rew}")

    torch.save(dict(tag=tag, config=cfg, complete=complete,
                    ladder=[s['t'] for s in stages],
                    stages=[{k_: v for k_, v in s.items() if k_ != 'flow'}
                            for s in stages],
                    low_ess=low_ess, fine_ess=fine_ess,
                    occ_push=occ_push, occ_rew=occ_rew, wall_s=wall),
               HERE / f'data_{tag}.pth')

    # ---- results table + figures ----
    rows = []
    for f_ in sorted(HERE.glob('data_*.pth')):
        if 'partial' in f_.name or 'FAILED' in f_.name:
            continue
        D_ = torch.load(f_, weights_only=False, map_location='cpu')
        rows.append((D_['tag'], D_['config']['method'], len(D_['ladder']),
                     D_['complete'],
                     ' '.join(f"{t:.3f}" for t in D_['ladder']),
                     D_['low_ess'], D_['fine_ess'],
                     len(D_['occ_rew']), D_['wall_s'] / 60))
    hdr = ('tag', 'method', 'K', 'complete', 'ladder', 'low_ess',
           'fine_ess', 'wells_rew', 'wall_min')
    with open(HERE / 'results_table.md', 'w') as f:
        f.write('# Poisson_Inverse - Algorithm 4 results\n\n| '
                + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(f"{x:.4f}" if isinstance(x, float)
                                      else str(x) for x in r) + ' |\n')
    with open(HERE / 'results_table.csv', 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    for s in stages:
        ax.plot(s['train_ess_hist'], lw=0.6)
    ax.set_xlabel('step'); ax.set_ylabel('direct ESS'); ax.set_ylim(0, 1)
    ax.set_title(f"{tag}: ladder {[round(s['t'], 2) for s in stages]}")
    ax = axes[1]
    keys = sorted(set(occ_push) | set(occ_rew))
    xs_ = range(len(keys))
    ax.bar([x - 0.2 for x in xs_], [occ_push.get(k, 0) for k in keys],
           width=0.4, label='pushforward', color='0.6')
    ax.bar([x + 0.2 for x in xs_], [occ_rew.get(k, 0) for k in keys],
           width=0.4, label='reweighted', color='tab:purple')
    ax.set_xticks(list(xs_), [f"{k}" for k in keys], rotation=45, fontsize=7)
    ax.set_ylabel('well occupancy'); ax.legend(fontsize=8)
    ax.set_title(f"low ESS {low_ess:.3f}, FINE ESS {fine_ess:.3f}")
    plt.tight_layout()
    plt.savefig(HERE / 'figures' / f'ladder_{tag}.png', dpi=400,
                bbox_inches='tight', pad_inches=0.02)
    log(f"wrote data_{tag}.pth, results_table.md/.csv, figures/ladder_{tag}.png")


if __name__ == '__main__':
    main()
