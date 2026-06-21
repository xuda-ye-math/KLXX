# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Driver: p-state clock Boltzmann generator via Algorithm 4 (Paper/main.tex).

One flow ladder per invocation; the loss is chosen by --method:
    --method balance   KL + X_mu + X_mix at balanced hyperparameters (default)
    --method kl        bare forward KL only (baseline: fused_kl_loss, mean of
                       z_k -- no X terms, no QT pool; all else identical)
The L sweep is separate invocations; tag = L{L}_{method}:
    ~/.envs/torch/bin/python train.py --L 6 --smoke             # tiny sanity pass first
    ~/.envs/torch/bin/python train.py --L 6 --method balance    # full run, D = 36
    ~/.envs/torch/bin/python train.py --L 6 --method kl         # bare KL baseline
    ~/.envs/torch/bin/python train.py --L 8 --method balance    # full run, D = 64

Writes (in this folder): train_status.log, data_<tag>.pth (incl. every stage's
flow state_dict), results_table.md/.csv, summary.md, figures/*.png.
"""
import sys
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                             # Clock_Lattice -> core/

from zflows.flow import NCSF
from zflows.potential import Uniform
from zflows.utils import compute_ESS_log, suppress_warnings
from core import (run_boltzmann, compose_pushforward, quench_and_temper_torus,
                  wrap_torus, bridge)
import parameters as PRM
from potential import Clock, sector_occupancy, torus_coverage

suppress_warnings()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
STATUS = HERE / 'train_status.log'


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(STATUS, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


def write_results():
    """Rebuild results_table.md/.csv from every data_*.pth in this folder."""
    rows = []
    for p in sorted(HERE.glob('data_*.pth')):
        d = torch.load(p, weights_only=False, map_location='cpu')
        c = d['config']
        rows.append(dict(
            tag=d['tag'], D=c['D'], P=c['P'],
            method=c.get('method', 'balance'), K=len(d['ladder']),
            complete=d.get('complete', '?'),
            B=c.get('n_batch', '?'), steps=c.get('steps', '?'),
            ladder=' '.join(f"{t:.3f}" for t in d['ladder']),
            final_ess=d['final_ess'],
            sectors=f"{d['sectors_push']}/{c['P']}",
            tv=d['tv_push'], abs_m=d['abs_m_push'], knn=d['knn_coverage'],
            wall_min=d['wall_s'] / 60.0))
    hdr = ['tag', 'D', 'P', 'method', 'B', 'steps', 'K', 'complete', 'ladder',
           'final_ess', 'sectors', 'tv', 'abs_m', 'knn', 'wall_min']
    with open(HERE / 'results_table.csv', 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in rows:
            f.write(','.join(str(r[h]) for h in hdr) + '\n')
    with open(HERE / 'results_table.md', 'w') as f:
        f.write('# p-state clock — Algorithm 4 results\n\n')
        f.write('| ' + ' | '.join(hdr) + ' |\n')
        f.write('|' + '---|' * len(hdr) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(
                f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h])
                for h in hdr) + ' |\n')


def write_summary(tag, cfg, stages, final_ess, sec, knn_cov, wall,
                  complete=True):
    cov, tv, counts, abs_m = sec
    lines = [
        f"# Clock_Lattice summary — latest run `{tag}`",
        "",
        ("" if complete else
         "**WARNING: INCOMPLETE LADDER — the run stopped before t=1; the "
         "metrics below are vs the FULL target and NOT target-faithful.**"),
        "",
        f"**Target.** p-state clock, P={cfg['P']}, J={cfg['J']}, H={cfg['H']}, "
        f"L={cfg['L']} (D={cfg['D']}), periodic square lattice, torus domain.",
        f"**Method.** Algorithm 4: NCSF per stage, uniform-on-torus source, "
        f"adaptive ladder (ADAPIVE_TAU={PRM.ADAPIVE_TAU}, "
        f"VALIDATION_TAU={PRM.VALIDATION_TAU}, SHRINK={PRM.SHRINK_FACTOR}).",
        "",
        f"**Headline.** K={len(stages)} stages, ladder "
        f"{[round(s['t'], 3) for s in stages]}; composed-generator direct ESS "
        f"= **{final_ess:.4f}**; sectors found (pushforward) = "
        f"**{int(cov * cfg['P'])}/{cfg['P']}** (TV from uniform {tv:.3f}, "
        f"mean |m| {abs_m:.3f}); kNN coverage vs QT set = {knn_cov:.3f}; "
        f"wall {wall/60:.1f} min.",
        "",
        "Per-stage validation ESS (post-training acceptance gate): "
        + ", ".join(f"t={s['t']:.3f}: {s['val_ess']:.3f}"
                    f"{' (retries ' + str(len(s['attempts']) - 1) + ')' if len(s['attempts']) > 1 else ''}"
                    for s in stages),
        "",
        "**Interpretation.** All P sectors found at near-uniform occupancy with "
        "a healthy composed ESS means the ladder bridged the BKT regime without "
        "sector collapse; missing sectors with a high ESS would be the fake-ESS "
        "failure the X functionals are designed to prevent.",
        "",
        "Files: `data_<tag>.pth` (per-stage state_dicts inside), "
        "`results_table.md/.csv`, `figures/ladder_<tag>.png`, "
        "`figures/sectors_<tag>.png`, `figures/magnetization_<tag>.png`, "
        "`train_status.log` (live).",
    ]
    with open(HERE / 'summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=PRM.L)
    ap.add_argument('--method', choices=['balance', 'kl'], default='balance',
                    help='balance: KL + X_mu + X_mix; kl: forward KL only')
    ap.add_argument('--smoke', action='store_true', help='tiny sanity run')
    ap.add_argument('--suffix', default='', help='tag suffix to keep output '
                    'files (data_<tag>.pth, figures) from clashing')
    args = ap.parse_args()
    L, smoke, method = args.L, args.smoke, args.method
    D = L * L

    # smoke overrides scale everything down but exercise the full path
    NV = 4000 if smoke else PRM.N_VALID
    NP = 1000 if smoke else PRM.N_POOL
    NB = 500 if smoke else PRM.N_BATCH
    # tag carries the batch size so runs at different B never clash
    blabel = f"B{NB // 1000}k" if NB % 1000 == 0 else f"B{NB}"
    tag = f"L{L}_{method}_{blabel}{args.suffix}" + ("_smoke" if smoke else "")
    ST = 60 if smoke else PRM.STEPS
    MC = 20 if smoke else PRM.MC_ITERS
    RUNG = 5 if smoke else PRM.SMC_RUNG_ITERS
    OPTI = 30 if smoke else PRM.OPT_ITERS
    MAXS = 3 if smoke else PRM.MAX_STAGES
    MAXR = 4 if smoke else PRM.MAX_RETRY

    cfg = dict(L=L, D=D, P=PRM.P, J=PRM.J, H=PRM.H, method=method,
               n_valid=NV, n_pool=NP,
               n_batch=NB, steps=ST, lr=PRM.LR, lam=PRM.LAMBDA,
               bins=PRM.BINS, transforms=PRM.TRANSFORMS, hidden=PRM.HIDDEN,
               mc_iters=MC, smc_rung_iters=RUNG,
               adaptive_tau=PRM.ADAPIVE_TAU, validation_tau=PRM.VALIDATION_TAU,
               shrink=PRM.SHRINK_FACTOR, smoke=smoke, t_safe=PRM.T_SAFE,
               grad_clip=PRM.GRAD_CLIP, max_skip=PRM.MAX_SKIP)

    torch.manual_seed(0)
    lim = PRM.NSF_LIM                       # canonical box half-width (= pi)
    u0 = Uniform([-lim] * D, [lim] * D, device=device)
    u = Clock(L, PRM.P, PRM.J, PRM.H).to(device)
    for pot in (u0, u):
        pot.enable_grad(mode="reduce-overhead")   # Langevin grad fast path
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

    log(f"##### CLOCK RUN START {tag} D={D} P={PRM.P} J={PRM.J} H={PRM.H} "
        f"method={method} device={device} #####")
    t0 = time.perf_counter()
    stages, Y, complete, flow, F_inv = run_boltzmann(
        u0, u, flow_factory, n_valid=NV, n_pool=NP, n_batch=NB, steps=ST,
        lr=PRM.LR, lam=PRM.LAMBDA,
        mc_step=PRM.MC_STEP, mc_iters=MC,
        smc_rungs=PRM.SMC_RUNGS, smc_rung_iters=RUNG,
        adaptive_tau=PRM.ADAPIVE_TAU,
        validation_tau=PRM.VALIDATION_TAU, shrink=PRM.SHRINK_FACTOR,
        wrap=wrap, qt_fn=qt_fn, device=device, status=log,
        max_stages=MAXS, max_retry=MAXR, t_safe=PRM.T_SAFE, method=method,
        grad_clip=PRM.GRAD_CLIP, max_skip=PRM.MAX_SKIP)
    wall = time.perf_counter() - t0

    # ---- final evaluation: compose the stage inverses (shared flow) ----
    n_eval = min(NV, 20000)
    torch.manual_seed(123)
    y_final, logw = compose_pushforward(
        flow, F_inv, [s['state_dict'] for s in stages], u0, u, n_eval, device)
    final_ess = compute_ESS_log(logw).item()
    sec_push = sector_occupancy(y_final, PRM.P, PRM.MODE_FRAC)
    sec_val = sector_occupancy(Y[:n_eval], PRM.P, PRM.MODE_FRAC)
    hat_final = qt_fn(bridge(u0, u, 1.0))
    knn_cov = torus_coverage(y_final[:10000], hat_final[:2000], k=5)

    ladder = [s['t'] for s in stages]
    if not complete:
        log(f"!!!!! {tag} INCOMPLETE LADDER: reached t={ladder[-1] if ladder else 0:.4f} < 1 "
            f"-- final ESS/sector metrics below are vs the FULL target and NOT "
            f"target-faithful !!!!!")
    log(f"##### {tag} DONE K={len(stages)} complete={complete} "
        f"ladder={[f'{t:.3f}' for t in ladder]} final_ESS={final_ess:.4f} "
        f"sectors(push)={int(sec_push[0]*PRM.P)}/{PRM.P} tv={sec_push[1]:.3f} "
        f"|m|={sec_push[3]:.3f} knn={knn_cov:.3f} wall={wall:.0f}s #####")

    torch.save({
        'tag': tag, 'config': cfg, 'ladder': ladder, 'complete': complete,
        'stages': stages,                       # includes every state_dict
        'final_ess': final_ess,
        'sectors_push': int(sec_push[0] * PRM.P), 'tv_push': sec_push[1],
        'counts_push': sec_push[2], 'abs_m_push': sec_push[3],
        'sectors_valid': int(sec_val[0] * PRM.P), 'tv_valid': sec_val[1],
        'counts_valid': sec_val[2], 'abs_m_valid': sec_val[3],
        'knn_coverage': knn_cov, 'wall_s': wall,
        'samples_push': y_final[:20000].cpu().clone(),
        'samples_valid': Y[:20000].cpu().clone(),
        'logw_push': logw[:20000].cpu().clone(),
    }, HERE / f"data_{tag}.pth")

    write_results()
    write_summary(tag, cfg, stages, final_ess, sec_push, knn_cov, wall,
                  complete=complete)
    log(f"saved data_{tag}.pth + results_table + summary")


if __name__ == '__main__':
    main()
