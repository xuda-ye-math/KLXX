# pyright: reportArgumentType=false
"""Monte Carlo scaling of the staged-sampler occupancy bias (B=10k run).

Runs the EXACT staged sampler of Algorithm 4 step (v) -- per stage: load
G_k state_dict (data_L8_balance_B10k.pth, no retraining), push the
compiled inverse, logw = U_{k-1}(x) - U_k(y) + ladj, resample, Langevin
on U_k -- at N = 10000*2^k for k = 0..7 with 2^(7-k) independent tests
(equal total work 1.28M particles per row), and reports the mean
occupancy bias
    err = (1/6) * sum_s |p_s - 1/6|
to test the N^{-1/2} (pure finite-size) hypothesis.

Writes (new files only): occupancy_bias_B10k.md/.csv,
figures/occupancy_bias_B10k.png, occ_bias_B10k_status.log.
Reads data_L8_balance_B10k.pth.

Usage: python occupancy_bias_B10k.py [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                             # Clock_Lattice -> core

from zflows.flow import NCSF
from zflows.potential import Uniform
from zflows.utils import langevin, resample, suppress_warnings

from core import wrap_torus, bridge
import parameters as PRM
from potential import Clock

suppress_warnings()
torch.set_float32_matmul_precision('high')

# dimension and physics come from the CHECKPOINT (train.py overrides the
# parameters.py defaults per run, this script must too)
DATA = torch.load(HERE / 'data_L8_balance_B10k.pth', weights_only=False,
                  map_location='cpu')
CFG = DATA['config']
L, D = CFG['L'], CFG['D']
LIM = PRM.NSF_LIM
N_BASE = 10000
KS = list(range(8))                      # N = N_BASE * 2^k
REPS = {k: 2 ** (7 - k) for k in KS}     # equal total work per row

STATUS = HERE / 'occ_bias_B10k_status.log'


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, 'a') as f:
        f.write(line + '\n')


def pick_chunk(device):
    """Largest chunk that fits free VRAM (one compiled inverse shape)."""
    free, _ = torch.cuda.mem_get_info(device)
    free_gb = free / 1024**3
    if free_gb >= 13.0:
        return 320000
    if free_gb >= 10.0:
        return 160000
    if free_gb >= 6.0:
        return 80000
    return 40000


def sector_bias(y):
    """err = (1/6) sum_s |p_s - 1/6| from magnetization sectors."""
    m = torch.exp(1j * y.to(torch.float32)).mean(dim=1)
    s = torch.remainder(torch.round(torch.angle(m) * PRM.P / (2 * np.pi)),
                        PRM.P).to(torch.long)
    p = torch.bincount(s, minlength=PRM.P).double() / y.shape[0]
    return (p - 1.0 / PRM.P).abs().sum().item() / PRM.P, p.tolist()


def staged_sample(n, G, seed, flow, F_inv, state_dicts, ladder, u0, u, wrap,
                  chunk, device):
    """G INDEPENDENT tests of n particles each, stacked into one [G*n, D]
    GPU pass. Map + Langevin act per particle (block-agnostic); the
    reweight/RESAMPLE is done PER TEST BLOCK, so the G tests are exactly
    independent staged samplers. Returns y of shape [G*n, D]."""
    total = G * n
    torch.manual_seed(seed)                       # resample/Langevin RNG too
    g = torch.Generator(device='cpu').manual_seed(seed)
    y = (torch.rand(total, D, generator=g) * 2.0 - 1.0).mul_(LIM).to(device)
    t_prev = 0.0
    for k, (sd, t_k) in enumerate(zip(state_dicts, ladder), start=1):
        u_prev, u_next = bridge(u0, u, t_prev), bridge(u0, u, t_k)
        flow.load_state_dict({kk: v.to(device) for kk, v in sd.items()})
        outs, lws = [], []
        for i in range(0, total, chunk):
            xb = y[i:i + chunk]
            nb = xb.shape[0]
            if nb < chunk:                                # pad: ONE compile shape
                xb = torch.cat([xb, xb[-1:].expand(chunk - nb, -1)], dim=0)
            with torch.no_grad():
                yt, ladj = F_inv.inv_ladj(xb)
                yt, ladj = yt[:nb].clone(), ladj[:nb].clone()
                lws.append(u_prev.eval(xb[:nb]) - u_next.eval(yt) + ladj)
                outs.append(yt)
        y, logw = torch.cat(outs), torch.cat(lws)
        del outs, lws                          # free before the resample gather
        big = total * D > 40_000_000           # huge pass: stage through CPU
        with torch.no_grad():
            for b in range(G):                            # resample per test block
                sl = slice(b * n, (b + 1) * n)
                lw = logw[sl]
                w = (lw - lw.max()).exp()
                idx = torch.multinomial(w / w.sum(), n, replacement=True)
                if big:                       # gather on CPU, free GPU first
                    yb = y[sl].cpu()[idx.cpu()]
                    if G == 1:
                        del y, w, lw, logw
                        torch.cuda.empty_cache()
                        y = yb.to(device)
                    else:
                        y[sl] = yb.to(device)
                else:
                    y[sl] = y[sl][idx]
            rejuv = []
            for i in range(0, total, chunk):
                r = wrap(langevin(y[i:i + chunk], u_next,
                                  step=PRM.MC_STEP, iters=PRM.MC_ITERS))
                rejuv.append(r.cpu() if big else r)
            if big:
                del y
                torch.cuda.empty_cache()
                y = torch.cat(rejuv).to(device)
            else:
                y = torch.cat(rejuv)
        t_prev = t_k
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='tiny sanity run: N_BASE=2000, k in {0,1}, 2/1 reps')
    ap.add_argument('--kmin', type=int, default=0,
                    help='run only rows k >= kmin (partial rerun)')
    args = ap.parse_args()

    global KS, REPS, N_BASE
    suffix = '_smoke' if args.smoke else ''
    if args.smoke:
        N_BASE, KS, REPS = 2000, [0, 1], {0: 2, 1: 1}
    KS = [k for k in KS if k >= args.kmin]
    if args.kmin > 0:
        suffix += f'_k{args.kmin}up'

    device = 'cuda'
    d = DATA
    ladder = list(d['ladder'])
    state_dicts = [s['state_dict'] for s in d['stages']]
    assert len(state_dicts) == len(ladder) == 6

    u = Clock(L, CFG['P'], CFG['J'], CFG['H']).to(device)
    u0 = Uniform([-LIM] * D, [LIM] * D, device=device)
    u.enable_eval(mode="default"); u0.enable_eval(mode="default")
    u.enable_grad(mode="reduce-overhead"); u0.enable_grad(mode="reduce-overhead")

    def wrap(x):
        return wrap_torus(x, LIM)

    flow = NCSF(a=[-LIM] * D, b=[LIM] * D, bins=PRM.BINS,
                transforms=PRM.TRANSFORMS,
                hidden_features=PRM.HIDDEN).to(device)
    flow.zeros()
    F_inv = flow.t().enable_inv_ladj()

    chunk_big = pick_chunk(device)
    log(f"##### OCC-BIAS-B10k START D={D} N_BASE={N_BASE} ks={KS} "
        f"reps={REPS} chunk_big={chunk_big} (free "
        f"{torch.cuda.mem_get_info(device)[0]/1024**3:.1f} GB) #####")

    rows, t0 = [], time.perf_counter()
    for k in KS:
        n, reps = N_BASE * 2 ** k, REPS[k]
        G = max(1, min(reps, chunk_big // n))    # tests stacked per GPU pass
        chunk = min(chunk_big, G * n)
        errs = []
        for grp in range(reps // G):
            t1 = time.perf_counter()
            y = staged_sample(n, G, seed=10000 * k + grp + 1, flow=flow,
                              F_inv=F_inv, state_dicts=state_dicts,
                              ladder=ladder, u0=u0, u=u, wrap=wrap,
                              chunk=chunk, device=device)
            dt = time.perf_counter() - t1
            for b in range(G):
                err, p = sector_bias(y[b * n:(b + 1) * n])
                errs.append(err)
                log(f"[k={k}] test {len(errs):>3}/{reps}  N={n}  "
                    f"bias={err:.5f}  sector_probs={['%.3f' % q for q in p]}  "
                    f"({dt/G:.0f}s)")
            del y
            torch.cuda.empty_cache()
        errs = np.asarray(errs)
        rows.append(dict(k=k, N=n, reps=reps, bias=errs.mean(),
                         sem=errs.std(ddof=1) / np.sqrt(reps) if reps > 1
                         else float('nan')))
        log(f"[k={k}] DONE  mean bias={errs.mean():.5f}  "
            f"sem={rows[-1]['sem']:.5f}")

    # ---- table ----
    import csv
    with open(HERE / f'occupancy_bias_B10k{suffix}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['k', 'N', 'reps', 'bias', 'sem'])
        w.writeheader()
        w.writerows(rows)
    ratios = [rows[i]['bias'] / rows[i + 1]['bias'] for i in range(len(rows) - 1)]
    slope = np.polyfit(np.log([r['N'] for r in rows]),
                       np.log([r['bias'] for r in rows]), 1)[0]
    with open(HERE / f'occupancy_bias_B10k{suffix}.md', 'w') as f:
        f.write("# Occupancy-bias Monte Carlo scaling (L=8 clock, staged "
                "sampler, balance B=10k, 6 maps)\n\nerr = (1/6) sum_s "
                "|p_s - 1/6|; equal total work per row (10000*128 "
                "particles); 2^(7-k) independent tests at N=10000*2^k.\n\n")
        f.write("| | " + " | ".join(f"N=1e4*2^{r['k']}" for r in rows)
                + " |\n")
        f.write("|---|" + "---|" * len(rows) + "\n")
        f.write("| mean occupancy bias | "
                + " | ".join(f"{r['bias']:.5f}" for r in rows) + " |\n")
        f.write("| sem (over tests) | "
                + " | ".join(f"{r['sem']:.5f}" for r in rows) + " |\n")
        f.write("| tests | " + " | ".join(str(r['reps']) for r in rows)
                + " |\n\n")
        f.write(f"Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): "
                f"{['%.2f' % x for x in ratios]}\n")
        f.write(f"\nLog-log slope of bias vs N: {slope:.3f} "
                f"(Monte Carlo rate = -0.5)\n")

    # ---- figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'mathtext.fontset': 'cm', 'font.family': 'serif',
    })
    Ns = np.array([r['N'] for r in rows], dtype=float)
    bs = np.array([r['bias'] for r in rows])
    sems = np.array([r['sem'] for r in rows])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.errorbar(Ns, bs, yerr=2 * np.nan_to_num(sems), marker='o',
                color='tab:blue', lw=1.2, capsize=3,
                label='measured bias')
    ax.plot(Ns, bs[0] * (Ns / Ns[0]) ** -0.5, ls='--', color='gray',
            label=r'$N^{-1/2}$ reference')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'particle count $N$')
    ax.set_ylabel('occupancy bias')
    ax.set_title(r'staged sampler, balance $B=10^4$')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / 'figures' / f'occupancy_bias_B10k{suffix}.png', dpi=300,
                bbox_inches='tight', pad_inches=0.02)
    log(f"##### OCC-BIAS-B10k DONE wall={time.perf_counter()-t0:.0f}s "
        f"biases={['%.5f' % r['bias'] for r in rows]} ratios="
        f"{['%.2f' % x for x in ratios]} slope={slope:.3f} #####")


if __name__ == '__main__':
    main()
