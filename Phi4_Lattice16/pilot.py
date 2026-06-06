# pyright: reportArgumentType=false
"""Parameter pilot for the 6x6 phi^4 benchmark: parallel-tempering MALA reference.

Scans KAPPA at H=0 to find a magnetization barrier in the 8-12 kT window,
then fixes H for Delta F approx 1 kT and produces the frozen reference set.

Writes (new files only): pilot_results.md, phi4_reference.pth,
figures/pilot_scan.png, pilot_status.log.  Reads nothing.

Usage: python pilot.py [--quick]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from parameters import L, D, LAMBDA

HERE = Path(__file__).resolve().parent
STATUS = HERE / 'pilot_status.log'
torch.set_float32_matmul_precision('high')


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, 'a') as f:
        f.write(line + '\n')


def action(phi, kappa, lam, h):
    p = phi.view(-1, L, L)
    hop = p * (torch.roll(p, 1, dims=1) + torch.roll(p, 1, dims=2))
    return (-2.0 * kappa * hop + p.square()
            + lam * (p.square() - 1.0).square() + h * p).sum(dim=(1, 2))


def grad_action(phi, kappa, lam, h):
    x = phi.detach().requires_grad_(True)
    s = action(x, kappa, lam, h).sum()
    (g,) = torch.autograd.grad(s, x)
    return g


def pt_run(kappa, h, n_steps, n_chains, device, step=0.02, ladder=None, seed=0):
    """Parallel tempering over action scalings t*S; MALA within each rung.
    Returns magnetization trace at t=1 and round-trip count."""
    torch.manual_seed(seed)
    if ladder is None:
        ladder = torch.linspace(0.05, 1.0, 20, device=device)
    K = len(ladder)
    lam = LAMBDA
    phi = 0.5 * torch.randn(K, n_chains, D, device=device)
    t = ladder.view(K, 1)

    def S(x):  # x: [K, C, D] -> [K, C]
        return action(x.view(-1, D), kappa, lam, h).view(K, n_chains)

    def gradS(x):
        return grad_action(x.view(-1, D), kappa, lam, h).view(K, n_chains, D)

    # track replica positions for round-trip counting (label which rung each
    # walker family is at; swaps exchange configurations, labels follow configs)
    lab = torch.arange(K, device=device).view(K, 1).expand(K, n_chains).clone()
    hit_top = torch.zeros(K, n_chains, dtype=torch.bool, device=device)
    roundtrips = torch.zeros(K, n_chains, dtype=torch.long, device=device)

    m_trace = []
    s_cur = S(phi)
    for it in range(n_steps):
        # MALA on t*S per rung
        g = gradS(phi)
        prop = phi - step * t.view(K, 1, 1) * g + np.sqrt(2 * step) * torch.randn_like(phi)
        s_prop = S(prop)
        gp = gradS(prop)
        fwd = (prop - phi + step * t.view(K, 1, 1) * g).square().sum(-1) / (4 * step)
        bwd = (phi - prop + step * t.view(K, 1, 1) * gp).square().sum(-1) / (4 * step)
        log_acc = -t * (s_prop - s_cur) - bwd + fwd
        acc = torch.rand(K, n_chains, device=device).log() < log_acc
        phi = torch.where(acc.unsqueeze(-1), prop, phi)
        s_cur = torch.where(acc, s_prop, s_cur)
        # neighbor swaps (alternate parity)
        for k0 in range(it % 2, K - 1, 2):
            d_log = (ladder[k0 + 1] - ladder[k0]) * (s_cur[k0 + 1] - s_cur[k0])
            sw = torch.rand(n_chains, device=device).log() < d_log
            for arr in (phi, s_cur, lab, hit_top, roundtrips):
                a, b = arr[k0].clone(), arr[k0 + 1].clone()
                arr[k0][sw], arr[k0 + 1][sw] = b[sw], a[sw]
        # round trips: a config counts one trip when it visits rung 0 then rung K-1
        hit_top[-1] |= True
        done = hit_top[0].clone()
        roundtrips[0][done] += 1
        hit_top[0][done] = False
        if it >= n_steps // 3:                      # burn-in: first third
            m_trace.append(phi[-1].mean(dim=1).cpu())   # t=1 rung
    m = torch.stack(m_trace)                            # [T, C]
    return m, phi[-1].cpu(), int(roundtrips.sum().item())


def barrier_from_hist(m, bins=61, lim=1.5):
    hist, edges = np.histogram(m.numpy().ravel(), bins=bins, range=(-lim, lim), density=True)
    c = 0.5 * (edges[:-1] + edges[1:])
    pk_neg = hist[(c < -0.3)].max() if (c < -0.3).any() else 0
    pk_pos = hist[(c > 0.3)].max() if (c > 0.3).any() else 0
    val = hist[np.abs(c) < 0.15].max()
    val = max(val, 1e-12)
    v_pos = c[(c > 0.3)][np.argmax(hist[(c > 0.3)])] if pk_pos > 0 else float('nan')
    return float(np.log(min(pk_neg, pk_pos) / val)), float(v_pos), (hist, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_steps = 2000 if args.quick else 16000
    n_chains = 64 if args.quick else 128

    # ---- stage 1: kappa scan at h = 0 ----
    scan = {}
    for kappa in ([0.34] if args.quick else [0.36, 0.38, 0.40]):
        t0 = time.perf_counter()
        m, _, rt = pt_run(kappa, 0.0, n_steps, n_chains, device, seed=10)
        b, v, (hist, c) = barrier_from_hist(m)
        scan[kappa] = dict(barrier=b, v=v, hist=hist, centers=c, roundtrips=rt)
        log(f"[scan] kappa={kappa:.2f}  barrier={b:.2f} kT  v={v:.3f}  "
            f"roundtrips={rt}  ({time.perf_counter()-t0:.0f}s)")

    # pick kappa with barrier closest to 10 kT (within [6, 14] acceptable)
    kappa = min(scan, key=lambda k: abs(scan[k]['barrier'] - 12.0))
    v = scan[kappa]['v']
    log(f"[pick] kappa={kappa}  barrier={scan[kappa]['barrier']:.2f}  v={v:.3f}")

    # ---- stage 2: tilt for Delta F approx 1 kT, frozen reference ----
    h = 1.0 / (D * max(v, 0.3))            # first guess: h*D*v ~ 1 kT
    m2, samples_t1, rt2 = pt_run(kappa, h, n_steps, n_chains, device, seed=20)
    mm = m2.ravel()
    p_plus = (mm > 0).float().mean().item()
    dF = float(np.log(max(p_plus, 1e-9) / max(1 - p_plus, 1e-9)))   # F_- - F_+
    b2, v2, (hist2, c2) = barrier_from_hist(m2)
    log(f"[tilt] h={h:.4f}  p(+)={p_plus:.3f}  DeltaF={dF:.2f} kT  "
        f"barrier={b2:.2f}  roundtrips={rt2}")

    torch.save(dict(kappa=kappa, lam=LAMBDA, h=h, v=v2, barrier=b2,
                    delta_F=dF, p_plus=p_plus, roundtrips=rt2,
                    m_trace=m2, samples_t1=samples_t1),
               HERE / 'phi4_reference.pth')

    with open(HERE / 'pilot_results.md', 'w') as f:
        f.write("# phi^4 6x6 pilot (PT-MALA reference)\n\n")
        f.write("| kappa | barrier (kT) | v | roundtrips |\n|---|---|---|---|\n")
        for k in sorted(scan):
            s = scan[k]
            f.write(f"| {k:.2f} | {s['barrier']:.2f} | {s['v']:.3f} | {s['roundtrips']} |\n")
        f.write(f"\nFrozen: KAPPA={kappa}, LAMBDA={LAMBDA}, H={h:.4f} -> "
                f"barrier {b2:.2f} kT, v={v2:.3f}, DeltaF={dF:.2f} kT "
                f"(p_+={p_plus:.3f}), roundtrips={rt2}.\n")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for k in sorted(scan):
        axes[0].semilogy(scan[k]['centers'], scan[k]['hist'] + 1e-12, label=f"$\\kappa={k}$")
    axes[0].set_xlabel('magnetization $m$'); axes[0].set_ylabel('$p(m)$')
    axes[0].legend(fontsize=7); axes[0].set_title('(a) $\\kappa$ scan, $h=0$')
    axes[1].semilogy(c2, hist2 + 1e-12, color='k')
    axes[1].set_xlabel('magnetization $m$'); axes[1].set_ylabel('$p(m)$')
    axes[1].set_title(f'(b) frozen: $\\kappa={kappa}$, $h={h:.3f}$')
    plt.tight_layout()
    plt.savefig(HERE / 'figures' / 'pilot_scan.png', dpi=300, bbox_inches='tight')
    log("pilot DONE")


if __name__ == '__main__':
    main()
