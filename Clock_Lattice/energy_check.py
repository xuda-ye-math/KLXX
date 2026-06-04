# pyright: reportArgumentType=false, reportCallIssue=false
"""Post-hoc physics check for a Clock_Lattice run (reviewer-adopted diagnostic).

From the saved data_<tag>.pth (no retraining): reweighted mean energy <E> and
heat-capacity proxy C_v = Var_w(E) of the composed pushforward, compared to a
classical SMC reference (uniform -> target, no flow), plus a per-sector ESS
breakdown. Appends the comparison to results_table.md and writes
figures/energy_<tag>.png.

Usage:  ~/.envs/torch/bin/python energy_check.py L4
"""
import sys
import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zflows.potential import Uniform
from zflows.utils import sequential_monte_carlo, compute_ESS_log
from core import wrap_torus
import parameters as PRM
from potential import Clock, magnetization

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weighted_stats(E: torch.Tensor, logw: torch.Tensor):
    w = (logw - logw.max()).exp()
    w = w / w.sum()
    mean = (w * E).sum()
    var = (w * (E - mean) ** 2).sum()
    return mean.item(), var.item()


def main(tag: str):
    d = torch.load(HERE / f"data_{tag}.pth", weights_only=False,
                   map_location='cpu')
    cfg = d['config']
    L, D, P = cfg['L'], cfg['D'], cfg['P']
    u = Clock(L, P, cfg['J'], cfg['H']).to(device)
    u.enable_grad(mode="default")
    u.enable_eval(mode="default")
    u0 = Uniform([-math.pi] * D, [math.pi] * D, device=device)
    u0.enable_grad(mode="default")
    u0.enable_eval(mode="default")

    # generator side: reweighted <E>, C_v from saved pushforward + weights
    y = d['samples_push'].to(device)
    logw = d['logw_push'].to(device)
    E = u.eval(y)
    gen_mean, gen_var = weighted_stats(E, logw)

    # reference side: classical SMC, uniform -> target, many rungs (no flow)
    torch.manual_seed(7)
    ref = u0.samples(min(20000, y.shape[0]))
    ref, _ = sequential_monte_carlo(ref, u0, u, ladder=4 * D,
                                    step=PRM.MC_STEP, iters=PRM.SMC_RUNG_ITERS)
    ref = wrap_torus(ref, math.pi)
    E_ref = u.eval(ref)
    ref_mean, ref_var = E_ref.mean().item(), E_ref.var().item()

    # per-sector ESS breakdown of the generator weights
    m = magnetization(y.cpu())
    sector = torch.round(torch.angle(m) * P / (2 * math.pi)).long() % P
    rows = []
    for k in range(P):
        sel = sector == k
        n = int(sel.sum().item())
        ess_k = compute_ESS_log(logw[sel.to(device)]).item() if n > 1 else float('nan')
        rows.append((k, n, ess_k))

    rel = abs(gen_mean - ref_mean) / abs(ref_mean)
    print(f"[{tag}] reweighted <E> = {gen_mean:.3f} (Var {gen_var:.3f})  vs  "
          f"SMC reference <E> = {ref_mean:.3f} (Var {ref_var:.3f})  "
          f"rel.err = {rel:.4f}")
    for k, n, e in rows:
        print(f"  sector {k}: n={n:>6}  in-sector ESS={e:.3f}")

    with open(HERE / 'results_table.md', 'a') as f:
        f.write(f"\n**Energy check ({tag})**: reweighted <E> = {gen_mean:.3f} "
                f"(Var {gen_var:.3f}) vs classical-SMC reference "
                f"<E> = {ref_mean:.3f} (Var {ref_var:.3f}), relative error "
                f"{rel:.4f}. Per-sector ESS: "
                + ", ".join(f"s{k}: {e:.3f} (n={n})" for k, n, e in rows)
                + "\n")

    fig, ax = plt.subplots(figsize=(5, 3))
    bins = 60
    ax.hist(E_ref.cpu().numpy(), bins=bins, density=True, alpha=0.5,
            label=f'SMC reference (<E>={ref_mean:.1f})')
    w = (logw - logw.max()).exp(); w = (w / w.sum()).cpu().numpy()
    ax.hist(E.cpu().numpy(), bins=bins, density=True, weights=w * len(w),
            alpha=0.5, label=f'generator, reweighted (<E>={gen_mean:.1f})')
    ax.set_xlabel('energy $U$'); ax.set_ylabel('density')
    ax.set_title(f'energy histogram ({tag})'); ax.legend(fontsize=8)
    plt.tight_layout()
    (HERE / 'figures').mkdir(exist_ok=True)
    plt.savefig(HERE / 'figures' / f'energy_{tag}.png', dpi=300)
    print(f"figures/energy_{tag}.png written; results_table.md appended")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'L4')
