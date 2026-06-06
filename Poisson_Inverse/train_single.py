# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Single-stage smoke test (PLAN.md S2a): train ONE flow from the whitened
prior N(0, I_{d_low}) to the t-tempered low-mode posterior

    U_t = 0.5|xi|^2 + t * Phi_low      (the Algorithm-4 stage-1 bridge),

and observe the direct training ESS and the validation ESS. Cheapest
end-to-end signal that the whitened pipeline trains; calibrates t_safe.

    ~/.envs/torch/bin/python train_single.py --t 0.1  --method balance
    ~/.envs/torch/bin/python train_single.py --t 0.01 --method balance
    ~/.envs/torch/bin/python train_single.py --t 0.1  --method kl

Appends one row per run to single_stage.md; ESS curve -> figures/.
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.loss import loss_compile
from zflows.utils import suppress_warnings
from core import (fused_stage_loss, fused_kl_loss, bridge, train_stage,
                  validation_update, identity_wrap)
import parameters as p
import potential as pot

suppress_warnings()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
STATUS = HERE / 'train_status.log'


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(STATUS, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t', type=float, default=0.1)
    ap.add_argument('--m-low', type=int, default=4)
    ap.add_argument('--m-full', type=int, default=8)
    ap.add_argument('--method', choices=['balance', 'kl'], default='balance')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=1000)
    args = ap.parse_args()
    p.M_LOW, p.M_FULL = args.m_low, args.m_full

    NV, NP = 20000, 4000
    torch.manual_seed(0)
    B = pot.build(p, device)
    d = B['d_low']
    u0 = Gaussian([0.0] * d, [1.0] * d, device=device)
    u = B['u_low']                                    # 0.5|xi|^2 + Phi_low
    for q in (u0, u):
        q.enable_grad(mode="reduce-overhead")
        q.enable_eval(mode="default")

    flow = NSF(a=[-p.NSF_LIM] * d, b=[p.NSF_LIM] * d, bins=p.BINS,
               transforms=p.TRANSFORMS, hidden_features=p.HIDDEN).to(device)
    flow.zeros()
    F_inv = flow.t().enable_inv_ladj()

    tp_buf = torch.zeros((), device=device)            # t_{k-1} = 0
    tn_buf = torch.full((), args.t, device=device)     # t_k = t
    if args.method == 'balance':
        closs = loss_compile(fused_stage_loss, u0, u, tp_buf, tn_buf, flow.t(),
                             float(p.LAMBDA), args.batch)
    else:
        closs = loss_compile(fused_kl_loss, u0, u, tp_buf, tn_buf, flow.t())

    u_prev = u0
    u_next = bridge(u0, u, args.t)
    u_next.enable_grad(mode="reduce-overhead")
    u_next.enable_eval(mode="default")

    hat = None
    if args.method == 'balance':
        hat = pot.quench_and_temper(u_next, NP, d, 2.0, p.OPT_STEP, p.OPT_ITERS,
                                    p.MC_STEP, p.MC_ITERS, device)
        log(f"QT pool: {NP} samples, theta0 range "
            f"[{(hat[:, 0] * B['std_all'][0].to(device)).min():.2f}, "
            f"{(hat[:, 0] * B['std_all'][0].to(device)).max():.2f}]")

    Y = u0.samples(NV)
    tag = f"m{args.m_low}_t{args.t}_{args.method}"
    log(f"##### SINGLE-STAGE START {tag} d={d} steps={args.steps} "
        f"batch={args.batch} device={device} #####")
    t0 = time.perf_counter()
    ess_hist, ok = train_stage(
        flow, F_inv, closs, Y, u_prev, u_next, hat, steps=args.steps,
        batch=args.batch, lr=p.LR, mc_step=p.MC_STEP,
        rungs=p.SMC_RUNGS, rung_iters=p.SMC_RUNG_ITERS,
        wrap=identity_wrap, method=args.method,
        grad_clip=p.GRAD_CLIP, max_skip=p.MAX_SKIP,
        status=log, report_every=50)
    _, _, val_ess = validation_update(F_inv, Y, u_prev, u_next)

    # fine validation ESS (user rule: validate against the ACCURATE potential):
    # extend by identity on the whitened high modes and reweight with the
    # t-tempered FULL posterior U_t = 0.5|xi|^2 + t*Phi_full.
    d_full = B['d_full']
    u_t_full = B['make'](d_full, temper=args.t)
    with torch.no_grad():
        x = torch.randn(NV, d_full, device=device)
        y_low, ladj = F_inv.inv_ladj(x[:, :d])
        y_low, ladj = y_low.clone(), ladj.clone()
        y = torch.cat([y_low, x[:, d:]], dim=1)
        logw = (0.5 * (x * x).sum(-1) - u_t_full(y) + ladj)
    from zflows.utils import compute_ESS_log
    fine_ess = compute_ESS_log(logw).item()
    wall = time.perf_counter() - t0
    log(f"##### DONE {tag}: train_ok={ok} final direct ESS={ess_hist[-1]:.4f} "
        f"validation ESS (low)={val_ess:.4f} FINE validation ESS "
        f"(d={d_full})={fine_ess:.4f} wall={wall/60:.1f} min #####")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'figures').mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ess_hist, lw=0.8)
    ax.set_xlabel('step'); ax.set_ylabel('direct ESS'); ax.set_ylim(0, 1)
    ax.set_title(f'single stage {tag}: val ESS {val_ess:.3f}')
    plt.tight_layout()
    plt.savefig(HERE / 'figures' / f'single_{tag}.png', dpi=400,
                bbox_inches='tight', pad_inches=0.02)

    md = HERE / 'single_stage.md'
    if not md.exists():
        md.write_text("# Single-stage smoke results\n\n"
                      "| tag | d | t | method | steps | batch | train_ok | "
                      "final direct ESS | val ESS (low) | FINE val ESS | wall (min) |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(md, 'a') as f:
        f.write(f"| {tag} | {d} | {args.t} | {args.method} | {args.steps} | "
                f"{args.batch} | {ok} | {ess_hist[-1]:.4f} | {val_ess:.4f} | "
                f"{fine_ess:.4f} | {wall/60:.1f} |\n")
    torch.save(dict(tag=tag, ess_hist=ess_hist, val_ess=val_ess, fine_ess=fine_ess, ok=ok,
                    state_dict={k: v.cpu() for k, v in flow.state_dict().items()},
                    config=vars(args)),
               HERE / f'data_single_{tag}.pth')
    log(f"wrote single_stage.md, figures/single_{tag}.png, data_single_{tag}.pth")


if __name__ == '__main__':
    main()
