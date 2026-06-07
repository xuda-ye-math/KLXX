# pyright: reportArgumentType=false
"""PT-MALA referee for the screened-Poisson posterior at FULL dimension d_full.

Parallel tempering over likelihood temper t (U_t = 0.5|xi|^2 + t*Phi_full),
geometric ladder, MALA within each rung with per-rung step auto-tuning during
burn-in, neighbor swaps each iteration, round-trip mixing certificate.
The cold-rung census on the (shift n, sign s) symmetry lattice gives the TRUE
well weights the trained generators are judged against.

    ~/.envs/torch/bin/python pilot.py                      # all sweep sigmas
    ~/.envs/torch/bin/python pilot.py --sigmas 0.01        # one sigma

Writes referee_o{sigma}.pth + referee.md (appended per sigma).
"""
import argparse
import math
import time
from pathlib import Path
from datetime import datetime

import torch

import parameters as p
import potential as pot

HERE = Path(__file__).resolve().parent
STATUS = HERE / 'train_status.log'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(STATUS, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


def pt_run(B, n_steps=6000, n_chains=128, n_rungs=20, t_min=1e-4, seed=0):
    """PT-MALA on U_t = 0.5|xi|^2 + t*Phi_full. Returns (cold samples after
    burn-in, roundtrips, per-rung acceptance, per-rung swap rate)."""
    torch.manual_seed(seed)
    d = B['d_full']
    u_full = B['u_full'].enable_grad()          # grad gives xi + grad(Phi)
    phi_fn = B['u_full'].misfit                 # untempered Phi

    ladder = torch.cat([torch.zeros(1, device=device),
                        torch.logspace(math.log10(t_min), 0.0, n_rungs - 1,
                                       device=device)])
    K = len(ladder)
    t = ladder.view(K, 1)
    xi = torch.randn(K, n_chains, d, device=device)
    step = torch.full((K, 1), 5e-3, device=device)        # per-rung, auto-tuned

    def phi_of(x):
        return phi_fn(x.view(-1, d)).view(K, n_chains)

    def grad_ut(x):
        g = u_full.grad(x.view(-1, d)).clone().view(K, n_chains, d)
        xr = x  # grad(U_full) = xi + grad(Phi); U_t-grad = xi + t*grad(Phi)
        return xr + t.unsqueeze(-1) * (g - xr)

    lab = torch.arange(K, device=device).view(K, 1).expand(K, n_chains).clone()
    hit_top = torch.zeros(K, n_chains, dtype=torch.bool, device=device)
    roundtrips = 0
    acc_acc = torch.zeros(K, device=device)
    swap_acc = torch.zeros(K - 1, device=device)
    swap_cnt = torch.zeros(K - 1, device=device)
    cold = []
    ph_cur = phi_of(xi)

    def ut(ph, x):
        return 0.5 * (x * x).sum(-1) + t * ph

    burn = n_steps // 3
    n_acc_window = torch.zeros(K, device=device)
    for it in range(n_steps):
        # MALA per rung (vectorized over rungs x chains)
        g = grad_ut(xi)
        s = step.unsqueeze(-1)
        prop = xi - s * g + (2 * s).sqrt() * torch.randn_like(xi)
        ph_prop = phi_of(prop)
        gp = grad_ut(prop)
        fwd = (prop - xi + s * g).square().sum(-1) / (4 * step)
        bwd = (xi - prop + s * gp).square().sum(-1) / (4 * step)
        log_acc = (ut(ph_cur, xi) - ut(ph_prop, prop)) - bwd + fwd
        acc = torch.rand(K, n_chains, device=device).log() < log_acc
        xi = torch.where(acc.unsqueeze(-1), prop, xi)
        ph_cur = torch.where(acc, ph_prop, ph_cur)
        n_acc_window += acc.float().mean(dim=1)
        acc_acc += acc.float().mean(dim=1)
        # hot rung (t=0): exact prior refresh
        xi[0] = torch.randn(n_chains, d, device=device)
        ph_cur[0] = phi_of(xi)[0]
        # step auto-tune during burn-in
        if it < burn and (it + 1) % 100 == 0:
            rate = n_acc_window / 100.0
            step[rate.unsqueeze(1) > 0.7] *= 1.5
            step[rate.unsqueeze(1) < 0.3] *= 0.6
            n_acc_window.zero_()
        # neighbor swaps (alternating parity)
        for k0 in range(it % 2, K - 1, 2):
            d_log = (ladder[k0 + 1] - ladder[k0]) * (ph_cur[k0 + 1] - ph_cur[k0])
            sw = torch.rand(n_chains, device=device).log() < d_log
            swap_acc[k0] += sw.float().mean()
            swap_cnt[k0] += 1
            for arr in (xi, ph_cur, lab, hit_top):
                a, b = arr[k0].clone(), arr[k0 + 1].clone()
                arr[k0][sw], arr[k0 + 1][sw] = b[sw], a[sw]
        # roundtrip count: visit top then bottom
        hit_top[-1] |= True
        done = hit_top[0].clone()
        roundtrips += int(done.sum().item())
        hit_top[0][done] = False
        if it >= burn and (it % 5 == 0):
            cold.append(xi[-1].clone())
    return (torch.cat(cold), roundtrips, (acc_acc / n_steps).cpu(),
            (swap_acc / swap_cnt.clamp(min=1)).cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigmas', type=str, default='0.01,0.015,0.02,0.025')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--chains', type=int, default=128)
    ap.add_argument('--rungs', type=int, default=20)
    args = ap.parse_args()
    p.N_SENSORS = int(1.5 * p.M_LOW ** 2 + 0.5)

    for sig in [float(s) for s in args.sigmas.split(',')]:
        p.SIGMA_OBS = sig
        torch.manual_seed(0)
        B = pot.build(p, device)
        d, std_all = B['d_full'], B['std_all'].to(device)
        log(f"##### REFEREE START o{sig:g} d={d} rungs={args.rungs} "
            f"chains={args.chains} steps={args.steps} #####")
        t0 = time.perf_counter()
        cold, rt, acc, swp = pt_run(B, n_steps=args.steps,
                                    n_chains=args.chains, n_rungs=args.rungs)
        wall = time.perf_counter() - t0
        # sym-label census on the cold chain
        tr_nc = B['xi_truth'][:d].to(device).clone(); tr_nc[0] = 0.0
        th0 = cold[:, 0] * std_all[0]
        n_lab = torch.round(th0 * p.ALPHA / (2.0 * math.pi)).long()
        s_lab = ((cold * tr_nc.unsqueeze(0)).sum(-1) > 0).long()
        pair = n_lab * 2 + s_lab
        uniq, cnt = pair.unique(return_counts=True)
        w = (cnt.float() / cnt.sum()).cpu()
        census = {}
        for u_, wi in zip(uniq.tolist(), w.tolist()):
            n_ = u_ // 2 if u_ >= 0 else -((-u_ + 1) // 2)
            census[(n_, u_ % 2)] = wi
        census = dict(sorted(census.items()))
        log(f"##### REFEREE DONE o{sig:g}: roundtrips={rt} "
            f"cold N={cold.shape[0]} wall={wall/60:.1f} min #####")
        log(f"referee wells o{sig:g}: " + str({k: round(v, 4)
                                               for k, v in census.items()}))
        torch.save(dict(sigma_obs=sig, census=census, roundtrips=rt,
                        cold=cold[::4].cpu(), acc=acc, swap=swp,
                        rungs=args.rungs, chains=args.chains,
                        steps=args.steps),
                   HERE / f'referee_o{sig:g}.pth')
        md = HERE / 'referee.md'
        if not md.exists():
            md.write_text("# PT-MALA referee: certified well weights\n\n")
        with open(md, 'a') as f:
            f.write(f"\n## sigma_obs = {sig:g} (roundtrips {rt}, "
                    f"{cold.shape[0]} cold samples, {wall/60:.1f} min)\n\n"
                    "| well (n, s) | weight |\n|---|---|\n")
            for k_, v in census.items():
                if v > 1e-4:
                    f.write(f"| {k_} | {v:.4f} |\n")
            f.write(f"\nacceptance per rung: "
                    f"{[round(a, 2) for a in acc.tolist()]}\n"
                    f"swap rate per pair: "
                    f"{[round(s_, 2) for s_ in swp.tolist()]}\n")
    log("wrote referee.md + referee_o*.pth")


if __name__ == '__main__':
    main()
