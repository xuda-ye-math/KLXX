#!/usr/bin/env python
"""Multimodal dihedral-marginal figure for a hetero molecule (default glycerol).

Reconstructs each trained flow (forward KL and KL+X) from its saved per-stage state_dicts,
generates samples via compose_pushforward, maps them to Cartesian (u.xi_to_cartesian), and
computes the SAME proper torsions as the MD reference -- then plots the clearly-multimodal
torsions: MD reference vs forward KL vs KL+X_mu+X_mix.

Whitening is reproduced by a fresh short MD (bond/angle stats are stiff -> reproducible; torsions
are raw). Device default CPU (GPU busy with the training campaign); pass --device cuda after it ends.
Reference MD default = short (matches training); pass --ref_frames / --ref_stride for a converged GPU run.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import parmed as pmd
import openmm as mm
from openmm import app, unit

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Molecular_BG")
from zflows_md.flow import NCSF
from zflows_md.boltzmann import build, compose_pushforward, bridge, validation_update
from zflows_md.utils import compute_ESS_log, resample, langevin, set_ess_metric
from zflows_md.dihedral import proper_torsions, dihedral, multimodality
from zflows_md.plot.dihedrals import fab_marginals
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

LOG = None                                                   # set in main() to a file in the molecule folder


def log(m):
    """Timestamped status line -> stdout (flushed) + the molecule-folder log file (if configured)."""
    line = f"[{datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    if LOG is not None:
        open(LOG, "a").write(line + "\n")


def short_md(prmtop, crd, n_frames, stride, platform, equil=20000):
    s = pmd.load_file(prmtop, xyz=crd)
    system = s.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
    sim = app.Simulation(s.topology, system, integ, mm.Platform.getPlatformByName(platform))
    sim.context.setPositions(s.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    if equil:
        sim.step(equil)
    fr = []
    for _ in range(n_frames):
        sim.step(stride)
        fr.append(sim.context.getState(getPositions=True).getPositions(asNumpy=True)
                  .value_in_unit(unit.nanometer))
    return np.array(fr)


def flow_proper(method, folder, B, dev, n_gen, bins, transforms, hidden, mu_b, mu_a, quads):
    """Reconstruct the `method` flow, generate n_gen samples, rebuild Cartesian from the
    MD-mean bonds/angles (correct geometry, whitening-robust) + the flow's raw torsions,
    and return the named proper torsions (deg), shape (n_gen, n_quads)."""
    data = torch.load(os.path.join(folder, f"data_{method}.pth"), weights_only=False)
    state_dicts = [s["state_dict"] for s in data["stages"]]
    flow = NCSF(a=B["a"].tolist(), b=B["b"].tolist(), bins=bins, transforms=transforms,
                hidden_features=hidden).to(dev)
    flow.zeros()
    F_inv = flow.t()
    F_inv._inv_ladj_fn = F_inv.inv.call_and_ladj
    y, _ = compose_pushforward(flow, F_inv, state_dicts, B["u0"], B["u"], n_gen, dev)
    tors = y[:, B["tor_start"]:]                                # raw torsions (correct)
    N = tors.shape[0]
    z = torch.cat([mu_b.expand(N, -1), mu_a.expand(N, -1), tors], dim=1)
    cart = B["ic"].to_cartesian(z)[0].detach().cpu().numpy()    # (N, M, 3) original atom order
    return np.stack([dihedral(cart, q) for q in quads], axis=-1)


FULLNAME = {"adp": "alanine dipeptide"}                            # folder abbreviation -> full molecule name in titles


def flow_proper_smc(folder, B, dev, NV, cfg, e_min, e_max, mu_b, mu_a, quads, method="klxx"):
    """FAITHFUL BG inference: replay the SMC chain exactly as run_boltzmann -- per stage
    (a) validation reweight through G_k^{-1}, (b) resample, (c) Langevin on U_k at the SOFT cap
    e(t_{k-1}), then (d) the MC SHARPENING e(t_{k-1})->e(t_k) reweight + Langevin at the
    SHARP cap. The returned Y is the BG's actual deliverable at the final cap e_max -- NOT the bare
    flow proposal that compose_pushforward gives (which sits one rung soft, at e(t_{K-1})).
    Returns proper torsions (deg), shape (NV, n_quads), and prints per-stage val/sharpen ESS."""
    data = torch.load(os.path.join(folder, f"data_{method}.pth"), weights_only=False)
    stages = data["stages"]
    u0, u, wrap, ts = B["u0"], B["u"], B["wrap"], B["tor_start"]
    ms, mi = cfg["mc_step"], cfg["mc_iters"]
    drop = cfg.get("drop", 0.0)                                   # mirror the run's resample drop -> no replay collapse at clash stages
    max_ref = cfg.get("max_refresh", 0)                           # per stage: refresh this many times, keep the BEST-ESS attempt
    set_ess_metric(cfg.get("ess_metric", "raw"), cfg.get("ess_metric_c", 2.5))   # replay ESS on the run's metric
    e_anneal = e_min is not None and e_max is not None and float(e_max) > float(e_min)
    r_min, r_max = cfg.get("r_min"), cfg.get("r_max")          # r_floor anneal (match run_boltzmann)
    r_anneal = r_min is not None and r_max is not None and float(r_max) > float(r_min)
    amode = cfg.get("anneal_mode", "geometric")               # "geometric" (default) or "arithmetic" (linear in t)
    _e_of = lambda t: (float(e_min) + (float(e_max) - float(e_min)) * float(t)) if amode == "arithmetic" else float(e_min) * (float(e_max) / float(e_min)) ** float(t)
    _r_of = lambda t: (float(r_max) + (float(r_min) - float(r_max)) * float(t)) if amode == "arithmetic" else (float(r_max) * (float(r_min) / float(r_max)) ** float(t) if r_anneal else None)
    def _set_reg(t):
        if e_anneal: u.set_regularization(_e_of(t))
        if r_anneal: u.set_r_floor(_r_of(t))
    anneal = e_anneal or r_anneal
    use_flow = "state_dict" in stages[0]                          # ASMC (data_asmc.pth) carries NO flow -> pure SMC
    if use_flow:
        flow = NCSF(a=B["a"].tolist(), b=B["b"].tolist(), bins=cfg["bins"],
                    transforms=cfg["transforms"], hidden_features=list(cfg["hidden"])).to(dev)
        flow.zeros(); F_inv = flow.t(); F_inv.enable_inv_ladj(mode="default")
    Y = u0.samples(NV).to(dev); tp = 0.0; stage_dump = []
    for s in stages:
        tk = s["t"]
        if anneal:
            _set_reg(tp)                                          # soft cap (e + r_floor): flow validation + Langevin
        up, un = bridge(u0, u, tp), bridge(u0, u, tk)
        if use_flow:
            flow.load_state_dict({k: v.to(dev) for k, v in s["state_dict"].items()})
            yt, lw, ve = validation_update(F_inv, Y, up, un, chunk=NV, drop=drop)
            best_ve, best_yt, best_lw = ve, yt, lw                # refresh (resample+Langevin on u_prev) max_ref times; keep BEST-ESS
            Yr = Y
            for _r in range(max_ref):
                Yr = resample(Yr, (lw - lw.max()).exp(), drop=drop)
                Yr = wrap(langevin(Yr, up, step=ms, iters=mi, taming=0.0))
                yt, lw, ve = validation_update(F_inv, Yr, up, un, chunk=NV, drop=drop)
                if ve > best_ve:
                    best_ve, best_yt, best_lw = ve, yt, lw
            ve, yt, lw = best_ve, best_yt, best_lw
        else:                                                     # ASMC: no flow -> pure DIRECT bridge reweight (no inverse)
            yt, lw = Y, up.eval(Y) - un.eval(Y); ve = compute_ESS_log(lw, drop=drop).item()
        Y = resample(yt, (lw - lw.max()).exp(), drop=drop)
        Y = wrap(langevin(Y, un, step=ms, iters=mi, taming=0.0))
        sess = None
        if anneal:                                                # MC sharpening to the SHARP cap e(tk) + r_floor(tk)
            ua = u.eval(Y); _set_reg(tk); ub = u.eval(Y)
            lws = -tk * (ub - ua); sess = compute_ESS_log(lws, drop=drop).item()
            Y = resample(Y, (lws - lws.max()).exp(), drop=drop)
            Y = wrap(langevin(Y, bridge(u0, u, tk), step=ms, iters=mi, taming=0.0))
        stage_dump.append(Y.detach().cpu().numpy())               # save per-stage N_VALID samples (reuse/analysis)
        log(f"    [SMC replay {method}] t={tk:.3f}  best_val_ess={ve:.3f}  "
            f"sharpen_ess={'%.3f' % sess if sess is not None else 'n/a'}  "
            f"e {_e_of(tp):.0f}->{_e_of(tk):.0f}")
        tp = tk
    np.savez(os.path.join(folder, f"replay_stages_{method}.npz"),
             samples=np.stack(stage_dump), ts=np.array([s["t"] for s in stages]))   # per-stage samples for reuse
    tors = Y[:, ts:]                                              # final Y torsions, at e_max
    z = torch.cat([mu_b.expand(tors.shape[0], -1), mu_a.expand(tors.shape[0], -1), tors], dim=1)
    cart = B["ic"].to_cartesian(z)[0].detach().cpu().numpy()
    return np.stack([dihedral(cart, q) for q in quads], axis=-1)


def two_line_plot(folder, name, d, bg, md, labels, kl_bg=None, symmetrize=True):
    """Torsion marginals vs the annealed SMC reference (grey fill; the ensemble/resampling SMC ground
    truth). bg = the X-regularized+delta BG (red); optionally kl_bg = the forward-KL BG (blue) for a
    method comparison. Both BG runs share the same sharpening schedule, so the legend
    distinguishes only the loss. Achiral target -> p(phi)=p(-phi), so the densities are +-phi symmetrized."""
    bg, md = np.asarray(bg, float), np.asarray(md, float)
    has_kl = kl_bg is not None
    if has_kl:
        kl_bg = np.asarray(kl_bg, float)
    n = bg.shape[1]
    grid = np.linspace(-np.pi, np.pi, 400)
    BLUE, RED = "#1F77B4", "#D62728"                            # forward KL (blue) ; KL+X+X delta = best (red)

    def kde(a):
        a = np.asarray(a, float).ravel(); a = a[np.isfinite(a)] * (np.pi / 180.0)   # degrees -> radians
        if symmetrize:
            a = np.concatenate([a, -a])                          # achiral: p(phi)=p(-phi)
        t = np.concatenate([a - 2 * np.pi, a, a + 2 * np.pi])    # periodic wrap
        return gaussian_kde(t, bw_method=0.08)(grid) * 3.0

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0), squeeze=False)   # less wide (toward 4:3)
    for i in range(n):
        ax = axes[0][i]
        km, kx = kde(md[:, i]), kde(bg[:, i])
        ax.fill_between(grid, km, color="0.6", alpha=0.45, lw=0, zorder=1)
        ymax = max(km.max(), kx.max())
        ax.plot(grid, kx, color=RED, lw=1.4, alpha=0.95, zorder=3)         # KL+X+X delta (red, solid)
        if has_kl:
            kk = kde(kl_bg[:, i]); ymax = max(ymax, kk.max())
            ax.plot(grid, kk, color=BLUE, lw=1.3, ls=(0, (5, 3)), alpha=0.95, zorder=4)  # forward KL (blue, dashed, on top)
        ax.set_title(labels[i]); ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.set_ylim(0, ymax * 1.15)                           # density on 0; headroom avoids clipping the omega peaks
        ax.set_yticks([]); ax.set_xlabel("dihedral (rad)")
        if i == 0:
            ax.set_ylabel("density")
    handles = [Patch(fc="0.6", alpha=0.45, label="annealed SMC reference")]
    if has_kl:
        handles.append(Line2D([0], [0], color=BLUE, lw=1.6, ls=(0, (5, 3)), label="BG: forward KL"))
    handles.append(Line2D([0], [0], color=RED, lw=1.6,
                          label=r"BG: forward KL$+\mathrm{X}_\mu+\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ ($\delta$-reweighted)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.suptitle(rf"{FULLNAME.get(name, name)} ($d={d}$) — torsion marginals: BG vs annealed SMC reference", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.985))
    out = os.path.join(folder, "dihedrals.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    log(f"  wrote {out}  ({n} torsions, {'3-line compare' if has_kl else 'BG vs MD'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="glycerol")
    ap.add_argument("--d", type=int, default=36)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--platform", default="CPU")             # openmm reference-MD platform
    ap.add_argument("--n_gen", type=int, default=5000)
    ap.add_argument("--ref_frames", type=int, default=2000)
    ap.add_argument("--ref_stride", type=int, default=200)
    ap.add_argument("--top", type=int, default=4)            # plot the top-K multimodal torsions
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--klxx_only", action="store_true")     # plot only KL+X vs MD (skip forward-KL)
    ap.add_argument("--three_line", action="store_true")    # best-BG vs a long, converged, SAVED MD reference
    ap.add_argument("--nv", type=int, default=50000)        # SMC-replay particle count (three_line)
    ap.add_argument("--ref_walkers", type=int, default=200000)   # MD-reference Langevin walkers (use max VRAM)
    ap.add_argument("--ref_minutes", type=float, default=20.0)   # MD-reference wall-clock budget (long => converged)
    ap.add_argument("--refresh_md", action="store_true")        # recompute + overwrite the saved md_reference.npz
    ap.add_argument("--compare", action="store_true")           # also overlay the forward-KL BG (kl_sharpen, blue)
    ap.add_argument("--replot", action="store_true")            # re-plot dihedrals.png from saved torsion_data.npz (no build, no replay)
    a = ap.parse_args()
    if a.smoke:
        a.n_gen, a.ref_frames, a.ref_stride = 200, 200, 100
    dev = torch.device(a.device)
    folder = os.path.join(REPO, f"{a.name}_{a.d}d")
    global LOG
    LOG = os.path.join(folder, "marginals.log"); open(LOG, "w").close()
    import shutil; shutil.copy(os.path.abspath(__file__), os.path.join(folder, os.path.basename(__file__)))  # snapshot into the molecule folder (like run.py)
    cfg = json.load(open(os.path.join(folder, "config.json")))
    prm = os.path.join(REPO, "tests", "data", cfg.get("prmtop", f"{a.name}.prmtop"))  # config-driven (name may != prmtop basename, e.g. adp)
    crd = prm[: -len(".prmtop")] + ".rst7"          # rst7 shares the prmtop basename
    bins, transforms, hidden = cfg["bins"], cfg["transforms"], list(cfg["hidden"])
    log(f"START [{a.name}] device={dev} n_gen={a.n_gen} ref={a.ref_frames}x{a.ref_stride} "
        f"NCSF bins={bins} transforms={transforms} hidden={hidden} "
        f"three_line={a.three_line} klxx_only={a.klxx_only} compare={a.compare} smoke={a.smoke}")

    if a.replot:                                                # re-plot from saved torsion_data.npz (no build, no SMC replay)
        z = np.load(os.path.join(folder, "torsion_data.npz"), allow_pickle=True)
        kl = z["kl"]
        two_line_plot(folder, a.name, a.d, z["bg"], z["md"], list(z["labels"]),
                      kl_bg=(kl if kl.shape[0] > 0 else None))
        log("  re-plotted dihedrals.png from torsion_data.npz (no replay)"); return

    log("  reference MD (short whitening MD) ...")
    frames = short_md(prm, crd, a.ref_frames, a.ref_stride, a.platform, equil=0 if a.smoke else 20000)
    log(f"  reference MD done: {len(frames)} frames")
    B = build(prm, crd, md_frames=frames, T=300.0, device=dev, dtype=torch.float32,
              r_floor=cfg.get("r_floor", cfg.get("r_max", 0.2)), e_cap=cfg.get("e_cap", cfg.get("e_max")))
    for pot in (B["u0"], B["u"]):                            # potentials need enabling before eval
        pot.enable_grad(mode="default"); pot.enable_eval(mode="default")
    log("  build done: potentials + IC transform ready")

    ic = B["ic"]
    tor_start = B["tor_start"]
    M = ic.M
    # MD-mean bonds/angles (correct geometry) used to rebuild Cartesian for BOTH flow + MD,
    # so the comparison reflects only the torsions (robust to the bond/angle whitening).
    fr_t = torch.tensor(np.asarray(frames), dtype=torch.float32, device=dev)
    z_md, _ = ic.to_internal(fr_t)
    mu_b = z_md[:, :M - 1].mean(0, keepdim=True)
    mu_a = z_md[:, M - 1:tor_start].mean(0, keepdim=True)
    # named proper torsions from the prmtop
    proper = proper_torsions(prm)
    quads = [t[0] for t in proper]
    names = ["-".join(t[2]) for t in proper]
    # MD reference: mean bonds/angles + the MD torsions
    md_tors = z_md[:, tor_start:]
    Nf = md_tors.shape[0]
    z_ref = torch.cat([mu_b.expand(Nf, -1), mu_a.expand(Nf, -1), md_tors], dim=1)
    cart_ref = ic.to_cartesian(z_ref)[0].detach().cpu().numpy()
    ref = np.stack([dihedral(cart_ref, q) for q in quads], axis=-1)

    if a.three_line:        # correctness check: best BG run (klxx_delta_sharpen) vs the MD reference
        rcfg = torch.load(os.path.join(folder, "data_klxx_delta_sharpen.pth"),
                          weights_only=False)["config"]                # the run's own cap-anneal range
        e_min, e_max = rcfg.get("e_min"), rcfg.get("e_max", rcfg.get("e_cap"))
        # --- reference: the identity-flow annealed SMC (ASMC; per-stage sharpening, cap annealed), replayed.
        # The ensemble/resampling SMC reaches the modes (incl. the high-barrier omega trans); a bare uniform-init
        # Langevin gets stuck behind the barrier and manufactures spurious modes, so the SMC is the trustworthy
        # ground truth (verified: a Langevin re-initialized from the SMC ensemble stays trans). ---
        log("  reference: identity-flow annealed SMC (ASMC) via pure-SMC replay ...")
        md_all = flow_proper_smc(folder, B, dev, a.nv, cfg, e_min, e_max, mu_b, mu_a, quads, method="asmc")
        log(f"  ASMC reference: {md_all.shape[0]} samples")

        log("  BG inference: klxx_delta_sharpen (best) via SMC replay with sharpening ...")
        bg = flow_proper_smc(folder, B, dev, a.nv, cfg, e_min, e_max, mu_b, mu_a, quads,
                             method="klxx_delta_sharpen")
        kl_bg = None
        if a.compare:                                          # overlay the forward-KL BG (same sharpening schedule)
            log("  BG inference: kl_sharpen (forward KL) via SMC replay with sharpening ...")
            kl_bg = flow_proper_smc(folder, B, dev, a.nv, cfg, e_min, e_max, mu_b, mu_a, quads,
                                    method="kl_sharpen")
        # auto-select the clearly-multimodal torsions from the ASMC reference (any molecule):
        # heavy-atom backbone first, dedup equivalent torsions, take the top --top by +-phi-symmetrised balance.
        def canon(p): s = p.split("-"); return min(p, "-".join(s[::-1]))
        def heavy(p): s = p.split("-"); return s[0] != "H" and s[-1] != "H"
        def mm(i): return multimodality(np.concatenate([md_all[:, i], -md_all[:, i]]))[0]
        seen = {}
        for i in range(len(names)):
            bal = mm(i)
            if bal < 0.3:
                continue
            c = canon(names[i])
            if c not in seen or bal > seen[c][1]:
                seen[c] = (i, bal)
        cands = sorted(seen.values(), key=lambda x: (not heavy(names[x[0]]), -x[1]))   # heavy-atom first, then balance
        keep = [i for i, _ in cands][:a.top] or sorted(range(len(names)), key=lambda i: -mm(i))[:a.top]
        log(f"  multimodal torsions: {[(names[i], round(mm(i), 2)) for i in keep]}")
        two_line_plot(folder, a.name, a.d, bg[:, keep], md_all[:, keep], [names[i] for i in keep],
                      kl_bg=(kl_bg[:, keep] if kl_bg is not None else None))
        np.savez(os.path.join(folder, "torsion_data.npz"),
                 bg=bg[:, keep], md=md_all[:, keep], labels=np.array([names[i] for i in keep]),
                 kl=(kl_bg[:, keep] if kl_bg is not None else np.zeros((0, len(keep)))))
        log(f"DONE -> {os.path.join(folder, 'dihedrals.png')}"); return

    log("  flow inference: KL+X ...")
    klxx = flow_proper("klxx", folder, B, dev, a.n_gen, bins, transforms, hidden, mu_b, mu_a, quads)

    if a.klxx_only:
        kl = None
    else:
        log("  flow inference: forward KL ...")
        kl = flow_proper("kl", folder, B, dev, a.n_gen, bins, transforms, hidden, mu_b, mu_a, quads)

    # selection: comp-physics-favored (heavy-atom, no terminal H) + multimodal, deduped by
    # canonical pattern so the 3 equivalent hydroxyl rotations collapse to one
    def canon(p): s = p.split("-"); return min(p, "-".join(s[::-1]))
    def heavy(p): s = p.split("-"); return s[0] != "H" and s[-1] != "H"
    seen = {}
    for i in range(len(names)):
        bal = multimodality(ref[:, i])[0]
        if bal <= 0.3:
            continue
        c = canon(names[i])
        if c not in seen or bal > seen[c][1]:
            seen[c] = (i, bal)
    cands = sorted(seen.values(), key=lambda x: (not heavy(names[x[0]]), -x[1]))   # heavy-atom first
    keep = [i for i, _ in cands][:a.top]
    if not keep:
        keep = sorted(range(len(names)), key=lambda i: -multimodality(ref[:, i])[0])[:a.top]
    log(f"  plotting BAT torsions: {[names[i] for i in keep]}")
    fab_marginals(folder, a.name, a.d, klxx[:, keep], None if kl is None else kl[:, keep],
                  ref[:, keep], labels=[names[i] for i in keep], kind="torsion")
    save = dict(klxx=klxx[:, keep], ref=ref[:, keep], labels=np.array([names[i] for i in keep]))
    if kl is not None:
        save["kl"] = kl[:, keep]
    np.savez(os.path.join(folder, "torsion_data.npz"), **save)
    log(f"DONE -> {os.path.join(folder, 'dihedrals.png')}")


if __name__ == "__main__":
    main()
