#!/usr/bin/env python
"""Alanine-dipeptide Ramachandran (phi-psi free-energy heatmap), FAB-paper style: a dense viridis +
LogNorm 2D landscape with a shared colorbar (cf. Midgley et al. 2022, 'Ground truth | Flow | FAB').

Reference (left) = the identity-flow annealed SMC (ensemble/resampling SMC; the trustworthy ground truth
-- a bare uniform-init Langevin gets stuck behind the high omega barrier and is unreliable). Right = the
trained-flow BG (KL+X+X, delta-reweighted). Both obtained by SMC replay (no saved samples needed) and
cached to ramachandran_data.npz, so layout tweaks re-plot instantly with --replot (no 30-min replay).
phi = C(ACE)-N(ALA)-CA-C(ALA) = (4,6,8,14);  psi = N(ALA)-CA-C(ALA)-N(NME) = (6,8,14,16).

Usage: python ramachandran.py [--nv 100000 --bins 64 --sigma 1.1 --device cuda] [--replot]
"""
import os, sys, json, argparse
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE                                                  # run in-place: the script's own dir is the molecule folder
from zflows_md.bg.multimodal_scan import dihedral

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "font.family": "serif"})
PHI, PSI = (4, 6, 8, 14), (6, 8, 14, 16)
ap = argparse.ArgumentParser()
ap.add_argument("--nv", type=int, default=100000)           # SMC-replay particle count (density -> smooth landscape)
ap.add_argument("--bins", type=int, default=64)
ap.add_argument("--sigma", type=float, default=1.1)         # periodic Gaussian smoothing of the 2D histogram
ap.add_argument("--device", default="cuda")
ap.add_argument("--replot", action="store_true")           # re-plot from ramachandran_data.npz (no build, no replay)
a = ap.parse_args()
folder = HERE                                               # run in-place: the script's own dir is the molecule folder
cache = os.path.join(folder, "ramachandran_data.npz")
LOG = os.path.join(folder, "ramachandran.log"); open(LOG, "w").close()
def log(m): line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")

if a.replot and os.path.exists(cache):
    z = np.load(cache)
    ref_phi, ref_psi, bg_phi, bg_psi = z["ref_phi"], z["ref_psi"], z["bg_phi"], z["bg_psi"]
    log(f"re-plot from {os.path.basename(cache)} (no build, no replay)")
else:
    from zflows_md.boltzmann import build, bridge, validation_update
    from zflows_md.flow import NCSF
    from zflows_md.utils import resample, langevin
    from zflows_md.bg.multimodal_figure import short_md
    cfg = json.load(open(os.path.join(folder, "config.json")))
    prm = os.path.join(REPO, "tests", "data", cfg["prmtop"]); crd = prm[:-len(".prmtop")] + ".rst7"
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    log(f"START ramachandran (FAB-style): device={dev} nv={a.nv} bins={a.bins} sigma={a.sigma}")
    log("  short whitening MD (IC means) ...")
    frames = short_md(prm, crd, 200, 100, "CPU", equil=2000)
    rfloor = cfg.get("r_floor", cfg.get("r_max", 0.2))
    B = build(prm, crd, md_frames=frames, T=300.0, device=dev, dtype=torch.float32, r_floor=rfloor, e_cap=cfg["e_max"])
    u, u0, wrap = B["u"], B["u0"], B["wrap"]; ic, ts = B["ic"], B["tor_start"]; M = ic.M
    for pot in (u0, u):
        pot.enable_grad(mode="default"); pot.enable_eval(mode="default")
    fr = torch.tensor(np.asarray(frames), dtype=torch.float32, device=dev); zmd, _ = ic.to_internal(fr)
    mu_b = zmd[:, :M - 1].mean(0, keepdim=True); mu_a = zmd[:, M - 1:ts].mean(0, keepdim=True)
    e_min, e_max = cfg["e_min"], cfg["e_max"]; r_min, r_max = cfg.get("r_min"), cfg.get("r_max")
    amode = cfg.get("anneal_mode", "geometric")
    e_anneal = e_min is not None and e_max is not None and float(e_max) > float(e_min)
    r_anneal = r_min is not None and r_max is not None and float(r_max) > float(r_min)
    mc_step, mc_iters, drop = cfg["mc_step"], cfg["mc_iters"], cfg.get("drop", 0.0)
    def _e_of(t): return (float(e_min) + (float(e_max) - float(e_min)) * t) if amode == "arithmetic" else float(e_min) * (float(e_max) / float(e_min)) ** t
    def _r_of(t): return (float(r_max) + (float(r_min) - float(r_max)) * t) if amode == "arithmetic" else float(r_max) * (float(r_min) / float(r_max)) ** t
    def _set_reg(t):
        if e_anneal: u.set_regularization(_e_of(t))
        if r_anneal: u.set_r_floor(_r_of(t))

    def replay(data_path, label):
        """SMC replay to t=1 (flow if state_dict present, else pure direct-reweight SMC). Returns phi, psi (deg)."""
        data = torch.load(data_path, map_location=dev, weights_only=False); stages = data["stages"]
        use_flow = "state_dict" in stages[0]
        log(f"  {label}: {len(stages)} stages ({'flow' if use_flow else 'pure-SMC'})")
        if use_flow:
            rc = data["config"]
            flow = NCSF(a=B["a"].tolist(), b=B["b"].tolist(), bins=rc["bins"], transforms=rc["transforms"],
                        hidden_features=list(rc["hidden"])).to(dev)
            flow.zeros(); F_inv = flow.t(); F_inv._inv_ladj_fn = F_inv.inv.call_and_ladj
        Y = u0.samples(a.nv).to(dev); tp = 0.0
        for rs in stages:
            tk = rs["t"]; _set_reg(tp); up, un = bridge(u0, u, tp), bridge(u0, u, tk)
            if use_flow:
                flow.load_state_dict({k: v.to(dev) for k, v in rs["state_dict"].items()})
                yt, lw, _ = validation_update(F_inv, Y, up, un, drop=drop)
            else:
                yt, lw = Y, up.eval(Y) - un.eval(Y)                # pure SMC: direct bridge reweight
            Y = resample(yt, (lw - lw.max()).exp(), drop=drop)
            Y = wrap(langevin(Y, un, step=mc_step, iters=mc_iters, taming=0.0))
            if e_anneal or r_anneal:
                ua = u.eval(Y); _set_reg(tk); ub = u.eval(Y); lws = -tk * (ub - ua)
                Y = resample(Y, (lws - lws.max()).exp(), drop=drop)
                Y = wrap(langevin(Y, bridge(u0, u, tk), step=mc_step, iters=mc_iters, taming=0.0))
            tp = tk
        tors = Y[:, ts:]
        z = torch.cat([mu_b.expand(tors.shape[0], -1), mu_a.expand(tors.shape[0], -1), tors], dim=1)
        cart = ic.to_cartesian(z)[0].detach().cpu().numpy()
        log(f"  {label}: {tors.shape[0]} samples")
        return dihedral(cart, PHI), dihedral(cart, PSI)

    ref_phi, ref_psi = replay(os.path.join(folder, "data_asmc.pth"), "SMC reference")
    bg_phi, bg_psi = replay(os.path.join(folder, "data_klxx_delta_sharpen.pth"), "BG (klxx)")
    np.savez_compressed(cache, ref_phi=ref_phi, ref_psi=ref_psi, bg_phi=bg_phi, bg_psi=bg_psi)
    log(f"  cached phi/psi -> {os.path.basename(cache)}")

# ---- plot (shared with --replot) -------------------------------------------------------------------
def wrap180(x): return ((np.asarray(x, float) + 180.0) % 360.0) - 180.0
edges = np.linspace(-180, 180, a.bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
def Hmap(phi, psi):
    h, _, _ = np.histogram2d(wrap180(phi), wrap180(psi), bins=[edges, edges], density=True)
    return gaussian_filter(h.T, a.sigma, mode="wrap")          # periodic smooth -> FAB-like free-energy landscape
HR, HB = Hmap(ref_phi, ref_psi), Hmap(bg_phi, bg_psi)
vmax = max(HR.max(), HB.max()); vmin = vmax * 1e-4             # ~4 decades, like the FAB colorbar
norm = LogNorm(vmin=vmin, vmax=vmax)
PHI_g, PSI_g = np.meshgrid(centers, centers)                   # phi/psi value at each (psi-row, phi-col) cell of Hm
Lm = (PHI_g < 0) & (PSI_g > 0); Dm = (PHI_g > 0) & (PSI_g < 0) # dominant-conformer quadrant of each enantiomer
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
for ax, (lab, Hm) in zip(axes, [("annealed SMC reference", HR), ("Boltzmann generator (BG)", HB)]):
    im = ax.imshow(np.clip(Hm, vmin, None), origin="lower", extent=(-180, 180, -180, 180),
                   cmap="viridis", norm=norm, aspect="equal", interpolation="bilinear")
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=10)
    ax.set_xlabel(r"$\phi$ (deg)", fontsize=11.5); ax.set_title(lab, fontsize=11.5, pad=7)
    for mask, tag in [(Lm, "L"), (Dm, "D")]:                    # per-panel argmax inside each enantiomer's quadrant
        pj, fj = np.unravel_index(int(np.argmax(np.where(mask, Hm, -np.inf))), Hm.shape)
        px, py = float(centers[fj]), float(centers[pj])
        ax.plot(px, py, "o", color="red", ms=8, markeredgecolor="white", markeredgewidth=1.4, zorder=6)
        ax.annotate(tag, (px, py), textcoords="offset points", xytext=(9, 7), ha="center",
                    color="white", fontsize=13, fontweight="bold", zorder=7,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="black")])
axes[0].set_ylabel(r"$\psi$ (deg)", fontsize=11.5)
axes[1].tick_params(labelleft=False)                           # right panel: drop redundant y-tick labels (shared axis)
cb = fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.035, shrink=0.86, aspect=26)
cb.set_label("density (log scale)", fontsize=10, labelpad=10); cb.ax.tick_params(labelsize=9)
fig.suptitle(r"alanine dipeptide ($d=60$) — Ramachandran: BG vs annealed SMC reference",
             y=0.96, fontsize=12.5)                             # lowercase + dihedrals suptitle format; formula -> key below
mle_handle = Line2D([0], [0], marker="o", color="red", linestyle="none", markeredgecolor="white",
                    markeredgewidth=1.0, markersize=8,
                    label="maximum-probability (MLE) conformer of the L / D enantiomers")
fig.legend(handles=[mle_handle], loc="lower center", bbox_to_anchor=(0.5, -0.03), frameon=False,
           fontsize=10, handletextpad=0.4)                      # actual red-dot marker in the legend (not the words)
fig.text(0.5, -0.10, r"BG: forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ ($\delta$-reweighted)",
         ha="center", fontsize=10)                              # method key, below the legend
out = os.path.join(folder, "ramachandran.png")
fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
log(f"DONE -> {out}")
