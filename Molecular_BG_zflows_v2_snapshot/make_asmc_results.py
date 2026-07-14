#!/usr/bin/env python
"""Output the FAB-accurate annealed-SMC results properly: (1) the Ramachandran R-plot
(ASMC vs the amber96-implicit MD reference, FAB Fig.19 style) and (2) the ASMC-only ESS
table (per SMC stage: t_k, SMC ESS_min, accept/bridge ESS, shrinks).

Reads the sharpening-free FAB ASMC run (data_asmc*_fab_nosharpen.pth -> stages + phi/psi)
and the MD reference (md_implicit.npz). Writes NEW files; never overwrites inputs.

    python make_asmc_results.py [data_asmc_fab_nosharpen.pth]

Live status log -> stdout AND make_asmc_results.log. No tqdm.
"""
import os, sys, glob, csv
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "make_asmc_results.log"); open(LOG, "w").close()
def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

BINS, SIGMA = 100, 1.0          # FAB Fig.19 style: 100x100, light periodic smoothing


def find_data():
    if len(sys.argv) > 1:
        return sys.argv[1]
    cands = sorted(glob.glob(os.path.join(HERE, "data_asmc*_fab_nosharpen.pth")))
    if cands:
        return cands[-1]
    raise FileNotFoundError("no data_asmc*_fab_nosharpen.pth found; pass the .pth path explicitly")


def hist2d(phi, psi):
    """FAB Fig.19 density: normalized 2D histogram over (phi,psi) in [-180,180] deg, log color,
    light periodic Gaussian smoothing. Returns H [psi, phi]."""
    edges = np.linspace(-180, 180, BINS + 1)
    H, _, _ = np.histogram2d(phi, psi, bins=[edges, edges], density=True)
    return gaussian_filter(H.T, SIGMA, mode="wrap")


def main():
    data_path = find_data()
    log(f"START make_asmc_results  data={os.path.basename(data_path)}")
    d = torch.load(data_path, map_location="cpu", weights_only=False)
    stages = d["stages"]
    phi, psi = np.asarray(d["phi"], float), np.asarray(d["psi"], float)
    N = len(phi)
    reg = d.get("reg_mode", "?"); ecut = d.get("energy_cut"); emax = d.get("energy_max")
    log(f"  ASMC: K={len(stages)} stages  N={N}  complete={d.get('complete')}  "
        f"reg={reg} energy_cut={ecut} energy_max={emax}  frac(phi>0)={float((phi>0).mean()):.3f}")

    # ---- MD reference (amber96 implicit, L-alanine) ----
    md_f = os.path.join(HERE, "md_implicit.npz")
    md = np.load(md_f) if os.path.exists(md_f) else None
    if md is not None:
        mphi, mpsi = np.asarray(md["phi"], float), np.asarray(md["psi"], float)
        log(f"  MD reference: {len(mphi)} frames  t={float(md['t_ns']) if 't_ns' in md.files else '?'} ns")
    else:
        log("  MD reference md_implicit.npz not found -> plotting ASMC panels only")

    # ---- fold the D enantiomer into L (mirror: phi>0 -> (-phi,-psi)) for the direct L comparison ----
    fold = phi > 0
    fphi = np.where(fold, -phi, phi); fpsi = np.where(fold, -psi, psi)

    # ---- R-plot ----
    tk = [-180, -90, 0, 90, 180]
    panels = [("annealed SMC (FAB-accurate, both enantiomers)", phi, psi),
              ("annealed SMC folded to L (mirror D->L)", fphi, fpsi)]
    if md is not None:
        panels.append(("amber96 implicit MD reference (L)", mphi, mpsi))
    Hs = [hist2d(p, q) for _, p, q in panels]
    vmax = max(H.max() for H in Hs); vmin = vmax * 1e-4
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (lab, _, _), H in zip(axes, panels, Hs):
        im = ax.imshow(np.clip(H, vmin, None), origin="lower", extent=(-180, 180, -180, 180),
                       cmap="viridis", norm=norm, aspect="equal", interpolation="bilinear")
        ax.set_xticks(tk); ax.set_yticks(tk); ax.tick_params(labelsize=9)
        ax.set_xlabel(r"$\phi$ (deg)", fontsize=11); ax.set_title(lab, fontsize=10.5, pad=6)
    axes[0].set_ylabel(r"$\psi$ (deg)", fontsize=11)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.02, shrink=0.85)
    cb.set_label("density (log scale)", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.suptitle(r"alanine dipeptide ($d=60$) Ramachandran — FAB-accurate annealed SMC vs MD reference",
                 y=0.98, fontsize=12)
    out_png = os.path.join(HERE, "ramachandran_asmc_fab.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    log(f"  R-plot -> {os.path.basename(out_png)}")

    # ---- ASMC-only ESS table ----
    rows = []
    for k, s in enumerate(stages, 1):
        rows.append(dict(stage=k, t_k=round(s["t"], 4),
                         smc_ess_min=round(s["smc_ess"], 4),
                         accept_ess=round(s["val_ess"], 4),
                         shrinks=s.get("n_shrink", 0)))
    ve = np.array([r["accept_ess"] for r in rows])
    # CSV
    csv_path = os.path.join(HERE, "ess_table_asmc.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "t_k", "smc_ess_min", "accept_ess", "shrinks"])
        w.writeheader(); w.writerows(rows)
    # Markdown
    md_path = os.path.join(HERE, "ess_table_asmc.md")
    with open(md_path, "w") as f:
        f.write(f"# ASMC ESS table (FAB-accurate, sharpening-free)\n\n")
        f.write(f"- target: amber96 + amber96_obc (implicit), true singular potential, boltzgen linlog "
                f"(energy_cut={ecut:g}, energy_max={emax:g})\n")
        f.write(f"- stages K={len(stages)}, particles N={N}, complete={d.get('complete')}, "
                f"wall={d.get('wall_s', float('nan')):.0f}s\n")
        f.write(f"- accept/bridge ESS: min={ve.min():.3f}, mean={ve.mean():.3f}, "
                f"floor(validation_tau)=0.5\n\n")
        f.write("| stage | t_k | SMC ESS_min | accept ESS | shrinks |\n")
        f.write("|------:|----:|-----------:|----------:|-------:|\n")
        for r in rows:
            f.write(f"| {r['stage']} | {r['t_k']:.4f} | {r['smc_ess_min']:.3f} | "
                    f"{r['accept_ess']:.3f} | {r['shrinks']} |\n")
    log(f"  ESS table -> {os.path.basename(csv_path)}, {os.path.basename(md_path)}")

    # print the table to the log/stdout
    log(f"  ---- ASMC ESS table (K={len(stages)}, N={N}) ----")
    log(f"   {'stage':>5} {'t_k':>8} {'SMC ESS':>9} {'accept ESS':>11} {'shrinks':>8}")
    for r in rows:
        log(f"   {r['stage']:>5} {r['t_k']:>8.4f} {r['smc_ess_min']:>9.3f} "
            f"{r['accept_ess']:>11.3f} {r['shrinks']:>8}")
    log(f"  accept ESS: min={ve.min():.3f} mean={ve.mean():.3f}   (target floor 0.5)")
    log(f"DONE -> {os.path.basename(out_png)}, ess_table_asmc.(csv|md)")


if __name__ == "__main__":
    main()
