#!/usr/bin/env python
"""Regenerate the repo SUMMARY.md from the per-molecule data_*.pth.

Headline metric: the MC error PROPAGATION FACTOR through the sequential ladder,
    F = prod_k (1 / ESS_k) = 1 / prod_k ESS_k
i.e. the factor by which the importance-sampling estimator's variance is enlarged
relative to perfect iid sampling after the K annealing stages (smaller = better;
F=1 is ideal). KLXX (X-regularized) vs bare forward KL.
"""
import torch, os, glob, math

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Molecular_BG")


def load(p):
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def prop_factor(d):
    if d is None:
        return None
    stages = d.get("stages", [])
    if not stages:
        return None
    F = 1.0                                          # F = prod_k (1/val_ess)(1/sharpen_ess): both
    for s in stages:                                 # per-stage reweights; sharpen None -> factor 1 (raw)
        if "val_ess" not in s:
            continue
        F *= 1.0 / max(s["val_ess"], 1e-6)
        se = s.get("sharpen_ess")
        if se is not None:
            F *= 1.0 / max(se, 1e-6)
    return F


# ---- ESS-ladder figure (matches ../Log* style: serif + mathtext cm; method colors
# forward KL blue #1F77B4, KLXX/KL+X red #D62728) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "mathtext.fontset": "cm", "font.family": "serif",
})
C_KL, C_KLXX = "#1F77B4", "#D62728"   # forward KL (blue, top) ; KL+X (red, bottom)


def plot_ladder(name, d, b, k, outpath):
    """Poisson-subsection style (paper Fig. 12): t-axis 0->1, each accepted stage an arc
    -- forward KL above (blue), the X-regularized loss below (red); per-stage validation
    ESS on the apex; a cross at the final bridge a run failed to reach. Method key (with
    the run's F) in a clean bottom legend. X is set upright (\\mathrm), not math-italic."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.axhline(0, color="0.3", lw=1.0, zorder=1)

    def draw(dd, color, sign):
        if not dd:
            return
        stages = dd.get("stages", [])
        ts = [0.0] + [s["t"] for s in stages]
        for i, s in enumerate(stages):
            t0, t1 = ts[i], ts[i + 1]
            mid, r = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
            th = np.linspace(0, np.pi, 60)
            h = 0.18 + 1.0 * r                       # arc height grows with step width
            ax.plot(mid + r * np.cos(th), sign * h * np.sin(th), color=color, lw=2.0, zorder=3)
            lift = 0.04 + (0.12 if r < 0.06 and i % 2 == 1 else 0.0)   # destagger tight arcs
            ax.text(mid, sign * (h + lift), f"{s['val_ess']:.2f}", ha="center",
                    va="bottom" if sign > 0 else "top", color=color, fontsize=12, zorder=4)
        ax.plot(ts, [0] * len(ts), "o", color="0.2", ms=4, zorder=5)
        if not dd.get("complete", True):             # cross at the unreached final bridge
            ax.plot(1.0, 0.0, "x", color=color, ms=11, mew=2.5, zorder=6)

    draw(k, C_KL, +1)        # forward KL on top (blue)
    draw(b, C_KLXX, -1)      # KL+X on bottom (red)
    Fb, Fk = prop_factor(b), prop_factor(k)
    fk_s = f"{Fk:.2f}" if Fk else "--"; fb_s = f"{Fb:.2f}" if Fb else "--"
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(r"$t$ (temperature)")
    ax.set_title(rf"{name} ($d={d}$)")
    handles = [Line2D([0], [0], color=C_KL, lw=2.5, label="forward KL"),
               Line2D([0], [0], color=C_KLXX, lw=2.5,
                      label=r"KL$+\mathrm{X}_\mu+\mathrm{X}_{(\hat\mu+\bar\nu)/2}$")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    folders = sorted([f for f in glob.glob(os.path.join(ROOT, "*_*d"))
                      if os.path.isdir(f) and "zflows_md" not in f],
                     key=lambda f: int(os.path.basename(f).rsplit("_", 1)[1][:-1]))
    rows = []
    for f in folders:
        base = os.path.basename(f)
        d = int(base.rsplit("_", 1)[1][:-1]); name = base.rsplit("_", 1)[0]
        b = (load(f"{f}/data_klxx_sharpen.pth") or load(f"{f}/data_klxx.pth") or load(f"{f}/data_balance.pth")
             or load(f"{f}/data_adp_balance_B2k.pth"))   # klxx_sharpen (new) -> klxx -> balance (old) fallback
        k = load(f"{f}/data_kl_sharpen.pth") or load(f"{f}/data_kl.pth")
        rows.append((d, name, b, k))

    L = ["# zflows_md — Boltzmann-generator ESS results (KL+X_μ+X_mix vs forward KL)\n",
         "KL+X_μ+X_mix vs forward KL adaptive-temperature",
         "Boltzmann generators on whitened internal coordinates, built on the Amber force field.\n",
         "**Headline metric — MC error propagation factor**",
         "`F = prod_k (1/ESS_k) = 1 / prod_k ESS_k`: how much the importance-sampling",
         "estimator variance is enlarged vs perfect iid sampling after the K sequential",
         "annealing stages. **F = 1 is ideal; smaller is better.** (ESS_k = per-stage",
         "validation ESS, the acceptance-gate metric.) **F is only meaningful for a",
         "ladder that ran to t=1; an early-stopped run (final t<1) has a TRUNCATED F",
         "and must be re-run on the no-early-stop engine.**\n",
         "Each method spans three columns — **stage** (K, number of annealing stages), "
         "**time** (training wall-clock, minutes), and **factor** (F).\n"]
    def _final_t(dd):
        ld = dd.get("ladder", []) if dd else []
        return ld[-1] if ld else None
    def _wall(dd):                               # training wall-clock in minutes
        return f"{dd['wall_s']/60:.0f}" if (dd and dd.get("wall_s")) else "—"
    def _triple(dd):                             # (stage K, time min, factor F) per method
        if not dd or not dd.get("stages"):
            return ("—", "—", "—")
        K, w, ft = str(len(dd["stages"])), _wall(dd), _final_t(dd)
        if ft is not None and ft < 0.999:        # incomplete ladder -> F truncated
            return (K, w, f"t={ft:.2f}!")
        F = prop_factor(dd)
        return (K, w, f"{F:.2f}" if F else "—")
    L += ['<table>',
          '<thead><tr><th rowspan="2">molecule</th><th rowspan="2">d</th>'
          '<th colspan="3">forward KL</th>'
          '<th colspan="3">KL+X<sub>μ</sub>+X<sub>mix</sub></th></tr>',
          '<tr><th>stage</th><th>time (min)</th><th>factor</th>'
          '<th>stage</th><th>time (min)</th><th>factor</th></tr></thead><tbody>']
    for d, name, b, k in rows:
        sk, tk, fk = _triple(k)
        sb, tb, fb = _triple(b)
        L.append(f'<tr><td>{name}</td><td>{d}</td>'
                 f'<td>{sk}</td><td>{tk}</td><td>{fk}</td>'
                 f'<td>{sb}</td><td>{tb}</td><td>{fb}</td></tr>')
    L.append('</tbody></table>\n')

    L += ["\n## Detailed per-molecule results (full ESS tables + parameters)\n"]
    for d, name, b, k in rows:
        cfg = ((b or k) or {}).get("config", {})
        L.append(f"\n### {name} — d = {d}")
        if cfg:
            L.append(f"**Parameters:** NCSF bins={cfg.get('bins')} transforms={cfg.get('transforms')} "
                     f"hidden={tuple(cfg.get('hidden', []))} · pool={cfg.get('n_pool')} "
                     f"batch={cfg.get('n_batch')} N_VALID={cfg.get('n_valid')} · "
                     f"mc_step={cfg.get('mc_step')} opt/mc_iters={cfg.get('opt_iters')}/{cfg.get('mc_iters')} "
                     f"· compiled_inverse={cfg.get('compile_inv')} · T={cfg.get('temperature_K')}K\n")
        # both methods in ONE table: separate t_k + validation-ESS columns per method
        Fb, Fk = prop_factor(b), prop_factor(k)
        fb = f"{Fb:.2f}" if Fb is not None else "—"
        fk = f"{Fk:.2f}" if Fk is not None else "—"
        bst = b.get("stages", []) if b else []
        kst = k.get("stages", []) if k else []
        L.append(f"**KL+X_μ+X_mix**: complete={b.get('complete') if b else '—'}, "
                 f"K={len(bst) if b else '—'}, **F={fb}**, train {_wall(b)} min  ·  "
                 f"**forward KL**: complete={k.get('complete') if k else '—'}, "
                 f"K={len(kst) if k else '—'}, **F={fk}**, train {_wall(k)} min\n")
        if not bst and not kst:
            L.append("*not run yet.*\n"); continue
        # ESS-ladder figure (instead of a table): t-axis 0->1, forward KL arcs above
        # (blue), KL+X arcs below (red), per-step validation ESS on each arc.
        figrel = f"{name}_{d}d/ladder.png"
        try:
            if name != "glycerol":               # glycerol keeps its custom 6-method ablation ladder (ablation.py)
                plot_ladder(name, d, b, k, os.path.join(ROOT, figrel))
            L.append(f'<p align="center"><img src="{figrel}" width="600" alt="{name} ESS ladder"></p>\n')
        except Exception as e:
            L.append(f"*(figure render failed: {e})*\n")

    L += ["\n## Notes",
          "- Smaller F = less compounded Monte-Carlo error across the ladder; F=1 ideal.",
          "- Config (per molecule `config.json`): NCSF bins=16 transforms=6 hidden=(256,256);",
          "  pool 100k-150k, batch 8k-15k, N_VALID 500k-600k; MC_STEP 1e-4, OPT/MC iters",
          "  250/350-400, GRAD_CLIP 1e2, 50% skip tolerance; ladder runs to t=1 explicitly (no early-stop).",
          "- Single-seed runs on a 5070 Ti sweet spot (d=18-27 compiled inverse). d=30-48",
          "  (raw inverse) and d=60 (alanine dipeptide) deferred to a larger GPU.",
          "\n## Package validation (provenance)",
          "`Amber_Force_Field` energy & gradient vs **openmm**: <=3e-5 kcal/mol, <=1e-6",
          "kcal/(mol*A); forward/grad/eval/loss torch.compile-clean (graphs=1, breaks=0)",
          "in both modes, CUDA-graph capturable; compiled grad ~35x eager autograd;",
          "validated on 8-600 atoms incl. protein + water + ions. Scripts in `tests/`.\n"]
    open(os.path.join(ROOT, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("wrote SUMMARY.md")
    print("\n".join(L[8:12 + len(rows)]))


if __name__ == "__main__":
    main()
