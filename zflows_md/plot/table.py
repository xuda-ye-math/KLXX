#!/usr/bin/env python
"""Standalone results-table writer: regenerate results_table.{md,csv} for a molecule folder
from its data_<method>.pth files, INDEPENDENT of the training run -- so a crashed or
ladder-failed run still gets tabulated (hetero_bg.py writes the table inline only if it
reaches the end). Per-stage validation ESS and sharpening ESS are the headline columns.

Usage:
    python table.py                 # default folder: glycerol_36d
    python table.py glycerol_36d    # folder name under the repo root
    python table.py /abs/path/to/folder
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                     # zflows-md root

METHODS = [                                                       # (data_<tag>.pth, label)
    ("klxx_sharpen",       r"forward KL$+X_\mu+X_{(\hat\mu+\bar\nu)/2}$ (sharpen)"),
    ("kl_sharpen",         "forward KL (sharpen)"),
    ("klxx_raw",           r"forward KL$+X_\mu+X_{(\hat\mu+\bar\nu)/2}$ (raw)"),
    ("kl_raw",             "forward KL (raw)"),
    ("klxx_delta_sharpen", r"forward KL$+X_\mu+X_{(\hat\mu+\bar\nu)/2}$ (sharpen, $\delta$-reweighted)"),
    ("klxx_delta_raw",     r"forward KL$+X_\mu+X_{(\hat\mu+\bar\nu)/2}$ (raw, $\delta$-reweighted)"),
]


def load(p):
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def _sharp(s):                                                    # per-stage sharpening ESS ('-' if skipped)
    v = s.get("sharpen_ess")
    return None if v is None else float(v)


def build(folder):
    base = os.path.basename(folder.rstrip("/"))
    name, dim = base.rsplit("_", 1)
    dim = dim[:-1] if dim.endswith("d") else dim                  # '36d' -> '36'
    data = {m: load(os.path.join(folder, f"data_{m}.pth")) for m, _ in METHODS}
    cfg = next((dd["config"] for dd in data.values() if dd), {})

    # ---- CSV: one row per (method, stage), with per-stage validation + sharpening ESS ----
    with open(os.path.join(folder, "results_table.csv"), "w") as f:
        f.write("method,stage,t_k,smc_ess,validation_ess,sharpening_ess,n_shrink\n")
        for m, _ in METHODS:
            dd = data[m]
            if not dd:
                continue
            for i, s in enumerate(dd["stages"]):
                sh = _sharp(s)
                f.write(f"{m},{i+1},{s['t']:.4f},{s.get('smc_ess', float('nan')):.4f},"
                        f"{s['val_ess']:.4f},{'nan' if sh is None else f'{sh:.4f}'},"
                        f"{s.get('n_shrink', 0)}\n")

    # ---- Markdown: per-method header (complete/K/F/wall) + a per-stage ESS table ----
    with open(os.path.join(folder, "results_table.md"), "w") as f:
        f.write(f"# {name} (d={dim}) — adaptive-temperature Boltzmann generator: 6-method ablation\n\n")
        f.write(f"NCSF bins={cfg.get('bins')} transforms={cfg.get('transforms')} "
                f"hidden={tuple(cfg.get('hidden', []))} | pool={cfg.get('n_pool')} "
                f"batch={cfg.get('n_batch')} valid={cfg.get('n_valid')} | r_floor={cfg.get('r_floor')} | "
                f"cap anneal e_min→e_max={cfg.get('e_min')}→{cfg.get('e_max')}, "
                f"raw fixed e_cap={cfg.get('e_cap', cfg.get('e_max'))}\n\n")
        f.write("Headline **F = ∏ₖ (1/val_essₖ)(1/sharpen_essₖ)** — the MC error-propagation factor through "
                "*both* per-stage reweights (flow validation + MC sharpening); smaller is better, F=1 ideal. "
                "For raw runs the sharpening step is skipped (sharpen ESS `-`, factor 1).\n")
        for m, lab in METHODS:
            dd = data[m]
            if not dd:
                f.write(f"\n*{lab}: not run yet.*\n")
                continue
            F = 1.0
            for s in dd["stages"]:                                # F = prod_k (1/val_ess)(1/sharpen_ess):
                F *= 1.0 / max(s["val_ess"], 1e-6)                #   both per-stage importance reweights
                se = s.get("sharpen_ess")                         #   (flow validation + MC sharpening);
                if se is not None:                                #   sharpen_ess None -> factor 1 (raw runs).
                    F *= 1.0 / max(se, 1e-6)
            wall = dd.get("wall_s")
            f.write(f"\n## {lab} — complete={dd['complete']}, K={len(dd['stages'])}, **F={F:.2f}**"
                    + (f", wall={wall/60:.0f} min" if wall else "") + "\n\n")
            f.write("| stage | t_k | SMC ESS | validation ESS | sharpening ESS |\n"
                    "|---|---|---|---|---|\n")
            for i, s in enumerate(dd["stages"]):
                sh = _sharp(s)
                f.write(f"| {i+1} | {s['t']:.3f} | {s.get('smc_ess', float('nan')):.3f} | "
                        f"**{s['val_ess']:.3f}** | {'-' if sh is None else f'{sh:.3f}'} |\n")
    n = sum(1 for m, _ in METHODS if data[m])
    print(f"wrote {folder}/results_table.{{md,csv}}  ({n}/{len(METHODS)} methods present)", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "glycerol_36d"
    folder = arg if os.path.isabs(arg) else os.path.join(REPO, arg)
    build(folder)
