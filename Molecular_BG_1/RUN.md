# Reproducing the results (zflows_md — 36 / 48 / 60 d)

Reproduction guide for the three molecular Boltzmann-generator targets: **glycerol** (`d = 36`),
**diethanolamine** (`d = 48`), and **alanine dipeptide** (`d = 60`, vacuum). Environment setup is in
[PYTHON.md](../PYTHON.md). Activate the env first (`conda activate zflows` — required for `torch.compile`),
then run each molecule's `train.py` from inside its own folder. MD topology/coordinate inputs
(`prmtop`/`rst7`) are loaded from the shared **`zflows_md/data`** package folder.

## The driver

Every run is driven by a self-contained per-molecule **`train.py`** that reads the molecule's **read-only**
`config.json` directly — no central driver, no `bgconfig` template indirection. Each run writes `data_<TAG>.pth`
and a live, timestamped `status_<TAG>.log` next to `train.py`. `TAG` matches the original `hetero_bg`
ablation: `asmc`, `asmc_raw`, `kl_sharpen`, `kl_raw`, `klxx_sharpen`, `klxx_raw`, `klxx_delta_sharpen`, `klxx_delta_raw`.

```bash
cd glycerol_36d            # or diethanolamine_48d / adp_60d
python train.py --method klxx          # {klxx | kl | asmc};  add --smoke for a fast end-to-end check
```

Flags:
- **`--method {klxx|kl|asmc}`** — `klxx` = X-regularized forward KL (KL + X_mu + X_(mu_hat+nu_bar)/2), the
  deliverable; by default it uses the δ-reweighted quench-and-temper pool with **δ taken from `config.json`**.
  `kl` = bare forward KL; `asmc` = no-flow identity-flow annealed-SMC reference.
- **`--raw`** — fixed cap = `e_max`, no cap/floor anneal, no per-stage sharpening (TAG `_raw`); the default is
  the *sharpening* schedule (TAG `_sharpen`).
- **`--no-delta`** — klxx only: drop the δ-QT pool → plain `klxx_sharpen`/`klxx_raw` instead of `klxx_delta_*`.
- **`--smoke`** — tiny end-to-end run for a fast sanity check; writes **no** `data_<TAG>.pth` (logs to
  `status_<TAG>.smoke.log`), so it never touches the committed paper files. `--n_valid/--n_pool/--n_batch/--steps`
  override the per-run sizes.

All hyperparameters (cap-anneal `e_min`/`e_max` = 100/200, `delta` = 0.1, `r_floor`, `ess_metric`, and the
adaptive-ladder knobs) come from each molecule's `config.json`. The 60d config uses the new flat schema; the
36d/48d configs use the legacy schema (`d`/`lam`/`e_cap`, omitting `max_stages`/`max_retry`/`release_cache`),
which `train.py` reads back with the same defaults the old `bgconfig` applied (max_stages=25, max_retry=8,
release_cache=True). The headline metric is `F = ∏_k (1/ESS_val_k)(1/ESS_sharp_k)` — the Monte-Carlo
error-propagation factor through both per-stage importance reweights (smaller is better; `F = 1` ideal).

> With `--raw` and `--no-delta`, `train.py` reproduces **all eight** original `hetero_bg` ablation rows
> (asmc/kl/klxx × sharpen/raw, and klxx with/without δ) — the full glycerol table below — so nothing the paper
> reports is left unreproducible.

## Boltzmann-generator runs (the result tables)

Run from inside each molecule folder, **one at a time** (confirm the GPU is free with `nvidia-smi` before the
next). The deliverable's achieved `F` is in the comment.

### Glycerol (`d = 36`) — full raw + sharpening ablation (8 rows)
```bash
cd glycerol_36d
python train.py --method asmc                   # asmc                F = 463   (no-flow reference)
python train.py --method kl                     # kl_sharpen          F = 44.0  (forward KL)
python train.py --method klxx --no-delta         # klxx_sharpen        F = 22.5
python train.py --method klxx                   # klxx_delta_sharpen  F = 14.2  (deliverable)
python train.py --method asmc --raw             # asmc_raw            F = 636
python train.py --method kl   --raw             # kl_raw              F = 40.9
python train.py --method klxx --no-delta --raw   # klxx_raw            F = 22.0
python train.py --method klxx --raw             # klxx_delta_raw      F = 16.6
```

### Diethanolamine (`d = 48`)
```bash
cd diethanolamine_48d
python train.py --method asmc            # F = 1455  (no-flow reference)
python train.py --method kl              # F = 92.6  (forward KL)
python train.py --method klxx            # deliverable (F = 35.9)
```

### Alanine dipeptide (`d = 60`)
```bash
cd adp_60d
python train.py --method asmc            # F = 129.7 (no-flow reference)
python train.py --method klxx            # deliverable (F = 67.95)
```

Each run writes `data_<TAG>.pth` (the per-stage records — `val_ess` / `sharpen_ess` arrays from which
`F = ∏_k (1/ESS_val_k)(1/ESS_sharp_k)` is computed, plus the flow state-dicts for the non-`asmc` methods,
used by the figure scripts below) and a live `status_<TAG>.log`. The `results_table.{md,csv}` already in each
folder are the summary tables from the original runs (the simplified `train.py` does not regenerate them).

## Figures

The figure scripts read the saved `data_<TAG>.pth` — no retraining. Run from the `Molecular_BG/` folder. The
per-molecule figure scripts default to glycerol, so pass `--name`/`--d` explicitly for diethanolamine and
alanine dipeptide.

| figure | script | glycerol command |
|---|---|---|
| ladder (per-stage ESS arcs) | `plot_ablation.py` | `python glycerol_36d/plot_ablation.py glycerol_36d` |
| conformers (structures) | `conformer_figure.py` | `python glycerol_36d/conformer_figure.py` |
| dihedral marginals vs MD | `multimodal_figure.py` | `python glycerol_36d/multimodal_figure.py --name glycerol --d 36 --device cuda` |

Diethanolamine — the same three:
```bash
python diethanolamine_48d/plot_ablation.py diethanolamine_48d
python diethanolamine_48d/conformer_figure.py --name diethanolamine --d 48
python diethanolamine_48d/multimodal_figure.py --name diethanolamine --d 48 --device cuda
```

Alanine dipeptide:
```bash
python adp_60d/conformer_figure.py                                  # conformers.png (L/D enantiomers, chiral-Cα wedge marker)
python adp_60d/multimodal_figure.py --name adp --d 60 --device cuda # dihedrals.png (vs converged-MD reference)
python adp_60d/ess_history.py                                       # ess_history.png (3x4 per-stage ESS history)
python adp_60d/ramachandran.py [--replot]                          # ramachandran.png (phi-psi + MLE L/D dots)
```

Notes:
- The converged-MD reference in the dihedral / Ramachandran figures is a long batched Langevin (`short_md`,
  ~2×10^5 walkers); raise `--ref_frames` for a tighter reference.
- `ramachandran.py` caches its φ/ψ samples to `ramachandran_data.npz`; `--replot` re-renders the heatmap
  (colours, MLE dots, labels) instantly without the SMC replay.

## Folder layout

```
Molecular_BG/
├── RUN.md                                    # this file
│
├── glycerol_36d/                             # glycerol, d = 36
│   ├── config.json                           # read-only per-molecule input (hyperparameters)
│   ├── plot_ablation.py                      # ladder.png: per-stage ESS arcs, raw vs sharpening panels
│   ├── conformer_figure.py                   # conformers.png: gauche−/trans/gauche+ ball-and-stick
│   ├── multimodal_figure.py                  # dihedrals.png: torsion marginals, BG vs annealed SMC
│   ├── data_asmc.pth                         # saved run: identity-flow annealed SMC reference (sharpen)
│   ├── data_asmc_raw.pth                     # saved run: annealed SMC, fixed-cap (raw)
│   ├── data_kl_sharpen.pth                   # saved run: forward KL, sharpening schedule
│   ├── data_kl_raw.pth                       # saved run: forward KL, fixed-cap (raw)
│   ├── data_klxx_sharpen.pth                 # saved run: KL+X, sharpening schedule
│   ├── data_klxx_raw.pth                     # saved run: KL+X, fixed-cap (raw)
│   ├── data_klxx_delta_sharpen.pth           # saved run: KL+X δ-reweighted, sharpening (deliverable)
│   ├── data_klxx_delta_raw.pth               # saved run: KL+X δ-reweighted, fixed-cap (raw)
│   ├── results_table.md                      # per-TAG headline F table (human-readable)
│   ├── results_table.csv                     # same table, machine-readable
│   ├── status.log                            # timestamped training log (klxx_delta_sharpen run)
│   ├── status_asmc.log                       # timestamped training log (asmc run)
│   ├── status_asmc_raw.log                   # timestamped training log (asmc_raw run)
│   ├── replay_stages_asmc.npz                # per-stage SMC-replay samples (asmc)
│   ├── replay_stages_kl_sharpen.npz          # per-stage SMC-replay samples (kl_sharpen)
│   ├── replay_stages_klxx_delta_sharpen.npz  # per-stage SMC-replay samples (klxx_delta_sharpen)
│   ├── torsion_data.npz                      # cached torsion arrays for dihedrals.png re-plot
│   ├── ladder.png                            # figure: per-stage ESS arcs (ablation)
│   ├── conformers.png                        # figure: combined 1×3 conformer panel
│   ├── conformer_gauche_minus.png            # figure: individual gauche− conformer
│   ├── conformer_gauche_minus.pdb            # structure: gauche− aligned coordinates
│   ├── conformer_trans.png                   # figure: individual trans conformer
│   ├── conformer_trans.pdb                   # structure: trans aligned coordinates
│   ├── conformer_gauche_plus.png             # figure: individual gauche+ conformer
│   ├── conformer_gauche_plus.pdb             # structure: gauche+ aligned coordinates
│   └── dihedrals.png                         # figure: torsion marginals vs annealed SMC
│
├── diethanolamine_48d/                       # diethanolamine, d = 48
│   ├── config.json                           # read-only per-molecule input (hyperparameters)
│   ├── plot_ablation.py                      # ladder.png: per-stage ESS arcs
│   ├── conformer_figure.py                   # conformers.png: gauche−/trans/gauche+ ball-and-stick
│   ├── multimodal_figure.py                  # dihedrals.png: torsion marginals, BG vs annealed SMC
│   ├── data_asmc.pth                         # saved run: annealed SMC reference
│   ├── data_kl_sharpen.pth                   # saved run: forward KL, sharpening schedule
│   ├── data_klxx_delta_sharpen.pth           # saved run: KL+X δ-reweighted, sharpening (deliverable)
│   ├── results_table.md                      # per-TAG headline F table (human-readable)
│   ├── results_table.csv                     # same table, machine-readable
│   ├── status.log                            # timestamped training log (klxx_delta_sharpen run)
│   ├── status_asmc.log                       # timestamped training log (asmc run)
│   ├── replay_stages_asmc.npz                # per-stage SMC-replay samples (asmc)
│   ├── replay_stages_kl_sharpen.npz          # per-stage SMC-replay samples (kl_sharpen)
│   ├── replay_stages_klxx_delta_sharpen.npz  # per-stage SMC-replay samples (klxx_delta_sharpen)
│   ├── torsion_data.npz                      # cached torsion arrays for dihedrals.png re-plot
│   ├── ladder.png                            # figure: per-stage ESS arcs
│   ├── conformers.png                        # figure: combined 1×3 conformer panel
│   ├── conformer_gauche_minus.png / .pdb     # figure + structure: gauche− conformer
│   ├── conformer_trans.png / .pdb            # figure + structure: trans conformer
│   ├── conformer_gauche_plus.png / .pdb      # figure + structure: gauche+ conformer
│   ├── dihedrals.png                         # figure: torsion marginals vs annealed SMC
│   └── multimodal_figure.log                 # log for the most recent multimodal_figure.py run
│
└── adp_60d/                                  # alanine dipeptide, d = 60
    ├── config.json                           # read-only per-molecule input (hyperparameters)
    ├── conformer_figure.py                   # conformers.png: L/D enantiomers with chiral-Cα wedge marker
    ├── multimodal_figure.py                  # dihedrals.png: torsion marginals, BG vs annealed SMC
    ├── ess_history.py                        # ess_history.png: 3×4 per-stage ESS training history
    ├── ramachandran.py                       # ramachandran.png: φ–ψ free-energy heatmap + MLE L/D dots
    ├── data_asmc.pth                         # saved run: annealed SMC reference
    ├── data_klxx_delta_sharpen.pth           # saved run: KL+X δ-reweighted, sharpening (deliverable)
    ├── results_table.md                      # per-TAG headline F table (human-readable)
    ├── results_table.csv                     # same table, machine-readable
    ├── status.log                            # timestamped training log (klxx_delta_sharpen run)
    ├── status_asmc.log                       # timestamped training log (asmc run)
    ├── replay_stages_asmc.npz                # per-stage SMC-replay samples (asmc)
    ├── replay_stages_klxx_delta_sharpen.npz  # per-stage SMC-replay samples (klxx_delta_sharpen)
    ├── torsion_data.npz                      # cached torsion arrays for dihedrals.png re-plot
    ├── ramachandran_data.npz                 # cached φ/ψ samples for ramachandran.png re-plot
    ├── conformers.png                        # figure: L and D enantiomer structures
    ├── conformer_L.png / conformer_D.png     # figure: individual L / D enantiomer panels
    ├── dihedrals.png                         # figure: torsion marginals vs annealed SMC
    ├── ess_history.png                       # figure: per-stage ESS training history (3×4 grid)
    ├── ramachandran.png                      # figure: Ramachandran φ–ψ landscape (BG vs SMC reference)
    ├── multimodal_figure.stdout              # log for the most recent multimodal_figure.py run
    └── ramachandran.log                      # log for the most recent ramachandran.py run
```

## Outputs & write-up

Each molecule folder collects: `config.json` (read-only input), `data_<TAG>.pth`, `results_table.{md,csv}`
(the `F` table), `status.log`, and the figures (`ladder.png` / `conformers.png` / `dihedrals.png`, plus
`ess_history.png` / `ramachandran.png` for ADP). The combined paper section that gathers all three molecules
(tables + figures) is in **`Paper/main.tex`** Section 6 (compiled in `Paper/main.pdf`).
