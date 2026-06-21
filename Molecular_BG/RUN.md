# Reproducing the results (zflows_md — 36 / 48 / 60 d)

Reproduction guide for the three molecular Boltzmann-generator targets: **glycerol** (`d = 36`),
**diethanolamine** (`d = 48`), and **alanine dipeptide** (`d = 60`, vacuum). Environment setup is in
[PYTHON.md](../PYTHON.md); run everything from the **`Molecular_BG/`** folder with the `zflows` (or
`torch`) env active. MD topology/coordinate inputs live in `tests/data/` (relative to `Molecular_BG/`).

## The driver

Every run is driven by the **`zflows_md.bg.hetero_bg`** module with a per-molecule, **read-only** `config.json` as the
single source of truth for every hyperparameter. `zflows_md.bg.bgconfig.load_config(d, name)` reads the
authoritative `<name>_<d>d/config.json` (e.g. `glycerol_36d/config.json`); the central
`zflows_md/bg/config.json` is a bootstrap-only template, never read at runtime. Each run writes
`data_<TAG>.pth`, appends to `results_table.{md,csv}`, and logs to `status.log` in the molecule folder, with
`TAG = {kl|klxx}{_delta}{_sharpen|_raw}` (or `asmc{_raw}` for the reference).

Flags:
- **`--method {klxx|kl|asmc}`** — `klxx` = X-regularized forward KL (KL + X_mu + X_(mu_hat+nu_bar)/2);
  `kl` = bare forward KL; `asmc` = no-flow identity-flow annealed-SMC reference.
- **`--delta`** — klxx only: the δ-reweighted quench-and-temper pool (δ from config).
- **`--raw`** — fixed cap, no anneal / no per-stage sharpening (the default is the *sharpening* schedule).

The cap-anneal range (`e_min`/`e_max` = 100/200), `delta` (0.1), `r_floor`, `ess_metric`, and the
adaptive-ladder knobs all come from each molecule's `config.json`. The headline metric is
`F = ∏_k (1/ESS_val_k)(1/ESS_sharp_k)` — the Monte-Carlo error-propagation factor through both per-stage
importance reweights (smaller is better; `F = 1` ideal).

## Boltzmann-generator runs (the result tables)

Each command reproduces one row of a result table (achieved `F` in the comment). Run one at a time; confirm
the GPU is free (`nvidia-smi`) before launching the next.

### Glycerol (`d = 36`) — full raw + sharpening ablation (8 rows)
```bash
G="--prmtop tests/data/glycerol.prmtop --crd tests/data/glycerol.rst7 --name glycerol"
python -m zflows_md.bg.hetero_bg $G --method asmc                 # F = 463    (no-flow reference, sharpen)
python -m zflows_md.bg.hetero_bg $G --method kl                   # F = 44.0   (forward KL)
python -m zflows_md.bg.hetero_bg $G --method klxx                 # F = 22.5
python -m zflows_md.bg.hetero_bg $G --method klxx --delta         # F = 14.2   (deliverable)
python -m zflows_md.bg.hetero_bg $G --method asmc --raw           # F = 636
python -m zflows_md.bg.hetero_bg $G --method kl   --raw           # F = 40.9
python -m zflows_md.bg.hetero_bg $G --method klxx --raw           # F = 22.0
python -m zflows_md.bg.hetero_bg $G --method klxx --delta --raw   # F = 16.6
```

### Diethanolamine (`d = 48`)
```bash
D="--prmtop tests/data/diethanolamine.prmtop --crd tests/data/diethanolamine.rst7 --name diethanolamine"
python -m zflows_md.bg.hetero_bg $D --method asmc                 # F = 1455   (no-flow reference)
python -m zflows_md.bg.hetero_bg $D --method kl                   # F = 92.6   (forward KL)
python -m zflows_md.bg.hetero_bg $D --method klxx --delta         # F = 35.9   (deliverable)
```

### Alanine dipeptide (`d = 60`)
```bash
A="--prmtop tests/data/alanine_dipeptide.prmtop --crd tests/data/alanine_dipeptide.rst7 --name adp"
python -m zflows_md.bg.hetero_bg $A --method asmc                 # F = 129.7  (no-flow reference)
python -m zflows_md.bg.hetero_bg $A --method klxx --delta         # F = 67.95  (deliverable)
```

After each run, `results_table.md` in the molecule folder holds the per-stage ESS table and the headline `F`
for that `TAG`; `data_<TAG>.pth` holds the per-stage records (and the flow state-dicts for the non-`asmc`
methods) used by the figure scripts below.

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
