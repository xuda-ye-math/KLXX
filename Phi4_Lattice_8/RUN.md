# Phi4_Lattice_8 — 8×8 phi^4 lattice (d = 64)

Tilted phi^4 scalar field theory on an 8×8 periodic lattice (d = 64 sites), broken Z2 phase.
Parameters `KAPPA = 0.40`, `LAMBDA = 0.50`, `H = 0.0144` are frozen by the pilot scan,
giving a magnetization barrier of ~10 kT and a tilted minority-phase weight p(m > 0) ≈ 0.139.
The experiment demonstrates that forward KL and KL+X_mu collapse onto a single vacuum while
reporting high ESS (the "fake ESS" phenomenon), whereas KL+X_mu+X_hat_mu and KL+X_mu+X_mix
repair coverage and recover the correct phase weight.

## How to run

Activate the environment and run all scripts from the **`Phi4_Lattice_8/`** folder (scripts
resolve paths relative to `__file__`):

```bash
conda activate zflows
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/Phi4_Lattice_8
```

### Step 1 — Pilot scan (freeze kappa / h; build PT reference)

```bash
# Parallel-tempering MALA kappa scan and tilt; writes phi4_reference.pth and pilot_results.md
python pilot.py

# Optional: faster but coarser (fewer chains and steps)
python pilot.py --quick
```

Outputs: `phi4_reference.pth`, `pilot_results.md`, `figures/pilot_scan.png`, `pilot_status.log`.
Skip if `phi4_reference.pth` already exists and the parameters in `parameters.py` are unchanged.

### Step 2 — Train all four methods (default seed 0)

```bash
# Train KL, KL+X_mu, KL+X_mu+X_hat_mu, KL+X_mu+X_mix in sequence; writes data.pth
python train.py
```

Additional seeds used in the paper (each writes its own data file):

```bash
# Seed 1 — inherits frozen QT set from data.pth; writes data_seed1.pth
python train.py --seed 1

# Seed 2 — writes data_seed2.pth
python train.py --seed 2
```

Other flags:

- `--methods KL,KL+X_mu` — run a comma-separated subset; appends to the existing data file
- `--steps N` — override the default 2000 Adam steps
- `--no-compile` — disable `torch.compile` (useful for debugging)
- `--qt-skew F` — subsample the frozen QT oracle to fraction F of m > 0 samples (robustness test)

Progress is written to `train_status.log` and stdout (timestamped, flushed per step).

### Step 3 — Figures and results table

```bash
# Reads data.pth + phi4_reference.pth; writes figures/fig_methods.png, figures/fig_background.png,
# figures/fig_ess.png, results_table.md, results_table.csv
python plot_results.py
```

`plot_results.py` takes no arguments and does not modify any data files.

## Figures

| file | script | description |
|---|---|---|
| `figures/pilot_scan.png` | `pilot.py` | kappa scan and frozen-tilt magnetization histogram |
| `figures/fig_background.png` | `plot_results.py` | per-site double well, PT reference p(m), and m < 0 / m > 0 lattice field snapshots |
| `figures/fig_methods.png` | `plot_results.py` | per-method reweighted p(m) vs PT reference — **paper figure** (included directly by `Paper/main.tex`) |
| `figures/fig_ess.png` | `plot_results.py` | training ESS history for all four methods |

## Folder tree

```
Phi4_Lattice_8/
├── parameters.py          # frozen hyperparameters (KAPPA, H, flow arch, training)
├── core.py                # Phi4 potential, loss_KL, loss_X, quench_and_temper
├── pilot.py               # Step 1: PT-MALA kappa/h scan
├── train.py               # Step 2: train the four methods
├── plot_results.py        # Step 3: figures and tables
│
├── phi4_reference.pth     # PT referee metadata and samples (written by pilot.py)
├── pilot_results.md       # kappa scan table and frozen parameters (written by pilot.py)
├── data.pth               # flow state dicts, weights, ESS histories, seed 0 (written by train.py)
├── data_seed1.pth         # same, seed 1 (written by train.py --seed 1)
├── data_seed2.pth         # same, seed 2 (written by train.py --seed 2)
├── pilot_status.log       # timestamped pilot log (regenerated on demand by pilot.py; not committed)
├── train_status.log       # timestamped training log (written by train.py)
│
├── results_table.md       # per-method final ESS / p(m>0) / Delta F table (written by plot_results.py)
├── results_table.csv      # same, CSV format (written by plot_results.py)
│
└── figures/
    ├── fig_methods.png    # paper figure (load-bearing — included by Paper/main.tex; written by plot_results.py)
    ├── fig_background.png # per-site double well, PT reference p(m), lattice snapshots (regenerated on demand by plot_results.py; not committed)
    ├── fig_ess.png        # training ESS history for all four methods (regenerated on demand by plot_results.py; not committed)
    └── pilot_scan.png     # kappa scan and frozen-tilt histogram (regenerated on demand by pilot.py; not committed)
```

## Outputs

A complete run writes:

- `phi4_reference.pth` — PT referee metadata and samples (from `pilot.py`)
- `pilot_results.md` — kappa scan table and frozen parameters (from `pilot.py`)
- `pilot_status.log` — live pilot log (from `pilot.py`)
- `data.pth` / `data_seed{N}.pth` — flow state dicts, importance weights, ESS histories (from `train.py`)
- `train_status.log` — live training log (from `train.py`)
- `results_table.md` / `results_table.csv` — final ESS, p(m > 0), and ΔF per method (from `plot_results.py`)
- `figures/fig_methods.png`, `figures/fig_background.png`, `figures/fig_ess.png`, `figures/pilot_scan.png` — all figures

Quantitative results (ESS values, p_+ estimates, ΔF) are reported in **`Paper/main.tex`**.
