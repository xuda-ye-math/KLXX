# Reproducing the results — tilted phi^4 lattice (L=6, d=36)

Single-flow importance-sampling benchmark on a 2D phi^4 scalar field on a 6×6 periodic
lattice (d = 36 sites, broken Z2 phase). A small external field H tilts the two vacua so
p(m>0) ≈ 0.13; the flow must recover the minority-phase weight. The experiment isolates the
fake-ESS-vs-coverage failure mode and demonstrates how the X-regularisation terms fix it.

## How to run

Activate the environment and run scripts from inside the `Phi4_Lattice_6/` folder:

```bash
conda activate zflows
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/Phi4_Lattice_6
```

### Step 1 — pilot scan (freeze lattice parameters)

Runs a parallel-tempering MALA scan over κ at h = 0, picks the κ with a 10 kT barrier,
tilts H to give ΔF ≈ 1 kT, and saves the frozen reference set.

```bash
# Full pilot (~40 k PT steps × 256 chains); writes pilot_results.md + phi4_reference.pth + figures/pilot_scan.png
python pilot.py

# Quick smoke-test (2 k steps × 64 chains, κ=0.40 only; same outputs, approximate values)
python pilot.py --quick
```

### Step 2 — train all four methods

Trains all four loss variants sequentially and appends results to `data.pth`.  Optional flags
run robustness variants that write to separate files.

```bash
# Default: trains KL, KL+X_mu, KL+X_mu+X_hat_mu, KL+X_mu+X_mix; writes data.pth
python train.py

# Train a subset of methods only (comma list, no spaces)
python train.py --methods KL,KL+X_mu

# Alternate random seed (writes data_seed1.pth, inheriting frozen QT set from data.pth)
python train.py --seed 1

# Alternate random seed 2 (writes data_seed2.pth)
python train.py --seed 2

# Skewed QT oracle: keep only 5 % of m>0 samples (writes data_skew05.pth)
python train.py --qt-skew 0.05

# Skewed QT oracle: keep only 25 % of m>0 samples (writes data_skew25.pth)
python train.py --qt-skew 0.25

# Change number of Adam steps (default: 2000)
python train.py --steps 4000
```

### Step 3 — figures and tables

Reads `data.pth` and `phi4_reference.pth`; writes all figures and summary tables. No data
files are modified.

```bash
# Writes figures/fig_background.png, figures/fig_methods.png, figures/fig_ess.png,
# results_table.md, results_table.csv
python plot_results.py
```

Progress logs are written to `train_status.log` (training) and `pilot_status.log` (pilot);
tail either file to monitor a run in real time.

## Figures

| file | written by | content |
|---|---|---|
| `figures/pilot_scan.png` | `pilot.py` | κ scan (magnetisation histograms) and frozen tilted reference |
| `figures/fig_background.png` | `plot_results.py` | per-site double-well potential, PT reference p(m), and the two vacua |
| `figures/fig_methods.png` | `plot_results.py` | per-method reweighted p(m) vs PT reference — diagnostic/reference figure (NOT in the paper; the paper uses `../Phi4_Lattice_8/figures/fig_methods.png`) |
| `figures/fig_ess.png` | `plot_results.py` | training-ESS history for all four methods |

## Folder tree

```
Phi4_Lattice_6/
├── parameters.py          # all hyperparameters (L, κ, λ, H, NSF arch, training)
├── core.py                # Phi4 energy, magnetization, loss_KL, loss_X, quench_and_temper
├── pilot.py               # stage-1 PT-MALA pilot scan; freezes κ and H
├── train.py               # stage-2 training (four loss variants)
├── plot_results.py        # stage-3 figures + results_table.{md,csv}
├── pilot_results.md       # pilot output: κ scan table + frozen parameters
├── phi4_reference.pth     # PT reference: samples, magnetisation trace, ΔF
├── data.pth               # default training run (all four methods, seed 0)
├── data_seed1.pth         # seed-1 robustness run
├── data_seed2.pth         # seed-2 robustness run
├── data_skew05.pth        # skewed-oracle run (5 % m>0 in QT set)
├── data_skew25.pth        # skewed-oracle run (25 % m>0 in QT set)
├── results_table.md       # per-method ESS / p(m>0) / ΔF table (written by plot_results.py)
├── results_table.csv      # same table in CSV format
├── train_status.log       # timestamped training log (appended by train.py)
├── pilot_status.log       # timestamped pilot log (written by pilot.py; absent until pilot runs)
├── BACKGROUND.md          # extended background notes
└── figures/
    ├── pilot_scan.png     # κ scan and frozen tilted reference (pilot.py)
    ├── fig_background.png # potential + PT reference (plot_results.py)
    ├── fig_methods.png    # reweighted p(m) per method — diagnostic/reference, NOT in paper (plot_results.py)
    └── fig_ess.png        # training ESS history (plot_results.py)
```

## Outputs

A complete run writes:

- `phi4_reference.pth` — frozen PT reference (κ, H, ΔF, magnetisation trace, samples)
- `pilot_results.md` — κ scan table and chosen frozen parameters
- `figures/pilot_scan.png` — pilot magnetisation histograms
- `data.pth` — per-method flow state dicts, ESS histories, pushforward samples and weights
- `figures/fig_background.png`, `figures/fig_methods.png`, `figures/fig_ess.png` — all figures
- `results_table.md` / `results_table.csv` — final ESS and coverage table

Quantitative results (final ESS per method, reweighted p(m>0), ΔF) are reported and
discussed in **`Paper/main.tex`**.
