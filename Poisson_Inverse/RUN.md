# Reproducing the results (Poisson_Inverse — screened-Poisson Bayesian inversion)

Reproduction guide for the Algorithm-4 staged-sampler test on the Bayesian screened-Poisson
inverse problem. The target is a low-mode Fourier posterior (`d_low = M_LOW^2`, default 36
from `parameters.py` M_LOW=6) embedded in a full spectral grid (`d_full = M_FULL^2`, default
64 from M_FULL=8); a 6×6 cosine-modulated source drives a screened Poisson PDE whose noisy
sensor observations induce a multimodal posterior with a discrete shift×sign well lattice.
Run everything from the **`Poisson_Inverse` folder**; all scripts use
`Path(__file__).resolve().parent` internally.

## How to run

```bash
conda activate zflows
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/Poisson_Inverse
```

All commands below assume that working directory.

---

### 0. (Optional) Truncation-ceiling diagnostic

Runs before any training. Checks that the likelihood barely constrains the extension
block and that a ceiling-ESS test passes, writing `diagnostic.md` and
`figures/diagnostic.png`. The `--m-low`/`--m-full` defaults come from `parameters.py`
(M_LOW=6, M_FULL=8).

```bash
# production geometry (6×6 in 8×8)
python diagnostic.py --m-low 6 --m-full 8
```

Optional flag overrides: `--n-samp` (default 4096), `--prior-s`, `--sigma-obs`, `--c2`,
`--delta`, `--n-sensors`, `--tilt`, `--tag` (appended to `diagnostic.md` output filename).

### 0b. (Optional) Well-structure inspection

Maps the posterior mode lattice via QT + MALA, writes `wells.md` and
`figures/wells_m{m_low}.png`.

```bash
# smoke geometry (4×4 in 8×8), fast
python inspect_wells.py --m-low 4 --m-full 8

# production geometry (6×6 in 8×8)
python inspect_wells.py --m-low 6 --m-full 8
```

---

### 1. PT-MALA referee (`pilot.py`)

Runs parallel-tempering MALA on the full posterior for each `sigma_obs` value. Certifies
the true well weights that the trained generators are judged against. Writes
`referee_o{sigma}.pth` and appends to `referee.md`. All runs log to `train_status.log`.

```bash
# all four sigma_obs values in one call (default: 0.01,0.015,0.02,0.025)
python pilot.py

# or a single sigma_obs
python pilot.py --sigmas 0.01
```

Flags: `--sigmas` (comma-separated list, default `0.01,0.015,0.02,0.025`), `--steps`
(default 6000), `--chains` (default 128), `--rungs` (default 20).

---

### 2. Single-stage smoke test (`train_single.py`)

Trains one flow for one bridge stage, reading hyperparameters from `parameters.py`.
Used to calibrate `T_SAFE` and check that the whitened pipeline trains end-to-end.
Appends one row to `single_stage.md`; saves the ESS curve to
`figures/single_{tag}.png` and the state dict to `data_single_{tag}.pth`.
The tag encodes all key parameters: `m{m_low}_s{prior_s}_o{sigma_obs}_t{t}_{method}`.
Logs to `train_status.log`.

```bash
# balance loss, t=0.1 (default m-low=6, m-full=8)
python train_single.py --t 0.1 --method balance

# forward KL baseline, same stage
python train_single.py --t 0.1 --method kl

# very shallow stage
python train_single.py --t 0.01 --method balance
```

Flags: `--t` (temper, default 0.1), `--method {balance|kl}` (default `balance`),
`--m-low` (default 6), `--m-full` (default 8), `--sigma-obs` (default from
`parameters.py`), `--steps` (default 400), `--batch` (default 1000).

---

### 3. Full staged-ladder run (`train.py`)

Runs the complete Algorithm-4 annealed ladder from the prior to the full posterior
for one `(method, sigma_obs)` cell. The `--m-low` default in `train.py` is 4 (a quick
sanity default); pass `--m-low 6 --m-full 8` explicitly for the paper's production
geometry. Writes `train_status.log`, `data_{tag}.pth`, `data_{tag}_partial.pth`
(checkpoint), `results_table.md`, `results_table.csv`, and `figures/ladder_{tag}.png`.

```bash
# forward KL baseline, sigma_obs = 0.01, production geometry
python train.py --method kl --m-low 6 --m-full 8 --sigma-obs 0.01

# X-regularised balance loss, sigma_obs = 0.01 (paper's deliverable)
python train.py --method balance --m-low 6 --m-full 8 --sigma-obs 0.01

# other sigma_obs cells (repeat for 0.015, 0.02, 0.025)
python train.py --method kl      --m-low 6 --m-full 8 --sigma-obs 0.015
python train.py --method balance --m-low 6 --m-full 8 --sigma-obs 0.015
```

Flags: `--method {kl|balance}` (default `kl`), `--m-low` (default 4), `--m-full`
(default 8), `--sigma-obs` (default from `parameters.py`), `--gate-tau` (validation
gate floor; 0 disables the gate), `--n-sensors` (0 = auto), `--suffix` (appended to
the output tag).

Run one cell at a time; confirm the GPU is free (`nvidia-smi`) before launching the next.

---

### 4. Staged-census sweep (`run_staged_sweep.sh`)

Replays the eight saved `data_*.pth` files (4 sigma_obs × 2 methods) with 3 independent
random seeds each via `staged_census.py`, computes TV distance to the PT-MALA referee
well weights, and writes `staged_tv.csv`. Requires all eight `data_*.pth` files (step 3)
and all four `referee_o*.pth` files (step 1). Must be run from `Poisson_Inverse/`.

```bash
bash run_staged_sweep.sh     # 8 cells × 3 seeds -> staged_tv.csv
```

A single cell can be run manually:

```bash
# replay data_balance_o0.01.pth with seed 1
python staged_census.py data_balance_o0.01.pth 1
```

The script passes the filename as `sys.argv[1]` and the seed as `sys.argv[2]` (default 123
if omitted).

---

### 5. Figures

#### Ladder figure (`plot_ladder.py`)

Reads all `data_*.pth` and `staged_tv.csv`; draws the per-sigma arc-ladder panels (forward
KL above, balance loss below; TV to referee annotated per panel).

```bash
python plot_ladder.py    # -> figures/poisson_ladders.png
```

#### Setup figure (`plot_setup.py`)

Shows the true source field, the PDE solution with sensor ring, and the posterior well
scatter from the PT referee at `sigma_obs = 0.01`. Reads `referee_o0.01.pth` via a
relative path; must be run from `Poisson_Inverse/`.

```bash
python plot_setup.py     # -> figures/fig_setup.png
```

---

## Figures

| figure | script | path |
|---|---|---|
| per-sigma arc-ladder (paper figure) | `plot_ladder.py` | `figures/poisson_ladders.png` |
| problem setup (source, PDE, referee wells) | `plot_setup.py` | `figures/fig_setup.png` |
| truncation-ceiling diagnostic | `diagnostic.py` | `figures/diagnostic.png` |
| well structure (QT+MALA census) | `inspect_wells.py` | `figures/wells_m{m_low}.png` |
| per-stage ESS + well occupancy (single run) | `train.py` | `figures/ladder_{tag}.png` |
| single-stage ESS curve | `train_single.py` | `figures/single_{tag}.png` |

---

## Folder tree

```
Poisson_Inverse/
├── core/                        # internal subpackage (potential-invariant Algorithm-4 loop)
│   ├── __init__.py              # re-exports run_boltzmann, identity_wrap, bridge, losses
│   └── boltzmann.py             # Algorithm-4 staged-sampler loop + SMC + QT utilities
├── figures/                     # all PNG outputs land here
│   ├── poisson_ladders.png      # paper figure: arc-ladder per sigma_obs
│   ├── fig_setup.png            # paper figure: source, PDE solution, referee wells
│   ├── diagnostic.png           # truncation-ceiling gate plot (generated on demand by diagnostic.py; not committed)
│   ├── wells_m4.png             # well census, 4×4 smoke geometry (generated on demand by inspect_wells.py; not committed)
│   ├── wells_m6.png             # well census, 6×6 production geometry (generated on demand by inspect_wells.py; not committed)
│   └── ladder_*.png             # per-run ESS curve + well-occupancy bar (from train.py)
├── parameters.py                # all hyperparameters (read-only truth; M_LOW=6, M_FULL=8)
├── potential.py                 # screened-Poisson PDE + posterior potential + whitening
├── pilot.py                     # PT-MALA referee: certifies true well weights (step 1)
├── train_single.py              # single-stage smoke test: one flow, one bridge (step 2)
├── train.py                     # full Algorithm-4 ladder run, one (method, sigma_obs) cell (step 3)
├── run_staged_sweep.sh          # 8-cell × 3-seed sweep driver: calls staged_census.py (step 4)
├── staged_census.py             # replay one data_*.pth, compute TV to referee (step 4)
├── plot_ladder.py               # arc-ladder figure (step 5)
├── plot_setup.py                # setup figure: source, PDE, referee scatter (step 5)
├── diagnostic.py                # truncation-ceiling gate: runs before training (step 0)
├── inspect_wells.py             # well-structure census via QT + MALA (step 0b)
├── data_kl_o0.01.pth            # saved run: forward KL, sigma_obs=0.01
├── data_balance_o0.01.pth       # saved run: balance loss, sigma_obs=0.01
├── data_kl_o0.015.pth           # saved run: forward KL, sigma_obs=0.015
├── data_balance_o0.015.pth      # saved run: balance loss, sigma_obs=0.015
├── data_kl_o0.02.pth            # saved run: forward KL, sigma_obs=0.02
├── data_balance_o0.02.pth       # saved run: balance loss, sigma_obs=0.02
├── data_kl_o0.025.pth           # saved run: forward KL, sigma_obs=0.025
├── data_balance_o0.025.pth      # saved run: balance loss, sigma_obs=0.025
├── data_*_partial.pth           # in-progress stage checkpoints (one per active run)
├── referee_o0.01.pth            # PT-MALA referee for sigma_obs=0.01
├── referee_o0.015.pth           # PT-MALA referee for sigma_obs=0.015
├── referee_o0.02.pth            # PT-MALA referee for sigma_obs=0.02
├── referee_o0.025.pth           # PT-MALA referee for sigma_obs=0.025
├── staged_tv.csv                # sweep output: TV per (file, seed) cell
├── results_table.md             # per-run summary table (written by train.py)
├── results_table.csv            # same, CSV format
├── train_status.log             # live timestamped log (appended by pilot.py, train*.py)
├── referee.md                   # PT-MALA well-weight table (written by pilot.py)
├── single_stage.md              # single-stage smoke-test results (written by train_single.py)
├── wells.md                     # well-structure report (written by inspect_wells.py, appended)
├── diagnostic.md                # truncation-ceiling gate verdict (written by diagnostic.py)
├── diagnostic_soft01_s6.md      # archived diagnostic run: sigma_obs=0.01, prior_s=6
├── diagnostic_sweeps.md         # notes on diagnostic parameter sweeps
├── MATH.md                      # derivation notes: whitening, PDE forward map, well lattice
├── PLAN.md                      # development plan and stage-by-stage milestones
└── comparison_referee.md        # notes comparing staged-census to PT-MALA referee
```

---

## Outputs

Each `train.py` run writes:
- `data_{tag}.pth` — per-stage records including `state_dict`, ESS histories, well
  occupancies, final fine ESS, and the annealing ladder.
- `data_{tag}_partial.pth` — checkpoint updated each stage (safe to inspect mid-run).
- `results_table.md` / `results_table.csv` — collated summary of all completed runs in
  the folder.
- `figures/ladder_{tag}.png` — per-stage ESS training curve + well-occupancy bar chart.

`pilot.py` writes `referee_o{sigma}.pth` and appends to `referee.md`.

`run_staged_sweep.sh` writes `staged_tv.csv` (the TV-to-referee table consumed by
`plot_ladder.py`).

Quantitative results (per-stage ESS, fine ESS, TV distances, comparison with the
PT-MALA referee) are reported in the paper; see **`Paper/main.tex`** (§5.6).
