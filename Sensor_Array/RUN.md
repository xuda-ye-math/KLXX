# Sensor-array source localization (3 sources, 3! = 6 permutation modes)

Reproduction guide for the sensor-array Bayesian inverse problem. A line of 25
sensors observes three identical point sources at unknown positions; because the
sources are identical, the posterior has exactly 3! = 6 degenerate, well-separated
permutation modes. The prior (Gaussian centred at the origin) cannot reach them,
so bare forward KL collapses — this experiment is a fake-ESS-collapse discriminator.
All four X-functional losses (KL, KL+X_mu, KL+X_mu+X_hat_mu, KL+X_mu+X_mix) are
trained and evaluated in a single run. All scripts must be executed from the
**`Sensor_Array/`** folder.

## How to run

```bash
conda activate zflows
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/Sensor_Array
```

### 1. Train all four methods and save results

```bash
# full run (2000 steps, all 4 methods); writes data.pth + train_status.log
python train.py

# sanity / smoke run (fast, fewer steps)
python train.py --steps 200

# run with a custom tag (saves data_<tag>.pth instead of data.pth)
python train.py --tag mytag
```

`train.py` trains each method sequentially, then evaluates all four on the same
target. Progress is logged to `train_status.log` (and stdout) in real time.

### 2. Build the results table

```bash
# reads data.pth; writes results_table.md and results_table.csv
python build_table.py
```

### 3. Generate figures

```bash
# reads data.pth; writes ESS.png, samples.png, resample.png
python plot_results.py
```

### 4. Recompute k-NN coverage (optional verification)

```bash
# reads data.pth; rebuilds the QT pool deterministically and recomputes
# Naeem coverage at k=3,4,5; validates QT occupancy against saved values
# and cross-checks each method's k=5 coverage against the stored knn_coverage
python recompute_coverage.py
```

## Figures

| file | script | content |
|------|--------|---------|
| `ESS.png` | `plot_results.py` | Training-target ESS trajectory (faint raw + bold moving average) for all four losses; illustrates fake-ESS collapse |
| `samples.png` | `plot_results.py` | 1×4 scatter panels (θ₁, θ₂ projection) showing which losses collapse vs. recover all 6 permutation modes |
| `resample.png` | `plot_results.py` | Same 1×4 layout after one-step IS resample + short Langevin polish; requires `state_dict` in `data.pth` |

## Folder tree

```
Sensor_Array/
├── train.py               # main driver: trains 4 methods, evaluates, saves data.pth
├── build_table.py         # reads data.pth -> results_table.{md,csv}
├── plot_results.py        # reads data.pth -> ESS.png, samples.png, resample.png
├── recompute_coverage.py  # optional: rebuild QT pool + recompute k-NN coverage
├── core.py                # sensor forward model, SensorArrayPosterior, loss/QT helpers
├── parameters.py          # all hyperparameters (N_SRC, N_SENSORS, STEPS, …)
├── data.pth               # saved run: per-method ESS history, samples, state_dicts
├── train_status.log       # timestamped progress log written by train.py
├── results_table.md       # markdown results table (one row per loss)
├── results_table.csv      # same table in CSV
├── ESS.png                # figure: ESS trajectories
├── samples.png            # figure: mode-coverage scatter
├── resample.png           # figure: post-resample scatter
└── full_run.stdout        # stdout captured from a previous full run (reference)
```

## Outputs

A complete run writes:

- `data.pth` — all per-method results: ESS history, final ESS, mode coverage,
  k-NN coverage, 20k pushforward samples, and flow `state_dict`s.
- `train_status.log` — timestamped progress log (START / per-step / DONE markers).
- `results_table.md` / `results_table.csv` — quantitative results table.
- `ESS.png`, `samples.png`, `resample.png` — the three experiment figures.

Quantitative results (ESS, modes found, coverage values) are reported in the
paper: **`Paper/main.tex`**.
