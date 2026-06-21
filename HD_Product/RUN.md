# Reproducing the results — HD Product multi-well (d = 2, 4, …, 128)

High-dimensional companion to the 2D benchmarks. The target is a separable
product multi-well in dimension d = 2^k (k = 1…7), with exactly 2^k modes; the
headline deliverable is the **4 losses × 7 dimensions ESS table** written to
`ess_table.csv` and `tables.md`. The ESS gap between bare forward KL and the
X-augmented losses widens monotonically with dimension — this is the
**ESS-discriminator regime**. Run everything from inside the `HD_Product/`
folder with the `zflows` conda environment active.

## How to run

```bash
conda activate zflows
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/HD_Product
```

### Step 1 — full sweep (all k, highest d first)

```bash
# Train all 4 losses for k=7…1 (d=128…2); writes data_k{1..7}.pth
python train.py
```

Optional flags (all from `argparse` in `train.py`):

| flag | default | meaning |
|---|---|---|
| `--klist K[,K,…]` | `parameters.K_LIST` = 7,6,5,4,3,2,1 | comma-separated k values to run |
| `--steps N` | `parameters.steps(k)` per k | override per-k step count |
| `--budget S` | none | per-method wall-clock soft budget in seconds (kills early if projected to exceed 2× budget) |

To rerun a single dimension (e.g. k=6, d=64):

```bash
# Rerun k=6 only, 1200 steps; writes data_k6.pth
python train.py --klist 6 --steps 1200
```

Progress is streamed live to `train_status.log` (per step) and
`sweep_status.log` (per dimension) in this folder; tail either file to watch:

```bash
tail -f sweep_status.log
```

### Step 2 — build the headline tables

```bash
# Reads data_k*.pth; writes ess_table.csv, mode_coverage_table.csv,
# mode_balance_table.csv, and tables.md
python build_table.py
```

`build_table.py` takes no arguments. It reads all `data_k*.pth` files it finds
in the folder and produces three CSV tables plus a Markdown summary.

## Figures

No figure script is currently present in this folder. The ESS-over-step
trajectory figure (paper Figure 4, `ESS_k7.png`) is produced by
`plot_ess.py` when that script is added; it reads `data_k7.pth`.

## Folder tree

```
HD_Product/
├── core.py                  # MultiWell potential; loss_KL, loss_X, quench_and_temper, mode_coverage
├── parameters.py            # K_LIST, per-k functions (dim, steps, bins, …), all hyperparameters
├── train.py                 # orchestrates the sweep; saves data_k{K}.pth; logs to *_status.log
├── build_table.py           # reads data_k*.pth; writes ess/coverage/balance CSVs + tables.md
│
├── data_k1.pth              # saved run: k=1 (d=2),  all 4 methods
├── data_k2.pth              # saved run: k=2 (d=4)
├── data_k3.pth              # saved run: k=3 (d=8)
├── data_k4.pth              # saved run: k=4 (d=16)
├── data_k5.pth              # saved run: k=5 (d=32)
├── data_k6.pth              # saved run: k=6 (d=64)
├── data_k7.pth              # saved run: k=7 (d=128)
│
├── ess_table.csv            # headline: final ESS (4 losses × 7 d)
├── mode_coverage_table.csv  # strict mode coverage (≥50% of uniform share)
├── mode_balance_table.csv   # mode imbalance TV(occupancy, uniform)
├── tables.md                # Markdown rendering of the three tables above
│
├── summary.md               # legacy narrative (retired; superseded by Paper/main.tex)
└── RUN.md                   # this file
```

## Outputs

A complete run writes:

- `data_k{K}.pth` for each k — per-method records: `ess_history`, `final_ess`,
  `mode_coverage`, `mode_balance`, `knn_coverage`, `samples` (20 000 × d),
  plus the training config (`k`, `d`, `steps`, scalar hyperparameters, arch
  params) and `qt_mode_coverage` (QT pool's mode-coverage fraction). Saved
  incrementally after each method so a mid-run interruption loses at most one
  method.
- `train_status.log` — per-step progress (loss, ESS, timing, ETA).
- `sweep_status.log` — per-dimension heartbeat (QT pool, per-method final ESS,
  mode coverage, snapshot confirmation).
- `ess_table.csv`, `mode_coverage_table.csv`, `mode_balance_table.csv`,
  `tables.md` — produced by `build_table.py` after training.

Quantitative results (the ESS table and mode-coverage commentary) are reported
in **`Paper/main.tex`** §4.2, Table 3 and Figure 4. Do not rely on `summary.md`
for the final numbers; that file is retired.
