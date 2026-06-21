# Clock_Lattice — p-state clock model on an 8×8 periodic lattice

Reproduction guide for the **p=6 clock model** Boltzmann-generator test (Algorithm 4,
`Paper/main.tex`). The target is the cold Boltzmann measure of

    U(θ) = −J Σ_{⟨ij⟩} cos(θ_i − θ_j) − H Σ_i cos(P θ_i)

on a periodic 8×8 square lattice (D = 64 angles, P = 6 sectors, J = 1.0, H = 0.5).
The experiment compares the X-regularized stage loss (`balance`) against bare forward KL
(`kl`) across three batch sizes (B = 1k, 10k, 100k) to demonstrate the bias-variance
effect in the adaptive-temperature ladder.

## How to run

Activate the environment and run from inside the `Clock_Lattice` folder:

```bash
conda activate zflows
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/Clock_Lattice
```

All scripts resolve paths relative to their own location (`HERE = Path(__file__).resolve().parent`),
so they must be launched from `Clock_Lattice/` or with an absolute path.

### Training runs

Each invocation trains one full ladder and writes `data_<tag>.pth`,
`results_table.md/.csv`, `summary.md`, and per-run figures under `figures/`; a live
status log is written to `train_status.log`.
Run one at a time; confirm the GPU is free (`nvidia-smi`) before launching the next.

The default lattice size is L=8 (D=64). Use `--smoke` for a quick sanity check before
a full run.

```bash
# sanity check — tiny run exercising the full code path
python train.py --L 8 --smoke

# X-regularized (KL + X_mu + X_mix) at three batch sizes — the deliverable rows
python train.py --L 8 --method balance   # default B=10k; tag L8_balance_B10k

# bare forward KL baseline at three batch sizes
python train.py --L 8 --method kl        # tag L8_kl_B10k
```

To reproduce the full B=1k / B=10k / B=100k sweep (six runs total), set `N_BATCH` in
`parameters.py` before each pair of runs; the tag encodes the batch size automatically
(e.g. `N_BATCH = 1000` → tag `L8_balance_B1k`). Use `--suffix` only to further
disambiguate within the same `N_BATCH` setting:

```bash
# example: re-run balance at B=10k with a distinguishing suffix
python train.py --L 8 --method balance --suffix _v2   # tag: L8_balance_B10k_v2
```

**Flags:**
- `--L INT` — lattice side length (default: 8 from `parameters.py`; D = L²).
- `--method {balance|kl}` — `balance`: KL + X_mu + X_mix at balanced hyperparameters;
  `kl`: bare forward KL only (no X terms, no QT pool).
- `--smoke` — tiny sanity run (overrides N, steps, stages to small values).
- `--suffix STR` — appended to the output tag to prevent file clashes.

### Occupancy-bias scaling test

Requires `data_L8_balance_B10k.pth` (written by `train.py` above). Runs the staged
sampler at eight particle counts N = 10 000 × 2^k (k = 0…7) with equal total work
and measures the mean sector-occupancy bias to verify N^{−1/2} Monte Carlo scaling.

```bash
# full occupancy-bias scaling test (reads data_L8_balance_B10k.pth; no retraining)
python occupancy_bias_B10k.py

# smoke version for a quick check
python occupancy_bias_B10k.py --smoke

# partial rerun starting from row k (0-indexed)
python occupancy_bias_B10k.py --kmin 3
```

Writes: `occupancy_bias_B10k.md`, `occupancy_bias_B10k.csv`,
`figures/occupancy_bias_B10k.png`, `occ_bias_B10k_status.log`.

### Publication figures

Reads the pre-trained `.pth` files — no retraining. Requires the six sweep files
`data_L8_{balance,kl}_B{1k,10k,100k}.pth` and the rebuild file
`rebuild_L8_balance_B10k_N1000000.pth`.

```bash
# target-structure figure (magnetization scatter, per-site marginal, sector occupancy)
# and ESS-curves figure (per-step training ESS across batch sizes and methods)
python plot_paper.py
```

Writes: `figures/fig_clock_target.png`, `figures/fig_clock_esscurves.png`.

## Figures

| file | producing script | description |
|---|---|---|
| `figures/fig_clock_target.png` | `plot_paper.py` | (a) magnetization-plane scatter by sector; (b) per-site angle marginal (reweighted vs pushforward); (c) sector occupancy ±2σ |
| `figures/fig_clock_esscurves.png` | `plot_paper.py` | per-step training ESS curves for the first three stages, at B = 1k / 10k / 100k, balance vs KL |
| `figures/occupancy_bias_B10k.png` | `occupancy_bias_B10k.py` | occupancy-bias vs N log-log plot with N^{−1/2} reference line |

`train.py` also writes per-run figures (`figures/ladder_<tag>.png`,
`figures/sectors_<tag>.png`, `figures/magnetization_<tag>.png`) after each training
run completes.

## Folder tree

```
Clock_Lattice/
├── train.py                        # main driver: Algorithm 4 ladder training
├── plot_paper.py                   # publication figures (reads .pth, no retraining)
├── occupancy_bias_B10k.py          # occupancy-bias N^{-1/2} scaling test
├── potential.py                    # Clock potential + sector_occupancy + torus_coverage
├── parameters.py                   # all hyperparameters (L, P, J, H, N_BATCH, …)
├── core/
│   ├── __init__.py                 # re-exports all public symbols from boltzmann.py
│   └── boltzmann.py                # run_boltzmann, compose_pushforward, quench_and_temper_torus, bridge, losses
├── data_L8_balance_B1k.pth         # saved run (state_dicts + metrics)
├── data_L8_balance_B10k.pth        # saved run
├── data_L8_balance_B100k.pth       # saved run
├── data_L8_kl_B1k.pth             # saved run
├── data_L8_kl_B10k.pth            # saved run
├── data_L8_kl_B100k.pth           # saved run
├── rebuild_L8_balance_B10k_N1000000.pth  # 1M-sample staged rebuild for fig_clock_target.png (paper Figure 8)
├── results_table.md                # auto-generated summary of all data_*.pth
├── results_table.csv               # same, CSV format
├── occupancy_bias_B10k.md          # occupancy-bias scaling table
├── occupancy_bias_B10k.csv         # same, CSV format
├── occ_bias_B10k_status.log        # live log for occupancy_bias_B10k.py
├── train_status.log                # live log written by train.py at runtime (regenerated on demand by train.py; not committed)
├── summary.md                      # auto-written after each train.py run (auto-written by train.py; not committed)
└── figures/
    ├── fig_clock_target.png        # publication figure (plot_paper.py)
    ├── fig_clock_esscurves.png     # publication figure (plot_paper.py)
    ├── occupancy_bias_B10k.png     # scaling figure (occupancy_bias_B10k.py)
    ├── ladder_<tag>.png            # per-run ESS training curve (regenerated on demand by train.py; not committed)
    ├── sectors_<tag>.png           # per-run sector-occupancy figure (regenerated on demand by train.py; not committed)
    └── magnetization_<tag>.png     # per-run magnetization figure (regenerated on demand by train.py; not committed)
```

## Outputs

A `train.py` run writes:
- `data_<tag>.pth` — per-stage records including every flow `state_dict`, validation
  ESS, ladder temperatures, sector occupancy counts, final composed ESS, and raw
  samples (up to 20 000 pushforward and validation samples).
- `results_table.md` / `results_table.csv` — rebuilt from all `data_*.pth` after each
  run; one row per tag.
- `summary.md` — human-readable headline metrics for the most recent run.
- `train_status.log` — timestamped live log (tail -f to watch progress).
- `figures/ladder_<tag>.png`, `figures/sectors_<tag>.png`,
  `figures/magnetization_<tag>.png` — per-run diagnostic figures written by
  `plot_results.py` (imported at the end of `train.py`; must be present to avoid an
  error at the figure-generation step — the `.pth` is already saved before that import).

Quantitative results (per-stage ESS, sector coverage, wall time) are reported in the
paper; see **`Paper/main.tex`**. The `summary.md` results report in this folder is
retired and preserved for reference only.
