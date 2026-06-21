# Reproducing the results (HD_Product_Ladder — AIS ladder sweep, d = 256)

AIS-ladder sweep on the product multi-well target at d = 256 (K = 8). A single
flow is trained with the KL + X_mu + X_mix loss for each ladder length
M ∈ {1, 2, 4, 8, 16, 32}; the only thing that varies across runs is M.
The M = 1 case is identical to the one-step IS surrogate in HD_Product; larger M
adds geometric AIS rungs that reuse the single inverse per step.

## How to run

Activate the environment and ensure the working directory is this folder:

```bash
conda activate zflows
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/HD_Product_Ladder
```

### Full sweep (all six M values, then plot)

`run_sweep.sh` drives the complete sweep: it iterates over M ∈ {1, 2, 4, 8, 16, 32},
calls `python train.py --M $M` for each, and then calls `plot_ess.py` with no
arguments (so the plot covers the default M ∈ {1, 2, 4, 8}; pass the full list
explicitly to include M=16 and M=32 — see Plot section below). The script changes
into its own directory before launching, so it can be invoked from anywhere.

```bash
# Run the full AIS-ladder sweep (foreground); or use nohup for overnight
bash run_sweep.sh
```

For an overnight run:

```bash
# Redirect stdout/stderr and detach
nohup bash run_sweep.sh > full_run.stdout 2>&1 &
```

### Single run (one M value)

```bash
# Train for ladder length M=8, d=256, 1000 steps
python train.py --M 8
```

The `--steps` flag overrides the default of 1000 (useful for a quick sanity check):

```bash
# Quick sanity check: 50 steps only
python train.py --M 8 --steps 50
```

### Plot

Once `data_M*.pth` files exist, regenerate `ESS_ladder.png`. By default, `plot_ess.py`
reads M ∈ {1, 2, 4, 8}; pass a comma-separated list to override:

```bash
# Plot curves for M = 1, 2, 4, 8 (default)
python plot_ess.py

# Plot all six M values
python plot_ess.py 1,2,4,8,16,32
```

## Figures

| Figure | Script | Output |
|---|---|---|
| ESS trajectory vs. training step, one curve per M | `plot_ess.py` | `ESS_ladder.png` |

`ESS_ladder.png` shows per-step direct mu/nu ESS (faint) plus a moving average
(bold) for each M value, at d = 256.

## Folder tree

```
HD_Product_Ladder/
├── run_sweep.sh          # sweep driver: iterates M, calls train.py + plot_ess.py
├── train.py              # training script; requires --M; writes data_M{M}.pth
├── plot_ess.py           # figure script; reads data_M*.pth; writes ESS_ladder.png
├── core.py               # MultiWell target, loss_KL, loss_X, fused_loss, quench_and_temper, coverage, mode_coverage
├── parameters.py         # all hyperparameters (K=8 → d=256, STEPS=1000, ...)
├── data_M1.pth           # saved run for M=1
├── data_M2.pth           # saved run for M=2
├── data_M4.pth           # saved run for M=4
├── data_M8.pth           # saved run for M=8
├── data_M16.pth          # saved run for M=16
├── data_M32.pth          # saved run for M=32
├── ESS_ladder.png        # ESS-trajectory figure (one curve per M)
├── train_status.log      # per-step progress log written by train.py
└── sweep_status.log      # sweep-level start/done lines written by train.py (not run_sweep.sh)
```

## Outputs

Each `python train.py --M $M` run writes:

- `data_M{M}.pth` — saved state: ESS history, final ESS, mode coverage, kNN
  coverage, flow `state_dict`, and 20 000 samples from the trained flow.
- `train_status.log` — per-step progress (timestamped loss, ESS, ms/step).
- `sweep_status.log` — sweep-level START and DONE lines with final metrics (written by train.py).

`plot_ess.py` writes `ESS_ladder.png` (reads the `data_M*.pth` files; no
retraining).

Quantitative results (final ESS by M, mode coverage, kNN coverage) are reported
in the paper at `Paper/main.tex`; no `summary.md` or results table is written by
these scripts.
