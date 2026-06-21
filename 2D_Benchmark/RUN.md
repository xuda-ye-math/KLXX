# Reproducing the 2D Benchmark results

Six self-contained 2D multimodal targets — Annulus, Chessboard, Himmelblau, Python, Sparse, Threewell — each
isolating a different failure mode of the bare forward KL. Every sub-folder trains four losses (forward KL,
forward KL + X_mu, forward KL + X_mu + X_{hat_mu}, forward KL + X_mu + X_mix) and writes figures that make
the coverage differences visible. These benchmarks correspond to §4.1, Figures 1–3, and Table 2 of
`Paper/main.tex`.

## How to run

Activate the environment and **run all scripts from inside the target sub-folder** — each script resolves
`data.pth` and figure paths relative to its own directory via `Path(__file__).resolve().parent`.

```bash
conda activate zflows
```

### Single benchmark (example: 2D_Sparse)

```bash
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/2D_Benchmark/2D_Sparse

# Train all four methods; writes data.pth (skipped automatically if data.pth already exists)
python train.py

# Render ESS.png, samples.png, and resample.png from the saved data.pth
python plot_results.py
```

Neither script takes command-line arguments.

### All six benchmarks

```bash
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/2D_Benchmark

for TARGET in 2D_Annulus 2D_Chessboard 2D_Himmelblau 2D_Python 2D_Sparse 2D_Threewell; do
    cd "$TARGET"
    python train.py        # train; skipped if data.pth present
    python plot_results.py # render figures
    cd ..
done
```

## Figures

Each `plot_results.py` writes three figures into its sub-folder:

| file | contents |
|---|---|
| `samples.png` | Pushforward samples $y = G^{-1}(x)$ for all four methods, with per-method ESS and coverage annotated in the title |
| `resample.png` | Importance-resampled points $\mathrm{resample}(y, w)$ overlaid on the target contour |
| `ESS.png` | ESS training history (one curve per method) over the full optimization |

## Folder tree

```
2D_Benchmark/
├── RUN.md                       # this file
├── summary.md                   # narrative overview (retired results report; kept for reference)
├── 2D_Annulus/
│   ├── core.py                  # Annulus potential + loss functions + QT helper
│   ├── parameters.py            # all hyperparameters (architecture, training)
│   ├── train.py                 # trains four methods; saves data.pth
│   ├── plot_results.py          # reads data.pth; writes ESS.png, samples.png, resample.png
│   ├── test_qt.py               # standalone QT diagnostic; writes qt.png (not needed for main results)
│   ├── data.pth                 # saved run (flow state_dicts + ESS histories + samples)
│   ├── ESS.png
│   ├── samples.png
│   ├── resample.png
│   └── qt.png                   # QT diagnostic visualisation (output of test_qt.py)
├── 2D_Chessboard/
│   ├── core.py                  # Chessboard potential + loss functions + QT helper
│   ├── parameters.py            # all hyperparameters (architecture, training)
│   ├── train.py                 # trains four methods; saves data.pth
│   ├── plot_results.py          # reads data.pth; writes ESS.png, samples.png, resample.png
│   ├── data.pth
│   ├── ESS.png
│   ├── samples.png
│   ├── resample.png
│   └── core.png                 # target potential visualisation (output of `python core.py`)
├── 2D_Himmelblau/
│   ├── core.py                  # Himmelblau potential + loss functions + QT helper
│   ├── parameters.py            # all hyperparameters (architecture, training)
│   ├── train.py                 # trains four methods; saves data.pth
│   ├── plot_results.py          # reads data.pth; writes ESS.png, samples.png, resample.png
│   ├── test_qt.py               # standalone QT diagnostic; writes qt.png (not needed for main results)
│   ├── data.pth
│   ├── ESS.png
│   ├── samples.png
│   ├── resample.png
│   └── qt.png                   # QT diagnostic visualisation (output of test_qt.py)
├── 2D_Python/
│   ├── core.py                  # Python-logo potential + loss functions + QT helper
│   ├── parameters.py            # all hyperparameters (architecture, training)
│   ├── train.py                 # trains four methods; saves data.pth
│   ├── plot_results.py          # reads data.pth; writes ESS.png, samples.png, resample.png
│   ├── data.pth
│   ├── ESS.png
│   ├── samples.png
│   ├── resample.png
│   ├── core.png                 # target potential visualisation (output of `python core.py`)
│   ├── python.jpg               # reference Python-logo image used to define the potential
│   └── train.log                # training log from the last run
├── 2D_Sparse/
│   ├── core.py                  # Sparse GMM potential + loss functions + QT helper
│   ├── parameters.py            # all hyperparameters (architecture, training)
│   ├── train.py                 # trains four methods; saves data.pth
│   ├── plot_results.py          # reads data.pth; writes ESS.png, samples.png, resample.png
│   ├── data.pth
│   ├── ESS.png
│   ├── samples.png
│   └── resample.png
└── 2D_Threewell/
    ├── core.py                  # Threewell potential + loss functions + QT helper
    ├── parameters.py            # all hyperparameters (architecture, training)
    ├── train.py                 # trains four methods; saves data.pth
    ├── plot_results.py          # reads data.pth; writes ESS.png, samples.png, resample.png
    ├── test_qt.py               # standalone QT diagnostic; writes qt.png (not needed for main results)
    ├── data.pth
    ├── ESS.png
    ├── samples.png
    ├── resample.png
    └── qt.png                   # QT diagnostic visualisation (output of test_qt.py)
```

`test_qt.py` (present in Annulus, Himmelblau, Threewell) is a standalone diagnostic that runs Quench-and-Temper
on the target and writes `qt.png`; it is not needed to reproduce the main paper results. `core.png` (present in
Chessboard, Python) is a target-potential visualisation produced by running `core.py` as a script; also not
needed for the main results.

## Outputs

A completed `train.py` + `plot_results.py` run writes into the sub-folder:

- `data.pth` — PyTorch checkpoint containing the flow `state_dict` for each method, the per-step ESS
  history, the final pushforward samples, and the final ESS scalar.
- `ESS.png`, `samples.png`, `resample.png` — the three result figures.

Quantitative results (ESS values, coverage scores) are reported in `Paper/main.tex` §4.1. The legacy
`summary.md` at the top of this folder documents the theoretical setup and is kept for reference but is not
the authoritative source for paper numbers.
