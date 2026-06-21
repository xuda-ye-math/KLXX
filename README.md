# Log-Likelihood-Ratio Discrepancy (X-regularized forward KL)

Numerical test suite for the paper in `Paper/main.tex` (X-functional regularization of
forward KL training for normalizing-flow Boltzmann generators). Each benchmark folder is
self-contained: parameters, training driver, saved run data, result tables, and the exact
figures `main.tex` includes by relative path. All flows use the `zflows` package
(`pip install zflows`); see `PYTHON.md` for the conda environment setup. Invoke
scripts with a plain `python`;
repo paths are never machine-specific. Data (`*.pth`) and logs (`*.log`) are
gitignored; code, tables, and paper figures are tracked.

`Poisson_Inverse/` is ongoing work (Bayesian screened-Poisson inversion with a low-mode
Boltzmann generator), not yet part of the paper; see its `PLAN.md` and `MATH.md`.
`.archive/` (gitignored) holds superseded experiments. `Review/` is an auxiliary
review/document, not part of the test suite.

## Project layout

```
Log-Likelihood-Ratio-Discrepancy/
├── Paper/                 # the manuscript: main.tex (the paper), main.pdf, references.bib
├── PYTHON.md              # conda environment setup
├── 2D_Benchmark/          # six 2D mode-discovery targets (one sub-folder per target)
├── Sensor_Array/          # Bayesian source localization
├── HD_Product/            # high-dimensional product multi-well (tables)
├── HD_Product_Ladder/     # AIS ladder-length sweep at d = 256
├── Phi4_Lattice_6/        # tilted phi^4 lattice field theory, L = 6
├── Phi4_Lattice_8/        # tilted phi^4 lattice field theory, L = 8
├── Clock_Lattice/         # adaptive-temperature Boltzmann generator (clock model)
├── Poisson_Inverse/       # Bayesian screened-Poisson source inversion
├── Molecular_BG/          # molecular Boltzmann generators (glycerol / diethanolamine / ADP)
└── tests/data/            # shared MD inputs (prmtop/rst7) for Molecular_BG
```

## Folder -> paper map (summary)

| folder | paper part | artifacts in main.tex |
|---|---|---|
| `2D_Benchmark/` | Section 5.1, `tab: 2d-benchmark`, `tab: 2d-benchmark-summary`, Figures 1-4 | per-target `samples.png`, `resample.png` |
| `Sensor_Array/` | Section 5.2, `tab: sensor-result`, Figure 5 | `samples.png`, `resample.png` |
| `HD_Product/` | Section 5.3, `tab: highd-ess` | tables only (no figures) |
| `HD_Product_Ladder/` | Section 5.3, Figure 6 `fig: highd-ladder` | `ESS_ladder.png` |
| `Phi4_Lattice_6/`, `Phi4_Lattice_8/` | Section 5.4, `tab: phi4`, Figure 7 | `Phi4_Lattice_8/figures/fig_methods.png` |
| `Clock_Lattice/` | Sections 4 + 5.5, Table `tab: clock-ladder`, Figures 8-10 | `figures/*.png` |
| `Paper/` | the manuscript | `main.tex`, `main.pdf`, `references.bib` |

---

## 2D_Benchmark/ — six 2D mode-discovery targets (Section 5.1)

Single-flow X-functional comparison on six analytic 2D targets (Threewell, Himmelblau,
Annulus, Python, Chessboard, Sparse): four losses (forward KL; +X_mu; +X_mu+X_hat_mu;
+X_mu+X_mix) trained on identical NSF flows from a Gaussian source, evaluated by final
ESS and the kNN coverage of the QT set. Demonstrates mode collapse of the non-oracle
losses and the repair by the QT-driven terms; Figures 1-4 show pushforward samples and
importance-resampled samples per target.

Some targets carry extra diagnostics: `qt.png` + `test_qt.py` (Threewell, Himmelblau,
Annulus) and `core.png` (Python, Chessboard).

Run / visualize (per target):

```bash
cd 2D_Benchmark/2D_Threewell
python train.py          # no flags; full run, writes data.pth
python plot_results.py   # rebuilds the three PNGs from data.pth
```

## Sensor_Array/ — Bayesian source localization (Section 5.2)

Six-mode permutation-symmetric Bayesian inverse problem: localize sources from a line of sensors; the posterior's wells sit ~4 prior-std out, so the non-oracle losses collapse at
high fake ESS while the QT-driven losses recover all modes. Same four-loss protocol and
pipeline as the 2D suite, plus a results table builder.

```bash
cd Sensor_Array
python train.py              # full run (--steps N for a sanity pass)
python plot_results.py
python build_table.py        # rebuilds tab: sensor-result numbers
```

## HD_Product/ — dimension sweep d = 2^k (Section 5.3, table only)

Product multi-well family: double well in the first k coordinates, standard Gaussian in
the rest, nominal dimension d = 2^k for k = 1..8 (d = 2..256). One-step IS surrogate
(M = 1, no AIS ladder). All four losses reach full coverage at every d, so the final ESS
is the sole discriminator (`tab: highd-ess`): bare KL degrades to 0.38 at d = 256 while
the QT-driven losses hold a +0.18-0.22 lead at high d.

```bash
cd HD_Product
python train.py --klist 8,7   # selected k (default K_LIST is k=1..7;
                                                #  k=8 explicit; --budget S caps wall-clock)
python build_table.py         # rebuilds tab: highd-ess source tables
```

## HD_Product_Ladder/ — AIS ladder-length sweep at d = 256 (Section 5.3, Figure 6)

Same target at k = 8 (d = 256); sweeps the AIS ladder length M in {1, 2, 4, 8, 16, 32}
for the mu-surrogate of the mixture loss. The only change vs `HD_Product` is the M-rung
annealed surrogate (each extra rung reuses the single inverse with a cheap forward
reweight). Shows M accelerates the first training stage but does not move the final ESS
(`fig: highd-ladder`).

```bash
cd HD_Product_Ladder
bash run_sweep.sh                               # full sweep (or train.py --M 4 single)
python plot_ess.py            # rebuilds Figure 6 (default mlist 1,2,4,8)
```

## Phi4_Lattice_6/ and Phi4_Lattice_8/ — lattice phi^4 fake ESS (Section 5.4)

Tilted broken phase of 2D lattice phi^4 (L x L periodic, d = L^2 = 36 / 64): the
sharpest fake ESS instance of the paper. A PT-MALA pilot (`pilot.py`) freezes kappa, h
per size and certifies the reference minority-phase weight with ladder round trips;
staged training then runs the four losses with identical frozen parameters and seeds
(the two drivers differ in minor CLI details, e.g. --no-compile exists at L = 8 only).
Bare KL and KL+X_mu collapse onto one vacuum (sometimes the minority one) at ESS up to
0.93 while the oracle-driven losses recover p_+ to within 0.012 of the referee
(`tab: phi4`, 3 seeds x 2 sizes; Figure 7 from the L = 8 folder). The L = 6 folder also
holds the skewed-oracle robustness runs (QT subsampled to 5% / 25% minority).

```bash
cd Phi4_Lattice_8
python pilot.py                          # (re)build the PT referee
python train.py --methods KL --seed 0    # stage 1: confirm collapse
python train.py --seed 1                 # all four methods, next seed
python plot_results.py                   # Figure 7 + results_table
```

## Clock_Lattice/ — adaptive-temperature Boltzmann generator (Sections 4 + 5.5)

The Algorithm 3/4 reference implementation and its benchmark. `core/boltzmann.py` is
potential-invariant: adaptive temperature-step selection by classical SMC (Algorithm 3),
per-stage flow training on the compiled fused X-regularized loss, validation-ESS
acceptance gate with abort-and-shrink, and final stage composition. The benchmark is the
6-state clock model on the 8 x 8 torus (d = 64, NCSF flow, uniform source): bare KL and
the balanced loss complete identical ladders at B = 1k/10k/100k (Table `tab: clock-ladder`); the X terms
give consistently higher per-stage validation ESS, largest on the early bridges. Figures:
target structure (Figure 8), per-rung ESS curves (Figure 9), and the occupancy-bias scaling of
the staged sampler, measured log-log slope ~ -1/3, slower than the Monte Carlo -1/2 (Figure 10).

```bash
cd Clock_Lattice
python train.py --L 8 --smoke              # tiny sanity ladder first
python train.py --L 8 --method balance     # full run; for the B-sweep edit
                                                             #  N_BATCH in parameters.py per run
python train.py --L 8 --method kl
python plot_paper.py                       # Figures 8-9
python occupancy_bias_B10k.py              # Figure 10 + occupancy csv/md
```

## Conventions shared by all folders

- `parameters.py` is the single source of truth (canonical names in STYLE.md).
- Long runs append a timestamped, tail-friendly `*_status.log` in the folder (no tqdm).
- Every run saves all flow / per-stage `state_dict`s inside its `data*.pth`, so any
  figure can be regenerated without retraining.
- Plot scripts write the exact PNGs `main.tex` includes; captions cite only numbers
  present in the saved tables.
