# Log-Likelihood-Ratio Discrepancy (X-regularized forward KL)

Numerical test suite for the paper in `Paper/main.tex` (X-functional regularization of
forward KL training for normalizing-flow Boltzmann generators). Each benchmark folder is
self-contained: parameters, training driver, saved run data, result tables, and the exact
figures `main.tex` includes by relative path. All flows use the `zflows` package
(`pip install zflows`); a reproducible GPU stack is the prebuilt image
`xudayemath/zflows` (https://hub.docker.com/r/xudayemath/zflows), needing an
NVIDIA driver >= 580 (>= 595 recommended). Invoke scripts with a plain `python`;
repo paths are never machine-specific. Data (`*.pth`) and logs (`*.log`) are
gitignored; code, tables, and paper figures are tracked.

`Poisson_Inverse/` is ongoing work (Bayesian screened-Poisson inversion with a low-mode
Boltzmann generator), not yet part of the paper; see its `PLAN.md` and `MATH.md`.
`.archive/` (gitignored) holds superseded experiments. `Review/` is an auxiliary
review/response document, not part of the test suite.

## Folder -> paper map (summary)

| folder | paper part | artifacts in main.tex |
|---|---|---|
| `2D_Benchmark/` | Sec 5.1, `tab: 2d-benchmark`, `tab: 2d-benchmark-summary`, Figs 1-4 | per-target `samples.png`, `resample.png` |
| `Sensor_Array/` | Sec 5.2, `tab: sensor-result`, Fig 5 | `samples.png`, `resample.png` |
| `HD_Product/` | Sec 5.3, `tab: highd-ess` | tables only (no figures) |
| `HD_Product_Ladder/` | Sec 5.3, Fig 6 `fig: highd-ladder` | `ESS_ladder.png` |
| `Phi4_Lattice_6/`, `Phi4_Lattice_8/` | Sec 5.4, `tab: phi4`, Fig 7 | `Phi4_Lattice_8/figures/fig_methods.png` |
| `Clock_Lattice/` | Secs 4 + 5.5, Table `tab: clock-ladder`, Figs 8-10 | `figures/*.png` |
| `Paper/` | the manuscript | `main.tex`, `main.pdf`, `references.bib` |

---

## 2D_Benchmark/ — six 2D mode-discovery targets (Sec 5.1)

Single-flow X-functional comparison on six analytic 2D targets (Threewell, Himmelblau,
Annulus, Python, Chessboard, Sparse): four losses (forward KL; +X_mu; +X_mu+X_hat_mu;
+X_mu+X_mix) trained on identical NSF flows from a Gaussian source, evaluated by final
ESS and the kNN coverage of the QT set. Demonstrates mode collapse of the non-oracle
losses and the repair by the QT-driven terms; Figures 1-4 show pushforward samples and
importance-resampled samples per target.

```
2D_Benchmark/
└── 2D_<Target>/            # Threewell, Himmelblau, Annulus, Python, Chessboard, Sparse
    ├── parameters.py       # SIGMA, NSF box, BINS/TRANSFORMS/HIDDEN, BATCH, STEPS, LR
    ├── core.py             # target Potential, loss_KL / loss_X, quench_and_temper
    ├── train.py            # trains all four losses on the same target; saves data.pth
    ├── plot_results.py     # regenerates samples.png / resample.png / ESS.png
    ├── data.pth            # samples, weights, ESS histories, flow state_dicts
    ├── samples.png         # pushforward per loss        (in main.tex)
    ├── resample.png        # reweighted+resampled per loss (in main.tex)
    └── ESS.png             # training-ESS curves (diagnostic, not in the paper)
```

Some targets carry extra diagnostics: `qt.png` + `test_qt.py` (Threewell, Himmelblau,
Annulus) and `core.png` (Python, Chessboard).

Run / visualize (per target):

```bash
cd 2D_Benchmark/2D_Threewell
python train.py          # no flags; full run, writes data.pth
python plot_results.py   # rebuilds the three PNGs from data.pth
```

## Sensor_Array/ — Bayesian source localization (Sec 5.2)

Six-mode permutation-symmetric Bayesian inverse problem: localize sources from a line of sensors; the posterior's wells sit ~4 prior-std out, so the non-oracle losses collapse at
high fake ESS while the QT-driven losses recover all modes. Same four-loss protocol and
pipeline as the 2D suite, plus a results table builder.

```
Sensor_Array/
├── parameters.py / core.py / train.py     # pipeline (2D_Benchmark layout)
├── build_table.py                          # data.pth -> results_table.{md,csv}
├── recompute_coverage.py                   # standalone coverage re-evaluation
├── plot_results.py                         # samples.png / resample.png / ESS.png
├── data.pth, results_table.{md,csv}, train_status.log, full_run.stdout
└── samples.png, resample.png               # (in main.tex, Fig 5)
```

```bash
cd Sensor_Array
python train.py              # full run (--steps N for a sanity pass)
python plot_results.py
python build_table.py        # rebuilds tab: sensor-result numbers
```

## HD_Product/ — dimension sweep d = 2^k (Sec 5.3, table only)

Product multi-well family: double well in the first k coordinates, standard Gaussian in
the rest, nominal dimension d = 2^k for k = 1..8 (d = 2..256). One-step IS surrogate
(M = 1, no AIS ladder). All four losses reach full coverage at every d, so the final ESS
is the sole discriminator (`tab: highd-ess`): bare KL degrades to 0.38 at d = 256 while
the QT-driven losses hold a +0.18-0.22 lead at high d.

```
HD_Product/
├── parameters.py / core.py
├── train.py                 # sweeps K_LIST (highest d first); saves data_k{K}.pth
├── build_table.py           # data_k*.pth -> ess_table.csv, mode_*_tables, tables.md
├── data_k1.pth ... data_k8.pth
├── ess_table.csv, mode_balance_table.csv, mode_coverage_table.csv, tables.md
└── summary.md
```

```bash
cd HD_Product
python train.py --klist 8,7   # selected k (default K_LIST is k=1..7;
                                                #  k=8 explicit; --budget S caps wall-clock)
python build_table.py         # rebuilds tab: highd-ess source tables
```

## HD_Product_Ladder/ — AIS ladder-length sweep at d = 256 (Sec 5.3, Fig 6)

Same target at k = 8 (d = 256); sweeps the AIS ladder length M in {1, 2, 4, 8, 16, 32}
for the mu-surrogate of the mixture loss. The only change vs `HD_Product` is the M-rung
annealed surrogate (each extra rung reuses the single inverse with a cheap forward
reweight). Shows M accelerates the first training stage but does not move the final ESS
(`fig: highd-ladder`).

```
HD_Product_Ladder/
├── parameters.py / core.py
├── train.py                 # one flow per M:  --M {1,2,...} [--steps N]
├── run_sweep.sh             # bash sweep over all M values
├── plot_ess.py              # data_M*.pth -> ESS_ladder.png  (in main.tex)
├── data_M1.pth ... data_M32.pth
└── ESS_ladder.png, train_status.log, sweep_status.log
```

```bash
cd HD_Product_Ladder
bash run_sweep.sh                               # full sweep (or train.py --M 4 single)
python plot_ess.py            # rebuilds Fig 6 (default mlist 1,2,4,8)
```

## Phi4_Lattice_6/ and Phi4_Lattice_8/ — lattice phi^4 fake ESS (Sec 5.4)

Tilted broken phase of 2D lattice phi^4 (L x L periodic, d = L^2 = 36 / 64): the
sharpest fake ESS instance of the paper. A PT-MALA pilot (`pilot.py`) freezes kappa, h
per size and certifies the reference minority-phase weight with ladder round trips;
staged training then runs the four losses with identical frozen parameters and seeds
(the two drivers differ in minor CLI details, e.g. --no-compile exists at L = 8 only).
Bare KL and KL+X_mu collapse onto one vacuum (sometimes the minority one) at ESS up to
0.93 while the oracle-driven losses recover p_+ to within 0.012 of the referee
(`tab: phi4`, 3 seeds x 2 sizes; Fig 7 from the L = 8 folder). The L = 6 folder also
holds the skewed-oracle robustness runs (QT subsampled to 5% / 25% minority).

```
Phi4_Lattice_{6,8}/
├── parameters.py            # frozen physics (KAPPA, LAMBDA, H) + flow/training sizes
├── core.py                  # Phi4 Potential (torch.roll), losses, QT, magnetization
├── pilot.py                 # PT-MALA referee: kappa scan, barrier, p_+, roundtrips
├── pilot_results.md         # frozen pilot numbers quoted in Sec 5.4
├── phi4_reference.pth       # referee samples + traces
├── train.py                 # staged runs: --methods a,b --seed N [--qt-skew F] [--steps N]
├── plot_results.py          # figures/fig_methods.png (Fig 7, L=8) + fig_background/ess
├── data.pth, data_seed1.pth, data_seed2.pth      # 3-seed table data
├── data_skew05.pth, data_skew25.pth              # L=6 only: skewed-oracle runs
├── results_table.{md,csv}, summary.md, train_status.log
├── BACKGROUND.md            # L=6 only: phi^4 primer for non-LFT readers
└── figures/fig_methods.png  # L=8 only: the paper's Fig 7 (unused phi^4 figures
                             #   live in .archive/phi4_figures/)
```

```bash
cd Phi4_Lattice_8
python pilot.py                          # (re)build the PT referee
python train.py --methods KL --seed 0    # stage 1: confirm collapse
python train.py --seed 1                 # all four methods, next seed
python plot_results.py                   # Fig 7 + results_table
```

## Clock_Lattice/ — adaptive-temperature Boltzmann generator (Secs 4 + 5.5)

The Algorithm 3/4 reference implementation and its benchmark. `core/boltzmann.py` is
potential-invariant: adaptive temperature-step selection by classical SMC (Algorithm 3),
per-stage flow training on the compiled fused X-regularized loss, validation-ESS
acceptance gate with abort-and-shrink, and final stage composition. The benchmark is the
6-state clock model on the 8 x 8 torus (d = 64, NCSF flow, uniform source): bare KL and
the balanced loss complete identical ladders at B = 1k/10k/100k (Table `tab: clock-ladder`); the X terms
give consistently higher per-stage validation ESS, largest on the early bridges. Figures:
target structure (Fig 8), per-rung ESS curves (Fig 9), and the occupancy-bias scaling of
the staged sampler, measured log-log slope ~ -1/3, slower than the Monte Carlo -1/2 (Fig 10).

```
Clock_Lattice/
├── core/boltzmann.py        # Algorithms 3+4 (potential-invariant, reused by Poisson_Inverse)
├── potential.py             # Clock Potential, sector occupancy, torus coverage
├── parameters.py            # canonical parameter names (see STYLE.md)
├── train.py                 # one ladder per run: --L 8 --method {balance,kl} [--smoke]
├── plot_paper.py            # figures/fig_clock_target.png + fig_clock_esscurves.png (Figs 8-9)
├── occupancy_bias_B10k.py   # occupancy-bias scaling study -> Fig 10 + csv/md
├── data_L8_{balance,kl}_B{1k,10k,100k}.pth     # Table 5 (B-sweep, per-stage state_dicts)
├── rebuild_L8_balance_B10k_N1000000.pth        # 10^6-sample resample behind Fig 8
├── occupancy_bias_B10k.{csv,md}, occ_bias_B10k_status.log
├── results_table.{md,csv}, summary.md
└── figures/{fig_clock_target,fig_clock_esscurves,occupancy_bias_B10k}.png
```

```bash
cd Clock_Lattice
python train.py --L 8 --smoke              # tiny sanity ladder first
python train.py --L 8 --method balance     # full run; for the B-sweep edit
                                                             #  N_BATCH in parameters.py per run
python train.py --L 8 --method kl
python plot_paper.py                       # Figs 8-9
python occupancy_bias_B10k.py              # Fig 10 + occupancy csv/md
```

## Conventions shared by all folders

- `parameters.py` is the single source of truth (canonical names in STYLE.md).
- Long runs append a timestamped, tail-friendly `*_status.log` in the folder (no tqdm).
- Every run saves all flow / per-stage `state_dict`s inside its `data*.pth`, so any
  figure can be regenerated without retraining.
- Plot scripts write the exact PNGs `main.tex` includes; captions cite only numbers
  present in the saved tables.
