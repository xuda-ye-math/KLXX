# Boltzmann-generator lattice benchmarks — test configs (saved BEFORE coding)

Governing procedure: **Algorithm 4** (adaptive-temperature Boltzmann generator training)
of `Paper/main.tex` — steps (i) training samples, (ii) adaptive temperature selection
(Algorithm 3: SMC ESS floor `ADAPIVE_TAU=0.5`, shrink `SHRINK_FACTOR=0.7`, M = d rungs),
(iii) wide-coverage set by QT on the stage target `U_k`, (iv) X-regularized forward KL
at balanced hyperparameters (lambda=1, alpha=beta=1/2), (v) validation set update with
the post-training acceptance gate `ESS(w) >= VALIDATION_TAU=0.3` (abort-and-shrink retry).

Coding rules: parameter names and file layout follow `parameters.txt` (canonical names;
potential-invariant code in `core/`, potential-specific files in each project folder).
Old code folders (2D_Benchmark, HD_Product*, Sensor_Array, Darcy_sweep) are FROZEN.
zflows interfaces used directly: `Uniform`, `linear_combination`, `NCSF`, `NSF`,
`sequential_monte_carlo` (returns per-rung ESS list -> Algorithm 3), `langevin`, `lbfgs`,
`resample`, `compute_ESS_log`. Never run output-producing scripts inside `../zflows`.

Environment: RTX 5070 Ti 16 GB (tty mode, full VRAM), torch 2.12.0+cu130, python
`~/.envs/torch/bin/python`. Reviews: Claude subagents + OpenAI + Gemini APIs
(keys in /mnt/backup/API.txt).

HARD RULE: do NOT write to Paper/ (main.tex or any paper file) from this research
run. Results stay in the project folders and .aris/ artifacts only; paper edits
happen only on the user's explicit request.

---

## Task 1 — p-state clock model (TOP priority)   [status: in progress]

### Physics importance
The p-state clock model is the canonical bridge between the Ising (p=2) and XY (p->inf)
universality classes. Jose-Kadanoff-Kirkpatrick-Nelson (1977) showed that in 2D, for
p >= 5, it hosts TWO Berezinskii-Kosterlitz-Thouless transitions enclosing a critical
quasi-long-range-ordered phase with an *emergent* U(1) symmetry, before locking into the
p-fold discrete ordered phase at low temperature. It is the standard minimal model for
discrete symmetry breaking with emergent continuous symmetry, realized experimentally in
adsorbed monolayers on graphite, six-state clock order in hexagonal magnets, and surface
reconstructions.

### Why the posterior (cold Boltzmann measure) is non-trivial
- Exactly **p = 6 symmetry-broken sectors** (all spins near 2*pi*k/p): six tight, equal-
  weight modes — beyond the trivial Z2 two-mode case, yet exactly enumerable by QT.
- Between sectors: domain-wall saddles; within the lattice: vortex excitations -> rugged
  metastability on top of the 6 global modes.
- The annealing path uniform -> cold crosses the two BKT regions where the correlation
  length diverges exponentially: SMC/validation ESS collapse exactly there, so the
  adaptive Delta-t selection AND the new abort-and-shrink validation gate are both
  genuinely exercised (a fixed ladder fails).
- Fake-ESS risk is real: a flow that covers only a subset of the 6 sectors still posts
  high in-sector ESS — the mode-coverage diagnostic must expose it (paper's
  mode-discovery regime).

### Target
U(theta) = -J * sum_<ij> cos(theta_i - theta_j) - H * sum_i cos(P * theta_i),
2D square lattice, L x L sites, periodic boundary conditions, one angle per site,
domain = torus [-pi, pi)^D with D = L^2.

### Config
| parameter | value | note |
|---|---|---|
| L | 4 first, then 8 | D = L^2 = 16, then 64 |
| P | 6 | number of clock states (6 modes) |
| J | 1.0 | NN coupling |
| H | 0.5 | Z_p anisotropy (h->0: XY; h large: Potts-like) |
| flow | NCSF(a=[-pi]*D, b=[pi]*D) | periodic per coord, `.zeros()` identity init |
| NSF_LIM | pi | box half-width (NCSF box [-pi, pi]^D) |
| BINS / TRANSFORMS / HIDDEN | 16 / 6 / (256, 256) | per parameters.txt |
| source | `Uniform([-pi]*D, [pi]*D)` | U_0 = 0 (infinite-temperature limit) |
| ladder | U_t = (1-t) U_0 + t U = t U | via `linear_combination([U, U0], [t, 1-t])` |
| N_VALID | 80000 | validation set size (no N_TRAIN) |
| N_POOL | 10000 | QT pool and adaptive-selection particle count |
| N_BATCH | 2000 | per-gradient-step batch (mu and hat_mu draws) |
| STEPS | 2000 | gradient steps per stage (no epochs) |
| LR | 1e-3 | Adam |
| OPT_STEP / OPT_ITERS | 1e-2 / 100 | QT quench (L-BFGS, armijo) |
| MC_STEP / MC_ITERS | 1e-3 / 100 | Langevin rejuvenation |
| SMC rungs / iters-per-rung | M = D / 20 | Algorithm 3, classical SMC |
| t_k init (stage >= 2) | min(t_{k-1} + 2(t_{k-1} - t_{k-2}), 1) | overshoot extrapolation: step sizes can grow (only the first few stages are essentially difficult), so only stage 1 iterates from the full jump. OBSERVATION (L=4 runs): the classical SMC gate is NON-BINDING here (all rungs >= 0.5 even at the full jump, ESS_min 0.93+) while the post-training validation ESS rejects t=1 hard (0.025) — the validation gate carries the real adaptivity on this target |
| LAMBDA / ALPHA / BETA | 1.0 / 0.5 / 0.5 | balanced hyperparameters |
| ADAPIVE_TAU | 0.7 (raised from 0.5) | SMC ESS floor (Algorithm 3). With M=4 coarse rungs the 0.7 gate binds (L=8: trims near-1 extrapolations to ~0.83 cheaply). Tradeoff noted: HIGH floor = stability, no costly retrains early; LOW floor (0.5-0.6) = larger steps later. Keep 0.7 this series; sweep 0.6/0.5 in FUTURE tests (empirical question) |
| VALIDATION_TAU | 0.3 | post-training validation-ESS floor (Algorithm 4 gate) |
| SHRINK_FACTOR | 0.7 | step shrink on abort |
| safety | MAX_STAGES=30, MAX_RETRY=6, snap t_k=1 when 1-t_k<1e-3 | termination guards |

### Periodic-domain specifics
- After every Langevin/L-BFGS/SMC call: wrap angles back to [-pi, pi) (the potential is
  2*pi-periodic, so wrapping is exact; NCSF requires in-box inputs).
- QT on the torus: melt = fresh **uniform** draw (max-entropy melt — no Gaussian needed),
  quench = `lbfgs` on U_k, temper = `langevin` on U_k, then wrap.

### Diagnostics (results quality = HD_Product / Sensor_Array level)
- Per stage: accepted t_k, SMC ESS_min trace, per-attempt validation ESS(w), retry count.
- Final: compose G_1^{-1} ... G_K^{-1} on fresh uniform draws; direct final ESS vs U;
  **sector occupancy** over the P modes via the global magnetization phase
  arg(sum_j e^{i theta_j}) binned to nearest 2*pi*k/P (coverage = sectors with >=
  frac/P mass, plus TV balance); kNN coverage vs QT pool; save every G_k state_dict.
- Figures: ladder staircase t_k; validation-ESS per stage (with retries marked);
  sector-occupancy bar chart; magnetization scatter in the complex plane.
- Files (in Clock_Lattice/): train_status.log, data_L{L}.pth, results_table.md/.csv,
  summary.md, figures/*.png.

### Run commands
```
cd /mnt/projects/Log-Likelihood-Ratio-Discrepancy/Clock_Lattice
~/.envs/torch/bin/python train.py --L 6 --method balance   # KL + X_mu + X_mix (default method)
~/.envs/torch/bin/python train.py --L 6 --method kl        # bare forward KL only (baseline)
~/.envs/torch/bin/python train.py --L 8 --method balance   # d=64 (larger GPU)
~/.envs/torch/bin/python train.py --L 6 --smoke            # tiny sanity run first
# --method {balance,kl}, default balance; tag = L{L}_{method}, so the two
# methods write separate data_*.pth / table rows / figures. 'kl' compiles
# fused_kl_loss (mean of z_k only, no X terms, no QT pool); all else shared.
```

---

## Task 2 — frustrated antiferromagnetic XY on the triangular lattice   [status: queued]

### Physics importance
THE paradigm of geometric frustration: antiferromagnetic bonds on elementary triangles
cannot all be satisfied, so the ground state is the three-sublattice 120-degree Neel
order. Its order-parameter space is **SO(2) x Z2**: a continuous global spin rotation
TIMES a discrete chirality (the sense of rotation of the 120-degree pattern around each
upward triangle). The interplay of the Ising-like chirality transition and the BKT spin
transition (occurring at close but distinct temperatures) was a famous multi-decade
controversy (Miyashita-Shiba 1984 onward). Experimental realizations: stacked triangular
antiferromagnets (CsMnBr3, CsCuCl3), fully-frustrated Josephson-junction arrays,
helimagnets.

### Why the posterior is non-trivial
- **Two genuinely distinct chiral sectors** (kappa = +1 / -1) NOT related by any spin
  rotation — a discrete Ising-like order parameter emerging from continuous spins.
- Each sector carries a **continuous U(1) ridge** of degenerate states: the flow must
  spread mass along a 1-parameter family within each sector (harder than point modes,
  but the sector count stays 2 — enumerable for QT via the chirality label).
- Frustration makes the bare energy landscape rugged (domain walls between chiral
  domains, Z2 vortices), and the uniform -> cold path crosses both transitions.

### Target
U(theta) = +J * sum_<ij> cos(theta_i - theta_j)  (J > 0, antiferromagnetic),
triangular lattice realized as an L x L square grid with bonds (i,j)->(i,j+1),
(i,j)->(i+1,j), (i,j)->(i+1,j+1) (periodic). **L must be divisible by 3** for the
120-degree order to be commensurate: L=6 (D=36) first, then L=12 (D=144) [or L=9, D=81].

### Config
Same canonical parameters as Task 1 (NCSF, uniform source, ladder, taus); J=1.0, no H.
Chirality per upward triangle (i,j,k oriented): kappa = sign( sin(t_j - t_i) +
sin(t_k - t_j) + sin(t_i - t_k) ); order parameter = lattice average; sector metric =
occupancy of kappa = +1 / -1 among final samples + magnitude |kappa| (should be ~ 1 cold).
Figures add: chirality histogram, per-sublattice angle differences (should be ~ 120 deg).

---

## Task 3 — ANNNI continuous-spin chain   [status: backlog]
U(x) = sum_i LAMBDA_W (x_i^2-1)^2 - J1 sum x_i x_{i+1} + J2 sum x_i x_{i+2}, 1D periodic,
D=32, J1=1, kappa=J2/J1 in {0.6, 1.0}. Modulated <2> stripe phase (++--): 8 enumerable
modes (4 translations x 2 flips); devil's staircase near kappa=0.5. Plain NSF on
[-NSF_LIM, NSF_LIM]^D, Gaussian source (SIGMA=1). Physics: competing-interaction
modulated order (spatially modulated magnets, ferroelectrics).

## Task 4 — Edwards-Anderson spin glass   [status: backlog]
U(x) = sum_i LAMBDA_W (x_i^2-1)^2 - sum_<ij> J_ij x_i x_j, J_ij = +-1 quenched (fixed
seed), 2D square periodic, D=16 then 64. Exponentially many inequivalent modes; QT
stress test (expected partial coverage — report honestly). Physics: canonical disorder/
frustration model (Parisi, replica symmetry breaking).

---

## Review protocol (every task)
1. Code review BEFORE long runs: 1 Claude reviewer (repository-level, adversarial) on
   core/ + project code — correctness of Algorithm 4 wiring, sign conventions, wrapping.
2. Results review AFTER runs: Claude + OpenAI + Gemini reviewers score result quality
   (rubric: correctness evidence, mode coverage honesty, ESS interpretation, figure
   quality); >= 6/10 with no critical items to accept. Reviews stored under
   .aris/reviews/<task>/.
3. Claims go into .aris/wiki/claims/ with supported / partial / invalidated verdicts.

---

## Task 1b — occupancy-bias Monte Carlo scaling   [status: in progress]

Question: is the staged-sampler sector-occupancy bias a pure finite-size
(Monte Carlo) effect, i.e. does it scale as N^{-1/2}?

Procedure: full staged sampler of Algorithm 4 step (v) — per stage k=1..5:
load state_dict G_k (from data_L8_balance.pth, NO retraining), push chunked
compiled inverse, logw = U_{k-1}(x) - U_k(y) + ladj, resample by w, Langevin
on U_k (MC_STEP=1e-3, MC_ITERS=100), wrap. After stage 5 compute occupancy
bias err = (1/6) * sum_s |p_s - 1/6| over the p=6 magnetization sectors.

| multiplier m | N = 80000*m | independent tests | total particles |
|---|---|---|---|
| 1  | 80000   | 64 | 5.12M |
| 4  | 320000  | 16 | 5.12M |
| 16 | 1.28M   | 4  | 5.12M |
| 64 | 5.12M   | 1  | 5.12M |

Equal total work per row; report mean bias (+- std/sqrt(tests)) as a 1x4
table; N^{-1/2} predicts the bias to halve per row. Reference point: the
training run's validation set (N=80000) had TV 0.17 => bias ~ 0.057.

VRAM: chunked compiled inverse, chunk chosen adaptively from free VRAM
(160k/80k/40k), last chunk padded to keep ONE compile shape. Outputs (new
files only): occupancy_scaling.{md,csv,png}, occ_scaling_status.log.
Run by root executor in background; independent code review BEFORE launch.

### Task 1b RESULT (2026-06-04): NOT Monte Carlo rate -- in-sample gate exposed
Controls validated the harness (iid multinomial bias halves 0.0014->0.0005
from N=80k->320k; single-map pushforward constant 0.003). The staged-sampler
bias does NOT scale: mean 0.121 (N=80k, 64 tests) -> ~0.106 (N=320k). Cause
(occ_debug per-stage trace): flows overfit the carried 80000-point set --
fresh-point stage-1 ESS 0.58 vs in-sample gate 0.845 (N-independent);
fresh-chain ESS collapses to ~0.006 by stage 5; winner-take-all resampling
makes the occupancy error O(1) per run. L=8 flows are dataset-bound; the
balance-vs-kl comparison stands (kl failed the easier in-sample test).
Artifacts: .archive/occ_debug_status.log, .archive/occ_scaling_status.log.

## Task 1c -- L=6 anti-overfitting rerun   [status: running]
6x6 lattice (D=36), still p=6 sectors. Parameters updated (parameters.py):
N_VALID=400000 (5x, covers the landscape), N_POOL=100000, N_BATCH=20000,
M=SMC_RUNGS=6, HIDDEN=(192,192) (slightly smaller net), STEPS=1000, LR 1e-3
unchanged. Old L=4/L=8 data + logs moved to .archive/. Goal: a generator
whose gates are honest (fresh-sample rebuild should reproduce gate ESS).
Post-run check: occ_debug-style fresh rebuild vs gate values.
