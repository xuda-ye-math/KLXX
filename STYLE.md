# Coding style for the X-regularized Boltzmann generator benchmarks

This file fixes the conventions every benchmark in this repository follows: the
parameter set, the file layout, and the Algorithm-4 sampling framework with its
failure fall-backs. New benchmarks copy these verbatim. Invoke scripts with a plain `python` (use
whatever interpreter your environment provides); paths in this repo are never
machine-specific.

---

## 0. Environment

The benchmarks depend on the `zflows` package (normalizing flows + energy-based
sampling). Install it with

```bash
pip install zflows
```

For a reproducible GPU stack, use the prebuilt image
`xudayemath/zflows` (https://hub.docker.com/r/xudayemath/zflows). It needs an
NVIDIA driver **>= 580** (**>= 595 recommended**); `torch.compile` / CUDA-graph
paths (`enable_inv_ldj`, `loss_compile`, `enable_grad`) require Linux + CUDA.

---

## 1. Parameters (`parameters.py`)

One `parameters.py` per benchmark is the single source of truth — no magic
numbers in the driver. Canonical names and groups:

```python
# ---- domain & flow ----
D        = 8           # physical dimension (= L*L, d_low, etc.)
SIGMA    = 1.0         # source Gaussian width (whitened source = N(0, I))
NSF_LIM  = 4.0         # NSF/NCSF box half-width; unused by RealNVP / OT-Flow
BINS     = 16          # spline bins per coordinate
TRANSFORMS = 6         # coupling transforms in the flow
HIDDEN   = (256, 256)  # coupling-MLP hidden widths

# ---- basic training ----
N_VALID  = 80000       # validation/inference set size. NO N_TRAIN: there is no
                       #   training set; batches are drawn from the validation
                       #   set at run time. Raising N_VALID costs inference, not
                       #   training difficulty.
N_POOL   = 10000       # pool size P: shared by quench-and-temper and the
                       #   adaptive temperature selection
N_BATCH  = 2000        # batch size B: per gradient step, for both mu and hat_mu
STEPS    = 2000        # gradient steps per stage. No epochs -- only steps.
LR       = 1e-3        # Adam learning rate

# ---- optimization & rejuvenation (stability/efficiency) ----
OPT_STEP = 1e-2        # L-BFGS quench step (QT)
OPT_ITERS = 100        # L-BFGS quench iterations
MC_STEP  = 1e-3        # Langevin step
MC_ITERS = 100         # Langevin iterations (QT and plain Langevin may differ)

# ---- empirical (should be insensitive) ----
SHRINK_FACTOR = 0.7    # gamma: increment shrink on a failed stage
ADAPIVE_TAU   = 0.5    # tau: SMC ESS floor for temperature-step selection
VALIDATION_TAU = 0.3   # tau_v: validation ESS floor for stage acceptance
```

Rules:
- **No `N_TRAIN`.** Training batches are drawn at run time from the carried
  validation set; "epochs" do not exist, only `STEPS`.
- **`N_BATCH` is the difficulty knob, not `N_VALID`.** A small `B` makes the
  per-step KL gradient noisy (where the X terms help most); `N_VALID` only
  affects the final estimate's resolution.
- Potential-specific constants (physical couplings, noise level, basis sizes)
  go at the **end** of `parameters.py`, clearly separated from the canonical block.
- Whitened coordinates are a hard rule when modes have unequal scale: all
  Langevin/QT/SMC steps are isotropic, so the prior must be `N(0, I)` and any
  per-coordinate scaling lives inside the potential, never in the dynamics.

---

## 2. File layout

```
<Benchmark>/
├── core/                  # POTENTIAL-INVARIANT (copied verbatim across benchmarks)
│   ├── __init__.py
│   └── boltzmann.py       # Algorithms 3 + 4: adaptive step selection, stage
│                          #   training, validation gate, (no composed map)
├── parameters.py          # canonical block + potential-specific tail
├── potential.py           # POTENTIAL-SPECIFIC: the zflows Potential subclass,
│                          #   observables, quench-and-temper, build()
├── pilot.py               # reference sampler (PT-MALA) that freezes the
│                          #   physical parameters and certifies ground truth
├── train.py               # driver: one ladder per invocation, CLI flags
├── plot_results.py        # rebuilds the exact figures the paper includes
├── data_<tag>.pth         # saved runs incl. every per-stage state_dict (gitignored)
├── train_status.log       # tail-friendly progress log (gitignored)
├── results_table.{md,csv} # headline numbers
└── summary.md             # one-page setup + result + interpretation
```

- **`core/` is target-agnostic** and identical across benchmarks; never put
  potential-specific code there. The potential enters only as a zflows
  `Potential` object, the flow as an identity-initialized factory, the domain
  as a `wrap` map (`identity_wrap` on R^d, `wrap_torus` on the torus).
- **Always save every per-stage `state_dict`** inside `data_<tag>.pth`; figures
  and the staged-sampler census are regenerated offline from these without
  retraining.
- **Data (`*.pth`) and logs (`*.log`) are gitignored**; only code, tables, and
  paper figures are tracked.
- **Run tags encode the regime** (e.g. `data_{method}_o{sigma}.pth`) so runs at
  different parameter points cannot collide.

---

## 3. The Boltzmann generator framework (Algorithm 4)

Bridge a simple source `mu_0 ~ exp(-U_0)` to the target `mu ~ exp(-U)` with an
adaptive temperature ladder `0 = t_0 < t_1 < ... < t_K = 1`,
`U_k = (1-t_k) U_0 + t_k U`, training one flow `G_k` per stage so that
`(G_k^{-1})_# mu_{k-1} ~ mu_k`. A validation set `Y_k ~ mu_k` is carried across
stages. Per stage `k`:

1. **(i) selection pool** — draw `P` samples from `Y_{k-1}`, rejuvenate by
   Langevin on `U_{k-1}`.
2. **(ii) adaptive step (Algorithm 3)** — classical SMC from `mu_{k-1}` to a
   candidate `mu_k`; accept the largest `t_k` whose smallest per-rung ESS stays
   above `ADAPIVE_TAU`, else shrink toward `t_{k-1}` by `SHRINK_FACTOR`. Stage 1
   starts from `t_safe`; later stages extrapolate `min(1, t_{k-1} + Gamma*(t_{k-1}-t_{k-2}))`.
3. **(iii) wide-coverage set** — quench-and-temper on `U_k` gives the `hat_mu`
   pool for the mixture term.
4. **(iv) training loss** — X-regularized forward KL at balanced hyperparameters
   `KL + X_mu + X_{(hat_mu+bar_nu)/2}`; the `mu_k` batch is the Algorithm-1
   (M-rung AIS, G = G_k) surrogate.
5. **(v) validation update** — push `Y_{k-1}` through `G_k^{-1}`, reweight by the
   importance weight, **gate on the validation ESS >= VALIDATION_TAU**; on
   failure shrink `t_k` and retrain, else resample + rejuvenate to get `Y_k`.

**Compile-once discipline.** Capture `F_inv = flow.t().enable_inv_ladj()` and the
fused loss ONCE per run; mutate per-stage temperatures via 0-d buffers
(`.fill_`) or `linear_combination.set_coeffs`, never rebuild. `flow.zeros()`
resets to identity in place. Clone any `reduce-overhead` output that must survive
the next compiled call (CUDA-graph static buffers).

**Inference is staged, never composed.** The reported sampler is the carried set
after the last accepted stage; per stage: map -> reweight -> resample ->
rejuvenate. **NEVER form a composed end-to-end map or report a composed ESS** in
training, gating, or results — every metric is per-stage. (Composed ESS
multiplies per-stage errors and measures a procedure nobody runs.)

**Coarse training, fine reweighting (separation trick).** The full potential
never enters a gradient step; it appears only as an importance weight (the
final-stage gate and inference). When the target splits `x = (x_L, x_H)` and is
approximately separable, `U(x) ~ U_L(x_L) + U_H(x_H)` so `mu ~ mu_L * mu_H` with
`mu_H` directly sampleable (a Gaussian for stiff modes, the prior when the data
barely constrain `x_H`), the difficulty lives in `mu_L` alone. Train a standard
Boltzmann generator `G_L` on the low block `(U_{0,L}, U_L)`, draw `x_H ~ mu_H`
fresh, and reweight the proposal `nu_L(x_L) mu_H(x_H)` against the full target by
`w = mu(x) / (mu_L(x_L) mu_H(x_H))`, i.e. log-weight `z_L + (U_L + U_H - U)`. The
correction `U_L + U_H - U` is identically zero under exact separation and small
otherwise; the fine ESS measures it. Exact for any proposal; efficient when the
separation is good. (In code the high prior cancels, so the residual is computed
in its reduced form, e.g. `t_k * (Phi_L - Phi)` for the screened Poisson test.)

---

## 4. Failure fall-backs (in order of reach)

Stability guards already in `core/train_stage`:
- **Non-finite skip.** A non-finite loss or gradient skips the step under
  `clip_grad_norm_(GRAD_CLIP)`; `MAX_SKIP` skips per stage before abort.
- **Early abort.** Direct ESS < 0.05 at the step-500 checkpoint aborts the
  attempt (a hopeless stage should not burn the full `STEPS`).
- **Stage retry + stall rule.** A failed validation gate shrinks `t_k` and
  retrains, up to `MAX_RETRY` attempts; two consecutive accepted increments
  below `~0.005` while `t < 1` is a Zeno stall -> claim failure rather than crawl.

When a hardened (sharp) target misbehaves, escalate in this order:
1. **Tighten the clip** (`GRAD_CLIP` 1e3 -> ~100) and **halve `LR`** — cures the
   Adam runaway where the loss blows up mid-stage (curvature ~ 1/sigma^2 spikes).
2. **Raise `N_BATCH`** for lower-variance gradients (slower per step).
3. **Lower `t_safe`** (e.g. 0.2 -> 0.1) if stage 1 over-shrinks from a tiny SMC ESS.
4. **Lower `ADAPIVE_TAU`** (0.7 -> 0.5) and/or **raise `SMC_RUNGS` (M)** to make
   the selector accept usable steps in the hardened regime.
5. **Raise `OPT_ITERS` / `MC_ITERS`** so QT and rejuvenation keep up with a
   sharper misfit.
6. If it still fails, the target is genuinely too hard at this budget: **soften
   the difficulty dial** (e.g. observation noise) rather than forcing it, and
   record the boundary.

VRAM:
- **Chunk every bulk pass at <= 50k** (validation update, pool rejuvenation, SMC,
  QT, the fine extension) — bit-equivalent, bounds peak memory; the silent
  "stuck" symptom is usually an OOM into virtual VRAM, not a hang.
- Launch long jobs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Process discipline (see `.claude/skills/autonomous-research`): launch GPU jobs
directly from the main session (`run_in_background`), confirm liveness with
`nvidia-smi` + `pgrep` (bracket trick `[t]rain.py`) — never trust a launch
message; kill the python PID, not the wrapper, and verify the GPU drained.

---

## 5. Numerical-results conventions

- Report **per-stage / per-rung ESS** and the certified observable (well weights,
  phase weight, occupancy) against an exact referee (PT-MALA, transfer matrix).
- Implement the user's stated metric **exactly**; label any auxiliary number as
  not-the-metric. Never substitute an "equivalent".
- Plot scripts write the **exact PNGs** the paper `\includegraphics` points at;
  captions cite only numbers present in the saved tables.
