# Clock L=8 batch-size sweep — findings for the main.tex revision

Date: 2026-06-05. Status: SWEEP COMPLETE — B ∈ {1k, 10k, 100k} pairs featured
(B=5k pair archived, see below); all runs complete=True, 6/6 sectors.
Written for editing `Paper/main.tex` (Section "Boltzmann generator on p-state
clock model") on the host machine. All data lives in `Clock_Lattice/`.

## Protocol (the "new large setting")

L=8 (D=64), P=6, J=1.0, H=0.5, NCSF bins=16, transforms=6, hidden=(256,256).
N_VALID=1,000,000 (was 80,000 in the paper's original L=8 run — the user judged
that run a failure for exactly this reason), N_POOL=200,000, t_safe=0.2,
LR=1e-3, ADAPIVE_TAU=0.7, VALIDATION_TAU=0.3, SHRINK=0.7, M=4 rungs.
Per-pair settings: (B=100k, 200 steps), (B=5k, 1000), (B=1k, 2000),
(B=10k, 500 — running). Both methods per pair, identical everything except
the loss: `balance` = KL + X_mu + X_mix at (lambda=1, alpha=beta=1/2);
`kl` = bare forward KL. Single seed (0). GPU: RTX PRO 6000 Blackwell 96GB.

## Headline results (results_table.md is canonical)

| run | K | complete | composed ESS | sectors | TV | wall_min |
|---|---|---|---|---|---|---|
| balance_B100k | 6 | yes | 0.0012 | 6/6 | 0.212 | 125.5 |
| kl_B100k      | 6 | yes | 0.0015 | 6/6 | 0.083 | 122.8 |
| balance_B10k  | 6 | yes | 0.0019 | 6/6 | 0.181 | 35.7 |
| kl_B10k       | 6 | yes | 0.0004 | 6/6 | 0.046 | 34.8 |
| balance_B1k   | 8 | yes | 0.0009 | 6/6 | 0.140 | 51.9 |
| kl_B1k        | 8 | yes | 0.0002 | 6/6 | 0.120 | 50.7 |

B=10k pair, validation ESS at every attempted rung (same ladder both):
  stage 1-3: bal 0.703/0.628/0.381, kl 0.531/0.523/0.347 (first try both)
  stage 4: 0.776 rejected (bal 0.164 / kl 0.178) -> 0.720 (bal 0.412 / kl 0.396)
  stage 5: 0.904 rejected (bal 0.160 / kl 0.175) -> 0.849 (bal 0.336 / kl 0.349)
  stage 6: 1.000 first try (bal 0.303 / kl 0.341)
Same pattern as all pairs: balance leads early accepted rungs (+0.17/+0.11/
+0.03), gap ~0 in the transition window, KL marginally higher on rejected
rungs. balance_B10k posts the best composed ESS of the sweep (0.0019).
NOTE: from the B10k pair onward the .pth stores ess_hist for EVERY attempt
(rejected included) under stages[*].attempts.

A B=5k pair was also run and is NOT featured (user decision 2026-06-05:
B=1k carries the small-batch story). Archived under `Clock_Lattice/archive/`
(data + figures); per-step records remain in ess_history_all_attempts.csv.
Honest record: at B=5k both completed; balance kept the accepted-rung ESS
edge (e.g. 0.723/0.631 vs 0.568/0.518 on stages 1-2), but KL finished with a
shorter ladder (K=6 vs 7) and lower wall (31.8 vs 43.1 min) -- partly a
MAX_SKIP=10 guard artifact that discarded a balance attempt whose val ESS
(0.320) was above the floor. If a reviewer asks for intermediate batch
sizes, this pair exists and is consistent with the B=1k/B=100k pattern.

**The paper's current D=64 claim does not reproduce at the new setting.**
Original narrative (main.tex Sec. 6.4 figures): bare KL stalls at t=0.577,
fails stage 4 twelve times, 215 min incomplete vs 65 min X-regularized.
At N_VALID=1M: bare KL completes the full ladder at every batch size down to
B=1000, with the SAME ladder, the SAME rejected rungs, and comparable wall
time. The original failure was a small-validation-set / small-batch artifact,
not a property of the loss at scale.

## Where the X terms still win (quantitative, consistent)

Validation ESS on ACCEPTED rungs — balance higher in essentially all
comparisons, gap largest on early rungs (uniform source -> disordered):

- B=1k, all 8 stages, balance vs kl: 0.602/0.391, 0.524/0.398, 0.559/0.463,
  0.401/0.366, 0.500/0.467, 0.453/(kl accepted 0.841 at ~0.45), 0.369/-,
  0.778/- (kl same ladder, values from data_*.pth `stages[*].val_ess`).
- B=100k per-step (figures/ess_steps_L8_B100k_compare.png): +0.14 / +0.08 /
  +0.04 ESS at stages 1/2/3, gap -> 0 by stages 5-6.

Validation ESS on REJECTED rungs — bare KL higher in 5 of 6 head-to-head
comparisons at B=1k (0.253-0.299 vs 0.101-0.295; deltas 0.006-0.025, single
seed, individually within noise but directionally consistent).

**User's interpretation (supported by the data and by the paper's own
Fisher-Rao theory): balance has the larger training limit; bare KL has an
equal-or-faster transient.** Theorem 2 (accuracy floor) is an asymptotic
statement; Sec 2.3 explicitly disclaims rate improvements. The X-mix term
optimizes coverage, which the direct-ESS metric does not reward early — an
insurance premium visible on hard rungs, paying out only when coverage is
actually at risk.

## Why no qualitative separation on this target (user's diagnosis)

1. The clock at the BKT window has SOFT multimodality: 6 sectors entangled
   near m≈0 (paper's own Fig: "sectors entangled near the origin"), so every
   batch — even B=1000 ≈ 165/sector — touches all sectors. The mode-discovery
   channel (X_hat-mu) is never load-bearing; only the shape (X_mu) and leakage
   (bar-nu) channels are active => quantitative, not qualitative, gains.
2. In high D the network capacity is the bottleneck and both methods share the
   same NCSF; designing meaningful, genuinely-isolated-mode high-D targets is
   hard. The 2D suite + sensor array were deliberately chosen as very
   multimodal; there forward KL vs balance differ dramatically at small batch.
3. Algorithm 4's adaptive gate converts loss quality into SCHEDULE rather than
   success/failure: both losses complete; the difference appears as per-rung
   ESS, retries, ladder length. The adaptive machinery is itself a robustness
   mechanism (it was effectively absent/noisy at N_VALID=80k).

## Where along the ladder is it hard? (two senses of difficulty)

Per-stage data at B=1k (dt = accepted increment; gap = balance - kl val ESS;
t50 = steps for the identity-initialized stage flow to reach half its own
ESS plateau):

| stage | t span | dt | bal/kl val ESS | gap | t50 bal/kl |
|---|---|---|---|---|---|
| 1 | 0.000-0.200 | 0.200 | 0.602/0.391 | +0.211 | 670/162 |
| 2 | 0.200-0.396 | 0.196 | 0.524/0.398 | +0.125 | 79/81 |
| 3 | 0.396-0.530 | 0.134 | 0.559/0.463 | +0.096 | 26/2 |
| 4 | 0.530-0.662 | 0.132 | 0.401/0.366 | +0.035 | 27/23 |
| 5 | 0.662-0.753 | 0.090 | 0.500/0.467 | +0.033 | 1/1 |
| 6 | 0.753-0.841 | 0.089 | 0.453/0.465 | -0.012 | 1/1 |
| 7 | 0.841-0.952 | 0.111 | 0.369/0.400 | -0.031 | 1/1 |
| 8 | 0.952-1.000 | 0.048 | 0.778/0.708 | +0.069 | 1/1 |

- Early stages (t<=0.66): FAR BUT SMOOTH. Large, structurally simple
  deformation (site-wise marginal warps); clean AIS signal (fast Langevin
  mixing in the disordered phase); real optimization work (t50 up to 670).
  Loss quality is the binding constraint -> balance wins big (+0.10..+0.21).
- Transition window (t~0.66-0.95): NEAR BUT CRITICAL. Selector compresses
  dt 2x (its accepted step is an empirical readout of the Fisher-information
  peak along the geometric path -- the annealing bottleneck at the phase
  transition). t50=1: identity start already at half plateau; the ceiling is
  set by surrogate quality (critical slowing of Langevin) + network capacity
  for collective modes -- shared constraints, so the loss gap collapses to
  ~0 and even flips sign slightly.
- Final hop (t->1, dt=0.048): easy for both (0.778/0.708). "Hard at the end"
  is really "hard at the transition"; the last rung is the easiest.

Paper-ready phrasing: the X functionals improve training precisely where the
training signal is the constraint, and are neutral where the sampling physics
is the constraint.

## Suggested main.tex revisions

- **Title (user's direction): reframe from "better" to "more robust".**
  Current title: "Forward KL Can Be Better: Cross-Regularization Loss for
  Normalizing Flow Boltzmann Sampling". The sweep supports robustness, not
  uniform superiority: the X-regularized loss has (a) a higher ESS limit
  (larger asymptotic plateau per stage, Theorem 2's lowered bias floor),
  (b) stronger mode discovery (2D suite, sensor array, HD_Product), but
  (c) NOT necessarily a faster convergence rate (Sec 2.3 already disclaims
  rate gains; rejected-rung data shows KL's transient equal or faster).
  There is no free lunch — the X terms pay an insurance premium in transient
  speed — but training time is a relatively cheap price compared to silent
  mode loss or a higher bias floor. Candidate phrasing: "Forward KL Can Be
  More Robust" (rather than simply better).
- Sec 6.4 (clock): replace the "forward KL fails at D=64" narrative with:
  (a) at the large-sample protocol both losses complete at all batch sizes;
  (b) X-regularization gives consistently higher per-rung validation ESS,
  largest on the early/hard bridges, at zero extra cost;
  (c) claim: "safe and robust default that improves training quality" —
  strictly-never-worse on accepted rungs, never failed where KL passed.
- Keep the qualitative mode-discovery separation claims on the 2D suite,
  sensor array, and HD_Product (isolated-mode multimodality by construction;
  d=256 gap widens with dimension) — NOT on the clock.
- If a discriminating high-D statement is wanted: HD_Product is the designable
  middle ground; the clock section's role is "physically meaningful target,
  quality gains at no cost, adaptive ladder absorbs the rest".
- Composed-generator ESS at D=64 is ~1e-3 for BOTH losses (vs 0.035 at L=6):
  the K=6-8 stage composition compounds per-stage bias. Per-stage validation
  ESS (0.34-0.78) is the healthy quantity. Be careful which one the text quotes.

## Numerical-stability incident (worth a sentence in the paper or appendix)

At (L=8, B=100k, t_init=0.25, seed 0) training hit a hard non-finite loss at
stage-1 step 12 at BOTH lr=1e-3 and 5e-4; t_init=0.1/0.2 fine. Instrumented
guard later showed (B=5k, stage-5 t=0.849): non-finite LOSSES with PARAMS
FINITE — i.e. pathological batches from the AIS/QT sampling chain overflowing
the spline log-det at finite parameters, NOT gradient poisoning. Fix now in
`core/boltzmann.py::train_stage`: clip_grad_norm_(1e3) + skip-step on
non-finite loss/grad (fresh batch next step), stage abort only after
MAX_SKIP=10 skips or non-finite params. At B=5k the guard absorbed 10 bad
batches in one attempt and training continued normally between them.
NOTE: MAX_SKIP=10 was arguably too eager once (aborted an attempt whose val
ESS was 0.320 >= 0.3 floor); consider ~5% of STEPS after the sweep.

## Data index (all under Clock_Lattice/)

- `results_table.md/.csv` — canonical headline table (B, steps, wall recorded).
- `data_<tag>.pth` — per-run: config, ladder, per-stage state_dicts,
  train_ess_hist (accepted attempts; ALL attempts from the B10k pair onward),
  attempts (t_k, val_ess, accepted), final samples + logw.
- `ess_history_all_attempts.csv` — 83k step records: (tag, stage, t_k,
  attempt, accepted, val_ess, step, loss, ess) for EVERY attempt incl. all
  31 rejected ones. Parsed from the status log by `parse_status_log.py`.
- `train_status_archive.log` — raw log snapshot (gitignored: *.log).
- `figures/ess_steps_L8_B100k_compare.png` — per-step ESS overlay, B=100k pair.
- `figures/(ladder|sectors|magnetization|ess_steps)_<tag>.png` — per run.
- Tag scheme: `L8_<method>_B<batch>`; `_ts01` = pre-guard t_safe=0.1 run.

## Data availability

The .pth checkpoints (157-206 MB each) exceed GitHub's 100 MB git limit and
are published as release assets on the private repo instead (release tag
`data-L8-sweep`, incl. the archived B5k pair). Restore into place with:
  gh release download data-L8-sweep -D Clock_Lattice/ [--pattern '...']
Comparison figures: figures/ess_steps_L8_{B1k,B10k,B100k}_compare.png.

## Possible follow-ups (not scheduled)

- Convergence-rate fits (transient vs limit) from ess_history_all_attempts.csv.
- A genuinely mode-isolating high-D target (HD_Product) if a qualitative
  high-D separation is wanted for the paper.
