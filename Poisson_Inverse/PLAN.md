# Poisson_Inverse: Bayesian screened-Poisson source inversion
## Low-mode Boltzmann generator x identity extension, full-potential inference

Date 2026-06-06. Chosen by 3 independent judges (unanimous) over 4 alternatives
(.aris/reviews/pde-inversion-idea/DECISION.md). Framework: Clock_Lattice/core/boltzmann.py
(Algorithm 4) reused verbatim; potential-specific files in this folder.

## 1. The idea under test
Train the flow ONLY on the low-frequency Fourier block; extend to the full space as
G_low x identity on the whitened high modes; inference reweights with the FULL posterior.
Valid when the posterior on high modes is close to the prior: the forward operator smooths
(Green's function ~ |k|^{-2}) and the prior decays fast (free design choice, s = 2).
phi^4 lesson: this FAILED for a lattice action with O(1) mode coupling -- the truncation
diagnostic (Sec. 6) gates the whole project.

## 2. Model
Unknown v(x) = sum_m theta_m phi_m(x), real Fourier basis (cos/sin half-plane pairs),
2D periodic [0,1]^2, enumerated low block first:
  - trained block: max(|k1|,|k2|) <= m_low  -> d_low modes
  - full set:      max(|k1|,|k2|) <= m_full -> d_full modes
  - smoke:    4x4 block (d_low=16)  inside 8x8  full (d_full=64)
  - headline: 6x6 block (d_low=36)  inside 12x12 full (d_full=144)
Prior theta_m ~ N(0, sigma_pr^2 / (1+|k_m|^2)^2)   [steep decay, user requirement]
Forward: (-Lap + c^2) u = G0 (1 + delta cos(alpha v) ) + eps_tilt * v, spectral solve on
N_GRID^2 (N_GRID = 4*m_full, dealiased), observe u at N_S ring sensors (bilinear),
y_obs from theta_truth at seed; likelihood N(y_obs, sigma_obs^2 I).
Posterior potential (whitened xi, theta = Sigma^{1/2} xi):
  U_full(xi) = 0.5|xi|^2 + Phi(Sigma^{1/2} xi),  Phi = |y_obs - F(theta)|^2 / (2 sigma_obs^2)
  U_low(xi_low) = same with high modes pinned to 0.
KEY IDENTITY (for the low-mode TRAINING problem only): with U_0 = 0.5|xi|^2 the
Algorithm-4 bridge is (1-t) U_0 + t U_low = 0.5|xi|^2 + t Phi_low -- the ladder tempers
exactly the low-mode likelihood. This says nothing about high modes; the extension test
(Sec. 5) is a separate measurement whose weight formula carries the full likelihood.

WHITENING IS A HARD RULE (user, 2026-06-06): high-frequency modes have very small prior
variance; the flow must NEVER see raw theta. The scaling Sigma^{1/2} lives INSIDE the
potential: the flow, the source, QT, Langevin, SMC, and the identity extension all operate
on whitened xi where EVERY coordinate is ~ N(0,1) under the prior. theta = Sigma^{1/2} xi
appears only inside Phi's forward solve. NSF_LIM is then a single number valid for all modes.
REASON (user): Langevin, QT, and SMC all use ONE isotropic step size; on raw theta the
high-mode coordinates (tiny prior variance) would need per-mode steps. Never use raw
coefficients anywhere in the dynamics.

## 3. Multimodality (structural, tiltable, referee-checkable)
cos(alpha v) symmetries: sign flip v -> -v and shift lattice v -> v + 2 pi n / alpha
(constant mode translate). Choose alpha, sigma_pr so ~3 shifts x 2 signs = 6 wells inside
prior reach (Darcy_2D precedent). eps_tilt * v breaks the sign symmetry -> nontrivial well
weights the sampler must earn (phi^4 design principle).

## 4. Training protocol (Algorithm 4, core/boltzmann.py, method='balance' and 'kl')
- Source N(0, I_{d_low}) on R^{d_low}, identity_wrap, NSF on [-NSF_LIM, NSF_LIM]^{d_low}.
- T_SAFE = 0.1 (user-fixed), SMC_RUNGS = M = 4 (user-fixed, as clock), ADAPIVE_TAU = 0.7,
  VALIDATION_TAU = 0.3, SHRINK 0.7, Gamma = 2.
- FIRST-CLASS GOAL: the balanced loss (KL + X_mu + X_mix) completes the ladder and passes
  the full-dimension test; 'kl' is the baseline expected to collapse wells.
- QT on the stage target U_k in xi_low space (R^d quench_and_temper with Gaussian melt
  sigma ~ prior, as in 2D pipeline; NOT the torus variant).
- VRAM <= 16 GB: d_low <= 36, compiled inverse known-affordable (phi^4 frontier: d<=64 ok).

## 5. Evaluation (the new test)
1. Ladder metrics per stage: t_k, val ESS (as clock Table 5).
2. EXTENSION TEST (headline): draw xi ~ N(0, I_{d_full}); push xi_low through the composed
   stage inverses, xi_high unchanged; logw = U0_full(xi) - U_full(y) + sum ladj.
   Report full-dimension ESS, well occupancies, reweighted well weights & Delta F.
3. Referee: PT-MALA at FULL dimension over the likelihood-tempered ladder t*Phi
   (phi^4 pilot machinery), roundtrip-certified; gives true well weights.
4. Control: truncated-only ESS (reweight against U_low) vs full ESS -- the gap isolates the
   high-mode contribution; near-equality certifies near-priorness post hoc.

## 6. Truncation-ceiling diagnostic (GATE, runs before any training)
(a) Spectral curvature ratio: rho_m = H^lik_mm / H^prior_mm at theta_truth and at 0, vs |k|.
    Demand rho_m << 1 for all extension modes (phi^4 had rho ~ O(1): hard fail).
(b) Var_{xi_m ~ prior} U_full(xi) holding others at truth/0, vs |k| -- decay curve.
(c) Prior-extension pilot ESS: freeze xi_low at referee posterior samples, draw xi_high from
    prior, measure ESS of full reweighting -- a direct, training-free ceiling estimate.
(c') flow-imperfection stress (review round 1): repeat (c) with the frozen low modes drawn
    from HEATED referee samples (t = 0.7, 0.5) and from prior-noised referee samples --
    an imperfect trained flow explores exactly such regions; (c) alone is blind to this.
GATE: proceed iff (a) max ratio over extension modes < 0.1 and (c) ceiling ESS > 0.5
and (c') does not collapse (> 0.2 at t = 0.7).

## 7. Stages & deliverables
S0 build + diagnostic gate (smoke geometry).      -> diagnostic.md + figures
S1 referee pilot at d_full=64 (freeze params: alpha, eps_tilt, sigma_obs giving 6 wells,
   barrier ~ 8-10 kT, minority weight 0.1-0.2).   -> pilot_results.md, reference.pth
S2a SINGLE-STAGE smoke (user): train ONE flow from the prior to the t = 0.1 (and 0.01)
    tempered posterior at 4x4 (and 6x6), observe the direct training ESS and the stage
    validation ESS. Cheapest end-to-end signal that the whitened pipeline trains at all;
    also calibrates t_safe = 0.1.                  -> single_stage.md
S2 BG smoke: full ladder 4x4 -> 8x8, balance + kl. -> data.pth, results_table.md, summary.md
S3 independent review of setup + S2 results (external agents) -> revise.
S4 headline: 6x6 -> 12x12, 3 seeds, balance + kl. -> paper-grade table + figures.
Each run saves stage state_dicts (resample/compose needs them; standing user rule).

## 8. Risks & mitigations
- Diagnostic gate fails at s=2 -> steepen prior (s=3) or raise c^2 (more smoothing); both
  are free design choices; re-run gate before any training.
- Too many/few wells after tilt -> tune alpha, sigma_pr on the pilot grid (cheap, referee-level).
- QT misses shifted wells (constant-mode direction) -> enlarge QT melt sigma along mode 0.
- Stage-1 instability at t_safe=0.1 (clock saw it at 0.25) -> guards already in train_stage.
