# Project status

Last updated: 2026-07-12T17:16:45-04:00 (America/New_York)

## Current state

- Repository: `/mnt/projects/X-regularization`, branch `main`, baseline HEAD
  `19b9b095e63b356fcf743761ef9859ddbb673368` (`Record jflows_md numerical
  hardening`), tracking `origin/main`. Four tracked paths are modified:
  `status.md`, `Molecular_BG/JFLOWS_MD_PLAN.md`, and the glycerol
  `parameters.py`/`train.py`; three paths are untracked: the preserved
  `debug_loss_ess/` diagnostic, `dihedrals.py`, and `dihedrals.png`. The
  diagnostic's NPZ/EQX/log payload is recovery-critical and occupies about
  290 MB, so it remains part of the ext4 mirror backup.
- **Public/private boundary:** `/mnt/projects/jflows` and
  `/mnt/projects/jflows_md` remain the public package repositories. Their
  READMEs present a conventional pip-created `.venv`, `source` activation, and
  editable `pip install -e .` interface without workstation-specific paths.
  `X-regularization` remains the private experiment tree: every active command
  activates `/home/xuda/.envs/jflows`, then uses ordinary `python` plus explicit
  live-source `PYTHONPATH=/mnt/projects/jflows` or
  `/mnt/projects/jflows:/mnt/projects/jflows_md`. This is the sole project
  `status.md`; neither public package carries one.
- Public `jflows` is clean and pushed at
  `0302829fe440b6241172b652ff914db1ebecc273` (`Repair forward AIS and quench
  correctness`), version 0.2.0. The repair uses the actual source-particle/image
  pair and matching Jacobian for the first forward-AIS correction, preserves a
  per-particle Armijo trial scale after repeated line-search failures, and gives
  OTFlow the same exact-selected-map/near-identity-warm-start contract as the
  other trainable flows. All 16 public smoke modules and the affected 2D,
  periodic-3D, 4D-Boltzmann, CNF, and OTFlow examples pass; three independent
  reviewers found no remaining material correctness or API issue. The local,
  GitHub, and ext4-mirror commit hashes are identical.
  Public `jflows_md` is clean and pushed at
  `da2251fcc2a322f80e4a7725ce7872c4eefd0302` (`Synchronize molecular training
  with jflows 0.2`), version 0.2.0. It now mirrors the direct-first forward-AIS
  semantics, uses the standard per-step ESS monitor, compares only the final
  trained flow with exact identity on the full validation set, and lets that
  selected ESS alone control post-training acceptance. The retired intermediate
  checkpoint/`selection_steps` API and target-ratio monitor are absent. A
  differentiable lin-log excess-energy regularizer is explicit and leaves the
  physical target immutable. Its complete 13-module GPU smoke suite passes from
  an isolated source copy, and three independent molecular reviews returned
  PASS. Private forward-AIS consumers in `Codes/` still require a controlled
  current-HEAD rerun; reverse-only results and reference ensembles are not
  invalidated by this AIS correction.
- **Active environment:** `/home/xuda/.envs/jflows` is a pip-only Python 3.14.6
  virtual environment. The former Conda `jflows` environment and
  `/home/xuda/.envs/jax` are retired. The current resolver-selected stack is
  JAX/JAXlib/CUDA-13 plugin/PJRT 0.10.2, Equinox 0.13.8, OpenMM and
  OpenMM-CUDA-13 8.5.2, ParmEd 4.3.1, MDTraj 1.11.1.post2, NumPy 2.4.6, SciPy
  1.18.0, Matplotlib 3.11.0, h5py 3.16.0, scikit-learn 1.9.0, and the optional
  `ambertools-unofficial` 26.0.0 command-line toolchain. `pip check`
  is clean; JAX selects `cuda:0`; and OpenMM's Reference, CPU, CUDA, and
  OpenCL installation tests agree within tolerance.
- Neither `jflows` nor `jflows_md` is installed in the local environment.
  Imports are intentionally absent without `PYTHONPATH`, while explicit roots
  resolve to the current repositories. Public readers may use editable pip
  installation; private runs continue to consume live source directly.
- **Done — canonical external bundle interface:** coordinate schema 2 and the
  rigid-motion-quotient measure remain unchanged, while public lookup names are
  now `adp_ff96_obc1`, `glycerol_gaff2_am1bcc_obc1`, and
  `diethanolamine_gaff2_am1bcc_obc1`. Old `_v2` names and their exact manifest
  hashes remain private compatibility aliases for reviewed schema-2 artifacts;
  arbitrary name, manifest, package-source, or dependency mismatches still
  reject.
- Bundle data remains deliberately outside the Python import package. Editable
  source checkouts support short-name lookup; built wheels are code-only and
  require an explicit external bundle path. `/mnt/projects/jflows_md/bundles`
  and `Molecular_BG/bundles` are byte-identical 50-file trees. Their provenance
  contains no Conda, username, workstation, temporary, or absolute installation
  path: Amber locations are normalized to `<AMBERHOME>` before hashing.
- The optional public extra `jflows_md[bundles]` installs OpenMM, ParmEd, and
  `ambertools-unofficial`. Frozen 24.8 verification remains exact and
  non-destructive; the explicit `--name`/`--output` builder mode creates a new
  externally stored bundle with the active toolchain. Real AmberTools 26 runs
  successfully produced and verified both glycerol and diethanolamine
  candidates with clean provenance; disposable candidates were removed.
- **Verification passed:** `jflows/smoke/test_flow.py` and the complete
  `jflows_md/smoke/run_all.py` suite passed on `cuda:0`; after strengthening
  the controller coverage, `smoke/test_boltzmann_checkpoints.py` passed again.
  Coverage includes all three OpenMM/JAX parity tests (maximum energy
  discrepancy `5.04e-8 kJ/mol`, maximum force RMSE `4.48e-8 kJ/mol/nm`),
  quotient Jacobians, ADP chirality, float32 execution, Mixed_NSF, MALA, SMC,
  score-free AIS, a tiny BG training stage, chunking, strict bundle closure,
  legacy artifact migration, wheel/external-data behavior, and the glycerol
  compile path (`2.45 s` energy+gradient, `5.06 s` one-step MALA in the latest
  13-module smoke). Two independent final checkpoint/API reviewers
  returned PASS with no remaining high- or medium-severity issue; a separate
  gradient audit found no
  deterministic potential, ESS, clipping, or spline-gradient cause for the
  observed ESS decline. The complete 14-module `jflows` smoke suite and focused
  `jflows_md` compatibility/artifact tests also pass with the circular-RQS
  correction; a final independent compatibility review found no high- or
  medium-severity issue.
- **Done — standard optimized-XLA 36D training smoke:** an isolated invocation
  of `Molecular_BG/glycerol_36d/train.py --smoke` ran on `cuda:0` without
  `JAX_DISABLE_MOST_OPTIMIZATIONS` or another reduced-optimization setting. It
  completed 100-step mixed-flow training attempts over 14 accepted adaptive
  levels, reached `t=1`, promoted its checked artifacts, and exited zero in
  `290.164 s`. The initial real-molecule SMC path produced its first result in
  about 24 seconds and the first packed trainer produced all 100 monitored
  steps about 14 seconds later; warm stage attempts were fast. All optimizer
  updates were applied and every sample passed the energy screen. This is an
  execution/compilation pass, not a quality result: the identity fallback won
  all 14 levels and the final direct flow-proposal ESS was only
  `1.66716545e-5`. The actual optimizer batch was 120 (not the 12,000-particle
  SMC pool). All 14 accepted per-step ESS histories had negative fitted slopes:
  their mean over steps 1--10 was `0.281636`, versus `0.090935` over steps
  91--100, with many batches reaching the one-dominant-weight floor
  `1/120 = 0.008333`.
- **Decision — smoke runs are forbidden for ESS testing:** molecular `--smoke`
  runs may be used only to verify compilation, execution, finite outputs, and
  controller wiring. Their reduced validation set, particle pool, optimizer
  batch, ladder, and step count do not provide a precise or scientifically
  comparable ESS curve and must not be used to tune or judge ESS. The current
  comparison already shows the scale dependence: stage-1 selection at
  `t=0.02` gave minimum SMC ESS `0.352` with the smoke pool and `0.512` with
  the full 200000-particle pool. All future ESS tests must use the full run
  configuration.
- **🚨 DANGER — severe unauthorized ESS monitor replacement stopped:** the
  AI-created `target-ratio C` label and history were not requested by the user
  and violated the paper's monitoring contract. The full-size glycerol KLXX
  run was killed after step 80, before any stage result was accepted or
  promoted; `.run_klxx.inprogress-1042567` is only an interrupted diagnostic.
  Per-step batch ESS is the only optimizer-loop monitoring quantity. The
  separate full-validation proposal ESS remains the stage checkpoint and
  acceptance metric. The AI-added monitor class, ratio-history alias, and
  private artifact field have been removed from source. Focused regressions and
  the complete molecular smoke suite pass, including an exact compiled check
  that each printed per-step ESS matches its returned `ess_history` value.
- **🚨 DANGER — second severe accident: unauthorized pre-ESS shrink/retry
  contained and fixed:** the full-size regularized glycerol KLXX run passed its stage-1
  SMC candidate check at `t=0.02` (minimum SMC ESS `0.993`) and completed all
  100 optimizer steps, but `jflows_md` then executed
  `zero optimizer updates -> shrink` before computing the final
  full-validation proposal ESS. It automatically began a smaller-bridge retry;
  that process was killed during retry step 50. No stage was accepted, no
  output was promoted, and no training process remains. The complete evidence
  is preserved in
  `Molecular_BG/glycerol_36d/.run_klxx_c50.inprogress-1153227/train_status.log`.
  Per-step ESS and update/finite/kept histories are diagnostics only. For a
  trained stage attempt, the better of exact identity and the final trained
  proposal on the full validation set is the sole acceptance or retry
  criterion; zero optimizer updates must still reach that final ESS comparison.
  The two early molecular shrink branches have now been replaced by diagnostic
  messages only. Focused tests prove that a zero-update attempt with final ESS
  `1.0` is accepted without retry and that a nonfinite trained final flow is
  safely excluded while identity is accepted. The complete isolated 13-module
  `jflows_md` GPU smoke suite passes. No experiment was relaunched.
- **Verified — original `jflows` never has this shrink error:** the live
  `jflows/boltzmann.py` is unmodified relative to Git HEAD and `origin/main`,
  and its SHA-256
  `9a83b9eb68143d09d9e1103b07163cd1e021bbb2452e82d643a7527290d8c2c8`
  exactly matches the dated `/mnt/games/jflows_071226` snapshot. All four
  adaptive drivers (`reverse_KL_F`, `forward_KL_G`, `forward_KLX_G`, and
  `forward_KLXX_G`) compute trained and identity full-set ESS, select the
  higher-ESS proposal, and enter post-training shrink/retry only when that
  selected final ESS is below `tau_ess`. They contain no update-history,
  kept-fraction, optimizer-count, or finiteness acceptance gate. A focused
  live-source controller audit forced every trainer to return its unchanged
  flow; all four drivers accepted attempt 1 at final ESS `1.0` and emitted no
  rejection/shrink line. Therefore this accident is confined to the molecular
  companion, so that specific molecular shrink accident did not invalidate any
  established `jflows`/`Codes/` run. The later direct-first forward-AIS repair
  is separate and does require the pending affected-private-run refresh. The
  optional pre-training `tau_smc` candidate-selection gate remains the
  original, separate ESS-based ladder mechanism; it is not a trained-stage
  acceptance substitute.
- **Prevention rule:** the paper algorithm and generic `jflows` controller are
  normative. `jflows_md` changes must be minimal and limited to molecular
  potential, mixed Euclidean/periodic domain, and necessary molecular execution
  details. Before changing adaptive control, compare it line by line with
  `jflows`, document every unavoidable deviation, and add a regression that
  forces the relevant edge case. No molecular diagnostic may become a new
  accept/reject signal. In particular, a zero-update or nonfinite trained flow
  must leave the identity/warm-start fallback available and proceed to the
  final full-validation ESS gate.
- **Stopped diagnostic — repaired c50 controller exercised at full size:** the
  user-configured glycerol KLXX rerun used `t_safe=0.1`, SMC ladder 8,
  `tau_smc=0.6`, `tau_ess=0.4`, and the full
  1000000/200000/50000 validation/pool/batch sizes. The repaired controller
  correctly continued past `zero optimizer updates` and evaluated every
  million-sample checkpoint. Attempt 1 selected identity at ESS `0.154004`,
  legitimately failed the `0.4` final gate, and shrank to `t=0.07`; the retry
  SMC ESS `0.893` was diagnostic only. The user killed the run at retry step 60
  because loss and batch ESS remained flat. No stage was accepted or promoted;
  evidence is preserved under
  `Molecular_BG/glycerol_36d/.run_klxx_c50.inprogress-1178333/` and no Python
  compute process remains.
- **Active autonomous diagnostic program:** the primary goal is an ordered,
  soft-c50 explicit-H alkane series (CH4, ethane, propane, n-butane; dimensions
  9, 18, 27, 36) with fixed-batch gradient/update audits and independently
  recomputed full-validation ESS. The explicit secondary goal is a 36D vacuum
  glycerol reconstruction close to the original `zflows_md` setup. The
  persistent contract and draft plan live under the ignored recovery-critical
  `.aris/experiments/molecular_training_diagnostics/` tree.
- **Done — production-scale stage-1 KL/ESS diagnosis:** the interrupted
  one-million-validation-particle run saved the initial flow and steps 10, 25,
  50, 100, 250, 500, 750, and 1000. Honest proposal ESS rose from `0.009114`
  to `0.189589` at step 100, then collapsed to `0.000905` at step 1000 while
  the fixed-pool training loss continued falling. Independent target holdout
  evaluation shows a growing generalization gap; forward KL also does not
  control the chi-square/Renyi-2 moment defining importance ESS. Exact
  float64 ESS recomputation, finite/infinite `e_clip` equivalence, kept fraction
  1.0, collision-tail checks, flow/Jacobian direction checks, and gradient
  replay exclude the proposed hidden ESS/sign/singularity explanations.
- **Done — original final-only stage selection restored:** per-step batch ESS
  remains the sole optimizer-loop monitor. After training, the controller
  evaluates exactly two proposals on every validation particle: exact identity
  and the final trained flow. The higher full-validation ESS is selected, and
  that selected ESS alone determines acceptance against `tau_ess`. The
  experimental intermediate checkpoint/`selection_steps` mechanism has been
  removed from code, tests, public API, and documentation.
- **Done — independent `jflows` correctness audit:** three blind reviewers
  returned clean clearance for core scientific logic, public APIs,
  serialization, examples, and smoke coverage. All 16 smoke modules passed on
  `cuda:0`; full 2D, 3D-periodic, and scaling examples passed; reduced
  CNF/OTFlow and 4D Boltzmann examples passed; and a forced incomplete 4D
  ladder correctly raised before producing a target-labelled figure. Normal
  finite NSF/NCSF reverse-KL, forward-KL, and KL+X probes remained bit-identical
  across the earlier compatibility patch. The subsequent forward-AIS initial
  correction is a deliberate scientific repair, so private `Codes/` runs that
  consume forward AIS must be rerun before their current-head provenance is
  claimed. `CNF(exact=False)` also deliberately refreshes its Hutchinson probe
  each optimizer step. X regularization retains its established empirical
  permutation V-statistic, with consistent `O(1/N)` finite-batch bias, for
  backward compatibility.
- **Done — `jflows_md` inherited-fix and dtype audit:** molecular code now
  atomically rejects any Adam update whose loss, gradients, moments, or
  resulting parameters are nonfinite; global clipping remains stable when a
  float32 sum of squares overflows; any NaN validation log weight forces stage
  ESS to zero; positive-infinite weights share mass; invalid mixed-domain,
  sampler, and ladder controls fail before compilation; and adaptive ladders
  stop on floating-point no-progress. The package inherits the corrected
  circular RQS seam and generic ESS/resampling behavior directly from current
  `jflows`; CNF, LU-mixing, one-dimensional NSF, and hard-box issues are not
  duplicated in the molecular companion. Neither package nor the active
  glycerol driver enables JAX x64 or constructs float64 JAX training arrays.
  Float64 remains only as explicit legacy-artifact compatibility and a
  host-side NumPy Boolean-mask calculation; offline diagnostic recomputation
  is separate from training. The stale design-plan sentence naming float64 as
  primary was corrected to the implemented default-float32 policy.

## Pending

- **Pending — `zflows_md` compilation engineering:** the archived PyTorch
  molecular implementation still has unresolved excessive compile latency and
  memory growth at realistic molecular sizes. Compare compilation boundaries,
  chunking, and wrapper granularity before any attempt to revive it; the
  successful `jflows_md` smoke compile does not resolve this separate issue.
- **Pending — new molecular targets:** every AmberTools-26 or otherwise changed
  small-molecule model must use a new descriptive bundle name and receive an
  explicit scientific/provenance review before promotion to the frozen registry.
- **Pending — alkane training diagnosis (primary):** independently review the
  predeclared plan, build and validate only CH4 first, audit a single KL/KLXX
  loss and gradient outside the packed scan, and require finite committed
  updates plus saved held-out ESS before advancing to ethane, propane, or
  n-butane. Do not infer scientific ESS from smoke-sized populations.
- **Pending — original-style vacuum glycerol (secondary):** extract the exact
  archived topology, `NoCutoff` Hamiltonian, coordinate convention, cap/floor,
  and hyperparameters; build a discrepancy ledger and run the same minimal
  diagnostic only after the CH4 gate is understood.
- **Pending — private forward-AIS reruns:** the public replanting/correctness
  audit is complete, but the direct-first AIS correction changes the stochastic
  path of downstream forward-AIS experiments. Rerun the affected private
  `Codes/` studies (2D benchmark, high-dimensional product, Lattice Clock, and
  Phi4 L6/L8) under current `jflows` before updating their scientific artifacts.
  Treat differences beyond floating noise as an algorithm-correction effect;
  reverse-only runs and stored reference ensembles need no rerun.
- **Pending — ESS degeneration and identity selection:** the
  standard-compiler smoke proved that the complete 36D pipeline runs quickly
  enough, but identity won every accepted level and the production-size bare-KL
  diagnostic peaked early before overfitting its fixed 200000-particle pool.
  This remains the primary unresolved molecular-training problem. The next
  authorized experiment should compare only final flow and identity on the full
  validation set, report independent holdout ESS, and compare KL+X or KLXX
  against bare KL before increasing step count. A run must select trained stages and
  achieve materially nonzero direct proposal ESS before it can be treated as a
  successful Boltzmann generator.
- **Pending:** train and evaluate ADP against the exact ff96/OBC1 bundle using
  MALA and optimizer-only finite-safe `e_clip`; retain honest unclipped target
  values for MCMC, SMC, ESS, and evaluation, and do not restore sharpening.
- **Pending:** establish matched reference diagnostics for glycerol and neutral
  diethanolamine after the first production pipeline passes.
- **Pending:** decide whether the ignored FAB HDF5/NPZ reference data and ignored `Codes/` arrays/checkpoints need an external release artifact in addition to the mirror backup.
- **Pending:** integrate the completed JAX molecular backend into the paper only after BG sampling results pass the planned physical and distributional gates; then perform the final full-paper consistency audit.

## Timeline

### 2026-07-10T22:13:45-04:00 — Standalone recovery snapshot created

- Copied the complete project, including `.git`, untracked data, and ignored research artifacts, to `/mnt/backup/X-regularization_100726`.
- Verified matching source/snapshot HEAD, `git fsck --full --no-dangling`, equal payload inventories, and a checksum-clean `rsync` comparison.

### 2026-07-10T22:16:00-04:00 — JAX numerical tests adopted

- Copied `/mnt/projects/jflows/Codes` into the project as `Codes/` with matching checksums, 110 files, and 2,574,939,484 bytes.
- Confirmed the copied drivers use public `jflows` modules and contain no `zflows` imports.
- Added `*.eqx` to `.gitignore` so large Equinox checkpoints stay local.

### 2026-07-10T22:25:57-04:00 — Paper Section 5 synchronized

- Rewrote only Section 5 of `Paper_Arxiv/main.tex` from the new 2D, product multi-well, tilted φ⁴, and clock-model results.
- Copied the eight required figures from `Codes/` into `Paper_Arxiv/figures/` and verified their checksums.
- Preserved the molecular section and all earlier manuscript sections during the main synchronization pass.

### 2026-07-10T22:47:39-04:00 — Numerical presentation finalized and PDF rebuilt

- Refined headings, method names, table bolding, the complete clock per-rung table, propagation-factor and occupancy-bias definitions, figure sizing, and unavailable-rung notation.
- Method rows in the clock comparison use the explicit loss `KL + X_μ + X_(μ̂ + ν̄)/2` rather than “X-regularized.”
- Rebuilt `Paper_Arxiv/main.pdf`: 34 pages with resolved references and no reported layout warnings.

### 2026-07-10T22:56:28-04:00 — Migration checkpoint committed and pushed

- Committed 257 changed paths as `e6e8094e4290bbbf13609d4c8fb24b3475d10dd1` (`Migrate numerical experiments to jflows`): 125 additions, 129 deletions, 3 modifications, and one exact rename.
- Pushed `main` to `xuda-ye-math/X-regularization` using GitHub CLI authentication and verified the remote branch hash through the GitHub API.
- Left ignored `*.eqx`, `*.npz`, checkpoint, log, and bytecode artifacts out of Git; the pre-commit mirror at `/mnt/backup/projects/X-regularization` contains the local project state.

### 2026-07-10T23:19:48-04:00 — Fisher–Rao supplement merged into the manuscript

- Appended the Fisher–Rao gradient-flow derivations, convergence proof, and biased-target accuracy analysis to `Paper_Arxiv/main.tex` as Appendix A, titled “Fisher–Rao gradient flow with log-ratio variation.”
- Updated Section 2 and the organization paragraph to distinguish the main-text result summary from the appendix proofs; removed the obsolete cross-document reference setup and retired `Paper_Arxiv/supp.tex` and `Paper_Arxiv/supp.pdf`.
- Rebuilt `Paper_Arxiv/main.pdf`: 39 pages and 7,895,229 bytes, with resolved references and no reported overfull or underfull boxes.

### 2026-07-10T23:37:22-04:00 — Sections 1--4 evidence and consistency pass

- Reframed the headline result into two evidence-bounded regimes: the intended robustness for small or imperfect batches, and a clock-specific aggregate advantage through the largest tested batch, with the late-stage reversal at `B=125` stated explicitly.
- Independent terminology, mathematical, and editorial audits checked method names, stage/step language, map directions, theorem assumptions, empirical claims, and consistency with `Codes/style.md` and the result files.
- Corrected the training-sampler description: the implemented score-free annealing produces a biased target surrogate and is not exact AIS/SMC because rejuvenation targets the final distribution. Tightened the Fisher–Rao positivity statement and accuracy theorem to `λ < 1/2`.
- Replaced QT diffusion scale `σ_QT` by melt scale `m_e`; removed molecular-result framing from the abstract, organization, and Sections 1--4; rebuilt and visually inspected the 39-page PDF with no unresolved references or reported layout warnings.
- Preserved the user's deletion of all 47 tracked files under the obsolete root `2D_Benchmark/`; `Codes/2D_Benchmark/` remains the canonical numerical-test location.

### 2026-07-11T00:18:07-04:00 — Molecular workspaces audited and clean OpenMM rebuild started

- Classified `Molecular_BG_1/` as the preserved original PyTorch/paper-era result tree. Its ADP result is not a trustworthy physical benchmark because it used the wrong topology/force-field lineage, vacuum rather than OBC implicit solvent, and reports full-target ESS `0.000`.
- Classified `Molecular_BG_2/` as the paused PyTorch repair attempt and preserved its self-contained `zflows_md/` package as a useful read-only molecular reference parallel to jflows. Its OpenMM MD reference and OBC validation were useful, but the replacement ASMC ensemble remained incorrect and active PyTorch development has been abandoned in favor of jflows.
- Made `Molecular_BG/` the canonical active workspace. Copied the exact FAB ground-truth panel from `arXiv-2208.01893v3.tar.gz` (SHA-256 `b53d6bc8607672f48e4347f93593345b8e81a05027dd514be5ba63e5581cf7d9`), recorded its provenance, and added a tested 100 x 100 log-density replotter.
- Created and CUDA-smoke-tested a restartable OpenMM replica-exchange benchmark using Amber96 + OBC, L-alanine coordinates, and the `jflows` Conda environment. Installed Matplotlib and SciPy into that environment. The live checkpoint is ignored by Git but contains all sampled angles and replica state needed for replotting and continuation.
- Completed the initial 0.5 ns-per-replica CUDA run in 152.6 s. The 12-replica ladder maintained adjacent exchange acceptance between `0.632` and `0.722`; the saved 300 K samples cover several expected negative-phi basins and include a small positive-phi population. Rendered `Molecular_BG/adp_remd_ramachandran.png` directly from the checkpoint, establishing the requested basic OpenMM mode benchmark while leaving convergence as pending work.

### 2026-07-11T07:53:02-04:00 — Reliable two-chain ADP REMD benchmark completed

- Matched the FAB data-generation protocol with two independent OpenMM chains: 21 replicas from 300 to 1300 K, Amber96/OBC, unconstrained 1 fs Langevin dynamics, exchanges every 200 steps, and 300 K samples every 1000 steps. Each chain reached 20 ns per slot in two restartable segments; all four production segments completed below two hours.
- Saved 18,000 strictly post-burn-in frames per chain. At 36 bins, cross-chain JS divergence is `0.01229` bits and per-chain divergence to the authors' one-million-frame Zenodo training data is `0.01104`/`0.01172` bits. All 21 walkers in both chains traversed the complete ladder.
- Regenerated the final FAB and OpenMM 100-bin figures solely from saved angle arrays, using identical one-bin periodic smoothing and the same fixed `1e-4`--`1` logarithmic color range. Independent visual review rated the result 8/10 and supports the claim that its orientation and dominant modes visibly resemble FAB; rare positive-phi fine structure remains qualified as sampling-limited.
- Preserved the original FAB panel from the arXiv source archive and downloaded the authors' `train.h5` from Zenodo record 6993124, verifying MD5 `8d34fdda8694ee8d6745fc45b9eb3380`; MDTraj 1.11.1 generated the compact phi/psi reference array.

### 2026-07-11T12:02:51-04:00 — Exact FAB model audited and `jflows_md` design frozen

- Corrected the target identity after independent source and numerical audits: FAB uses ff96 + OBC1 (`igb=2`), while the completed local REMD chains use `amber96_obc.xml` OBC2. The local OBC2 result remains a close and reproducible comparison but is no longer described as the exact FAB Hamiltonian.
- Verified a PDB-built `amber96.xml + implicit/obc1.xml` OpenMM system against the canonical OpenMMTools ff96/OBC1 system on 100 FAB frames: maximum energy difference `1.90e-4 kJ/mol` and force RMSE `1.21e-3 kJ/mol/nm`.
- Verified the complete million-frame FAB split is L-alanine by an invariant signed-volume test, with zero D or planar frames. Confirmed the shipped topology PDB coordinates are D and the saved minimum is L, so the implementation plan requires a corrected canonical L PDB.
- Wrote `Molecular_BG/JFLOWS_MD_PLAN.md`: bundle-driven pure-JAX `Molecular_Potential`, exact OBC1/ACE energy, log-bond/logit-angle BAT coordinates, smooth ADP L half-chart, mixed spline flow on molecule-specific `R^p x T^q`, torus-correct MALA, potential-space SMC, forward KL + X, finite-safe optimizer-only `e_clip`, and explicitly no sharpening.
- Retained `Molecular_BG/adp_fab.png` as a fresh raw histogram of exactly 1,000,000 FAB ground-truth configurations with no KDE; removed the exploratory ADP PNG variants as requested.
- Refined the proposed facade to the project naming convention, `Molecular_Potential.from_bundle(...)`: a public benchmark bundle must carry topology, a complete serialized Hamiltonian, temperature, units, atom map, provenance, and hashes; trajectory coordinates remain independent validation data. Froze a three-target matrix: exact FAB L-ADP (60D), glycerol (36D), and the archived neutral diethanolamine microstate (48D). Audited the two archived small-molecule inputs as OpenFF 2.1.0/AM1-BCC vacuum systems without validated OBC1 parameters, so each new implicit-solvent target requires its own explicit GAFF2/AM1-BCC/OBC1 bundle and matched reference.
- Iterated the revised plan through independent force-field, chirality/domain, and architecture reviews. After incorporating a hash-bound `CoordinateSpec`, exact AmberTools/GAFF2 build artifacts, full small-molecule parity support, target-specific reference/gate criteria, wrapped-normal accuracy, and honest unclipped ESS, all three reviewers returned `PASS — no main issues` on round two.

### 2026-07-11T14:13:38-04:00 — `jflows_md` molecular core completed

- Added the companion `Molecular_BG/jflows_md/` package without modifying the copied base `jflows` source. Public APIs remain at package level and force-field, coordinate, chirality, validation, and builder internals live under `jflows_md/core/`.
- Implemented a bundle-driven, unsharpened `Molecular_Potential` with pure-JAX Amber/OBC1/ACE energy and exact internal-coordinate Jacobian. ADP uses an L-only half-chart; glycerol and neutral diethanolamine retain full parity support.
- Built and hash-bound three runtime targets: 60D ff96/OBC1 ADP, 36D GAFF2/AM1-BCC/OBC1 glycerol, and 48D GAFF2/AM1-BCC/OBC1 neutral diethanolamine. The relocated bundle builder reproduced all 25 scientific payload files byte-for-byte using only bundle-contained seeds.
- Added mixed-domain source sampling, torus-correct MALA, potential-space SMC, an identity flow baseline, and smoke tests. The full GPU suite passed with maximum OpenMM/JAX energy discrepancy `5.04e-8 kJ/mol` and maximum force RMSE `4.48e-8 kJ/mol/nm`.
- Minimized `Molecular_BG/` to packages, bundles, smoke tests, root documentation, and the standalone FAB reference folder. Renamed the reference figure to `reference/adp_truth.png`; the original HDF5 and derived NPZ remain local, ignored, and mirror-backed.

### 2026-07-11T20:58:39-04:00 — Public packages separated from private experiment runs

- Kept `jflows` and `jflows_md` as standalone public repositories and removed
  workstation-specific `PYTHONPATH=/mnt/projects/...` instructions from their
  public README, example, smoke, and bundle-builder interfaces. GitHub readers
  receive the conventional `pip install -e .` workflow.
- Kept `X-regularization` as the private/local experiment layer. Its Codes and
  molecular drivers continue to activate the `jflows` Conda environment and
  resolve the current public source trees through absolute `PYTHONPATH` values.
- Verified that the local Conda environment contains neither package
  distribution, that imports are absent without `PYTHONPATH`, and that the
  explicit one-root/two-root paths resolve the intended repositories. Bundle
  integrity and public API/float32 checks passed.
- Published public heads `6910c3cedfd8fc74314e74a1b475c3caec0861d9`
  (`jflows`) and `bab0f7edcc7e14855b848ef1b440f7323a18ae8d`
  (`jflows_md`).
- Retired the ignored package-root status record. Operational handoff state is
  maintained only in `/mnt/projects/X-regularization/status.md`.

### 2026-07-11T22:09:55-04:00 — Quotient target and persistence contracts corrected

- Replaced the built-in gauge-slice v1 coordinate targets by versioned v2
  targets using the rigid-motion-quotiented Cartesian configurational measure.
  An independent square Jacobian found and verified the restored anchor factor;
  full-molecule Jacobian probes for all three systems agreed to about `3e-14`.
- Hardened bundle verification to enforce hash-bound runtime specifications,
  safe relative paths, complete file-tree closure, no symlinks or special
  nodes, consistent measure metadata, exact built-in manifest pins, and exact
  v1-to-v2 lineage. Rebuilds now use non-destructive candidates and reject a
  changed seed, AmberTools version/data, or output under an existing name.
- Upgraded molecular flow artifacts to schema 2 with activation, dtype,
  coupling mask, dependency version, source hashes, positive stage count, and
  exact serialized-payload reconstruction. Retained the intentionally narrow
  explicit-target path for forensic schema-1 artifacts.
- Ran the complete `jflows_md` smoke suite in the Conda `jflows` environment
  using explicit live-source `PYTHONPATH`; all tests passed. Iterative
  independent scientific, persistence, sampling, and API reviews ended with
  three PASS reports and no remaining high- or medium-severity issue.
- Synchronized the public v2 bundle tree and schema-2 glycerol artifact driver
  into `Molecular_BG/`. No production molecular training was launched, and the
  environment-split decision remains pending.

### 2026-07-11T23:18:10-04:00 — Pip-only CUDA environment adopted

- Rebuilt `/home/xuda/.envs/jflows` as a pip-only Python 3.14 environment using
  current resolver-selected packages. Verified CUDA JAX execution, the full
  OpenMM installation test, and a clean `pip check`; retired the former Conda
  environment and `/home/xuda/.envs/jax`.
- Kept both local packages uninstalled so private runs always consume the live
  repositories through explicit `PYTHONPATH`. Updated every active private
  launch docstring and the root environment guide; public READMEs instead show
  ordinary `.venv` creation and editable pip installation. Interactive Python
  examples replace terse `python -c` checks.
- Removed Conda-specific AmberTools discovery from the synchronized bundle
  builders. Optional regeneration now discovers a coherent AmberTools prefix
  through `AMBERHOME` or `PATH`, while frozen molecular targets remain usable
  without AmberTools.
- From isolated copies, passed the standalone `jflows` flow suite and the full
  `jflows_md` suite on the pip-only GPU stack, including the three molecule
  potentials and the float32 glycerol compile path. No production training was
  run.

### 2026-07-12T00:13:17-04:00 — Canonical bundle and pip workflow finalized

- Changed active environment instructions to the user-facing sequence
  `source ~/.envs/jflows/bin/activate`, followed by ordinary `pip` and `python`
  commands. Added the installed `ambertools-unofficial` 26.0.0 toolchain to the
  reproducible environment snapshot and exposed it publicly through the
  optional `jflows_md[bundles]` extra.
- Shortened the three public bundle names to `adp_ff96_obc1`,
  `glycerol_gaff2_am1bcc_obc1`, and
  `diethanolamine_gaff2_am1bcc_obc1` without changing coordinate schema 2 or
  any physical Hamiltonian. Recomputed and pinned their manifests, synchronized
  both outer bundle trees, and retained narrow old-name/manifest/source-hash
  migration for genuine pre-rename schema-2 artifacts.
- Removed every Conda/workstation path from frozen bundle provenance and made
  transcript normalization deterministic. Clarified that wheels intentionally
  carry code only, while source checkouts and explicit external paths provide
  molecular data.
- Added and exercised the explicit AmberTools-26 `--name`/`--output` path for
  new glycerol and diethanolamine bundles. The final isolated GPU smoke suite
  passed, followed by three independent PASS reviews with no high- or
  medium-severity finding. No production molecular training was run.

### 2026-07-12T00:33:27-04:00 — Standard-XLA 36D Boltzmann smoke completed

- Ran the committed glycerol 36D `--smoke` driver from an isolated copy with
  ordinary optimized JAX/XLA on the RTX 5090. The complete adaptive bridge
  reached `t=1` in 14 accepted levels and exited zero after `290.164 s`; the
  first real-molecule compile/execute result arrived in seconds rather than
  stalling for tens of minutes.
- Verified the promoted schema-2 artifacts and exact run record: 60,000
  validation particles, 12,000 SMC particles, batch 120, 100 Adam steps per
  attempt, six SMC levels, 20 MALA steps per level, float32, all updates
  applied, and kept fraction 1.0.
- Recorded the scientific limitation separately from the compilation pass:
  the identity safeguard won every accepted level and final direct proposal
  ESS was `1.66716545e-5`, so this smoke validates execution but does not yet
  establish a useful trained generator. Every accepted stage's optimizer ESS
  trended downward, and the last level collapsed from `0.724` before its first
  update to `0.027` before its second; `LR=1e-3` is therefore the first tuning
  parameter to revisit. Removed the optional
  `JAX_DISABLE_MOST_OPTIMIZATIONS` reporting field from the public
  `jflows_md` compile benchmark and its documentation; normal package code was
  already using ordinary JAX compilation and required no change.

### 2026-07-12T02:05:38-04:00 — Molecular KL/ESS failure diagnosed

- Stopped the production-size glycerol stage after its first rejected attempt
  and retained nine 3.1-million-parameter flow states, the fixed 200000-sample
  SMC training pool, all one-million-sample validation log weights, and the
  optimizer/SMC histories under `Molecular_BG/glycerol_36d/debug_loss_ess/`.
  The diagnostic NPZ SHA-256 is
  `6a280484c01cc17467578ddc5f4b21da388cb7b6a902dcdd83d25e12716b47b7`.
- Recomputed the importance weights independently in float64 and audited map
  direction, Jacobians, clipping, collision tails, held-out target loss, and
  saved-flow gradients. The evidence identifies fixed-pool overfitting plus the
  forward-KL/chi-square objective mismatch: proposal ESS peaks near step 100
  even as the empirical training loss continues decreasing. Molecular
  singularities and a hidden ESS implementation error are excluded.
- Added opt-in sparse post-update flow snapshots and a full-validation stage
  selector that tests identity and the accepted warm start while preserving
  the original empty-schedule behavior and unchanged `tau_ess` gate. The same
  change also introduced an unauthorized AI-created replacement for the
  per-step ESS monitor; that regression was later classified as severe and
  reverted.
- Ran the complete live `jflows_md` smoke suite successfully on `cuda:0`, then
  reran the strengthened checkpoint-controller smoke. It proves multi-stage
  warm starts, exact tie behavior, unique warm-start and intermediate-flow
  selection, nonfinite-final rescue, float32 snapshot indexing, positive-melt
  KLXX, and the 32-snapshot bound. Two independent final reviewers reported no
  remaining high- or medium-severity issue. No replacement production run was
  launched.
- Updated the local `jflows` skill references to the pip-only activation and
  live-source `PYTHONPATH` workflow, and documented the molecular monitor,
  KLXX, snapshot, and selection contracts. Removed the last retired-environment
  reference from the skill tree.

### 2026-07-12T06:53:55-04:00 — Circular NCSF seam corrected and compatibility audited

- Corrected `MonotonicRQSTransform.searchsorted` only at exact equality with
  the first knot. Circular RQS/NCSF now evaluates the learned seam derivative
  and log-Jacobian instead of the identity fallback; interior knots, the upper
  endpoint, outside tails, NSF values and derivatives, public signatures, and
  Equinox pytree/serialization structure remain unchanged.
- Added exact seam, arbitrary-period representative, inverse, autodiff,
  float32, full-NCSF, bin-convention, and torus-normalization regressions. The
  complete 14-module `jflows` smoke suite and four focused `jflows_md`
  compatibility/artifact tests passed on the live pip-only CUDA stack. An
  independent read-only review reported no high- or medium-severity issue.
- Audited the original `Codes/` tree: only `Codes/Lattice_Clock` uses NCSF.
  Its reconstructed 400000-row validation source has six exact seam rows, but
  its evaluation source and saved pushforward samples have none. Away from the
  seam, saved-flow output and log-Jacobian evaluation are unchanged exactly.
  Existing scientific results therefore require no rerun; only a strict demand
  for bit-identical current-HEAD training provenance would justify rerunning
  `Codes/Lattice_Clock`. No training run was launched.

### 2026-07-12T09:10:04-04:00 — `jflows` correctness and compatibility audit completed

- Hardened approximate-CNF trace-key refresh and non-default PRNG
  serialization; finite-safe energy masking, Adam updates, and gradient
  clipping; one-dimensional NSF/NCSF conditioning; LU map/log-determinant
  consistency; degenerate ESS/resampling and ladder handling; constructor
  validation; checkpoint contracts; and example completion semantics.
- Added standalone checkpoint and edge-case regression modules and expanded
  adaptive/fixed Boltzmann interface coverage. The frozen reviewed source
  manifest has SHA-256
  `a07866d7a00b5d4274ebbbccdd9798aa1c05d7b1a686383ae56108b019d2f693`.
- Three independent final audits reported no remaining main correctness issue.
  The immutable-copy verification passed all 16 smoke modules, the full 2D,
  3D-periodic, and flow-scaling examples, reduced CNF/OTFlow and 4D Boltzmann
  runs, visual artifact checks, and the forced-incomplete-ladder failure path.
  No molecular production run or old `Codes/` rerun was launched.

### 2026-07-12T09:46:20-04:00 — `jflows_md` numerical hardening completed

- Audited every recent `jflows` correction against the molecular companion.
  Confirmed that circular splines and generic ESS/resampling are inherited,
  while approximate CNF, LU mixing, one-dimensional NSF, and uniform hard-box
  behavior are outside `jflows_md`'s mixed-coupling design.
- Corrected the duplicated molecular failure modes: overflow-prone global
  clipping, non-atomic derived Adam overflow, NaN stage weights being silently
  converted to `-inf`, permissive integer/step/source/flow controls, and
  adaptive-ladder floating-point stalls. Added exact torus-seam and focused
  molecular edge regressions.
- From isolated source copies in `/home/xuda/.envs/jflows`, all 13 molecular
  smoke modules passed on `cuda:0`, including KLX/KLXX training, checkpoint
  selection, SMC/AIS/MALA, all three stored OpenMM energy/force comparisons,
  ADP chirality, and the real float32 glycerol compile path. The dtype smokes
  explicitly verified x64 disabled and float32 samples, flow leaves, ESS,
  energies, gradients, and MALA outputs. No production training run was
  launched.

### 2026-07-12T10:01:15-04:00 — Full-size ESS testing rule adopted

- Established that reduced molecular smoke runs are compile and execution
  checks only and must never be used as ESS tests or scientific quality
  evidence. ESS comparisons and tuning decisions require the full configured
  validation, pool, batch, ladder, and training sizes.
- Stopped the reduced KLXX experiment and launched the authorized full-size
  glycerol KLXX run with `LR=1e-3`. At stage 1, the full 200000-particle SMC
  selection passed `t=0.02` with minimum ESS `0.512`. The run was subsequently
  stopped at optimizer step 80 when the missing per-step ESS interface was
  identified; it produced no accepted or promoted result.

### 2026-07-12T10:07:41-04:00 — 🚨 Severe per-step ESS regression contained

- Recorded the AI-created `target-ratio C` monitor as an unauthorized severe
  regression against the paper and user-established interface. Per-step batch
  ESS is the only permitted optimizer-loop monitoring quantity; full-validation
  proposal ESS remains a separate stage checkpoint and acceptance statistic.
- Killed the full-size glycerol KLXX process before accepting a stage. Removed
  the replacement monitor class and ratio-history naming from `jflows_md`, and
  restored the private driver to `jflows.train.Monitor` plus
  `stage_ess_history`. The required focused and complete smoke verification
  subsequently passed; no production run was automatically relaunched.

### 2026-07-12T10:25:06-04:00 — ESS interface restored and companion simplified

- Verified that the standard compiled monitor prints the same per-step ESS
  values returned in `ess_history` for molecular training. KL, KL+X, and KLXX
  controller paths expose only the ESS history name; the private glycerol NPZ
  field is `stage_ess_history`.
- Simplified `jflows_md` by directly reusing public `jflows.train.Monitor`,
  `importance_weights_log(..., "G")`, `linear_weights_from_log`, ESS,
  resampling, potential algebra, L-BFGS, and flow/spline bases. Kept mixed
  MALA, SMC/AIS, molecular sources, and checkpoint-aware training local because
  their mixed Euclidean/torus contracts differ from generic `jflows`.
- Passed the full 13-module `jflows_md` GPU suite after simplification, plus the
  focused `jflows` metrics suite and private glycerol `--help` import check.
  Searches across all three source trees found none of the removed monitor,
  history, or artifact identifiers outside this explicit danger record.

### 2026-07-12T13:54:47-04:00 — 🚨 Second severe accident contained, fixed, and original controller cleared

- Killed the full-size regularized glycerol KLXX process after the molecular
  controller used `zero optimizer updates` to shrink before final validation
  ESS and automatically started a retry. The preserved staging log shows the
  complete first 100-step attempt and retry through step 50; no stage or final
  artifact was accepted or promoted.
- Audited every adaptive acceptance path in the clean public `jflows` source,
  its Git history, and the dated ext4 snapshot. Generic `jflows` has no
  zero-update-like acceptance branch: after the optional SMC candidate gate,
  its only post-training reject/shrink condition is selected full-set proposal
  ESS below `tau_ess`.
- Ran an isolated controller regression against live `jflows` that replaced
  each of the four trainers by an unchanged-flow stub. Reverse KL, forward KL,
  KL+X, and KLXX all accepted attempt 1 on final ESS and emitted no shrink or
  rejection. Existing `jflows` and private `Codes/` results therefore need no
  rerun.
- Adopted the prevention rule that the paper and generic `jflows` algorithm are
  normative, molecular changes stay minimal, and new diagnostics can never
  control acceptance without explicit authorization and dedicated regression
  evidence.
- Removed only the two molecular pre-validation shrink branches. Zero-update
  and nonfinite-final conditions are now diagnostics; candidate selection still
  reaches the identity/warm-start/checkpoint/final full-validation ESS
  comparison. The only post-training shrink assignment remaining is below the
  final `tau_ess` gate; the two other assignments belong to the original SMC
  candidate-selection gate.
- Passed the focused controller module and the complete isolated 13-module
  `jflows_md` GPU smoke suite, including all three OpenMM/JAX potential parity
  checks and the float32 glycerol compile path. No glycerol training process is
  running and no experiment was relaunched.

### 2026-07-12T14:15:42-04:00 — Full-size repaired path stopped; controlled diagnostic goals opened

- Relaunched full glycerol KLXX with the user's updated stage/SMC parameters and
  verified the repaired behavior on real data: zero updates led to full
  checkpoint ESS evaluation, not an unauthorized shrink. The selected final
  ESS `0.154004` alone caused the first retry. The user stopped the retry at
  optimizer step 60 because neither stochastic loss nor batch ESS improved;
  no stage or output was accepted.
- Opened a bounded primary investigation over CH4, ethane, propane, and
  n-butane soft-c50 targets, with analytic gradient/update gates before any size
  scaling. Added the archived 36D vacuum glycerol reconstruction as the
  secondary goal.
- Started independent read-only reviews of alkane target design, archived
  vacuum-glycerol provenance, and the detailed `zflows` to `jflows` scientific
  replanting. The replant audit must distinguish genuine bug fixes from semantic
  changes and account for any new per-step, per-stage, compilation, or memory
  cost before it is accepted.

### 2026-07-12T17:16:45-04:00 — jflows 0.2 and jflows_md 0.2 synchronized and published

- Completed the public `jflows` repair at commit
  `0302829fe440b6241172b652ff914db1ebecc273`: forward AIS now initializes from
  the actual source/image/Jacobian pair, repeated Armijo failures retain a
  smaller per-particle trial scale, and OTFlow preserves the exact selected map
  while constructing a separate near-identity trainable continuation. All 16
  smoke modules and the affected public examples passed; three independent
  reviews returned PASS.
- Synchronized the molecular companion at commit
  `da2251fcc2a322f80e4a7725ce7872c4eefd0302`. Molecular AIS is direct-first,
  the stage controller compares only final flow and identity using full-set ESS,
  the intermediate checkpoint-selection API is retired, and the explicit
  differentiable molecular regularizer has endpoint-safe tests. The complete
  isolated 13-module GPU smoke suite and three independent reviews passed. No
  molecule training was launched during this package verification.
- Pushed both repositories to `origin/main` and verified their local, remote,
  and `/mnt/games/projects/` mirror commits. The controlled CH4-to-n-butane c50
  diagnosis remains the active molecular goal. A current-head rerun of private
  forward-AIS `Codes/` experiments is explicitly pending before their numerical
  artifacts are refreshed.
