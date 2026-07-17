# Project status

Last updated: 2026-07-17T16:59:19-04:00 (America/New_York)

## Current state

- Repository: `/data/projects/X-regularization`, branch `main`, tracking
  `origin/main`; the committed baseline is
  `1bd7b794509c58654dd7a3fb18429f207c8a7c98` (`Organize molecular workspaces
  and regularization controls`). Before this backup/commit checkpoint, the
  worktree has 15 modified tracked paths, 369 tracked deletions from the
  user-owned archive/paper moves, and two untracked top-level trees:
  `Molecular_BG/` and `Paper/`. The locked `Codes/` numerical outputs are
  unchanged; their source changes are command-path corrections only. The four
  old molecular test trees now live below ignored `.archive/` (5.6 GiB), and
  `Paper_Arxiv/` was renamed to `Paper/`. Large NPZ/HDF5/EQX/log payloads and
  the 932 MiB current molecular artifacts remain excluded from Git but
  recovery-critical in the ext4 mirror; ignored bytecode and LaTeX caches are
  disposable. All 365 files tracked at the old molecular paths are present in
  the archive; 13 retain small pre-archive working-copy edits, chiefly the
  already requested `/mnt/projects` to `/data/projects` path migration, and
  therefore require exact mirror preservation rather than reconstruction from
  the prior Git commit.
- **Public/private boundary:** `/data/projects/jflows` and
  `/data/projects/jflows_md` remain the public package repositories. Their
  READMEs present a conventional pip-created `.venv`, `source` activation, and
  editable `pip install -e .` interface without workstation-specific paths.
  `X-regularization` remains the private experiment tree: every active command
  activates `/home/xuda/.envs/jflows`, then uses ordinary `python` plus explicit
  live-source `PYTHONPATH=/data/projects/jflows` or
  `/data/projects/jflows:/data/projects/jflows_md`. This is the sole project
  `status.md`; neither public package carries one.
- Public `jflows` version 0.2.1 is clean and pushed at
  `21c5ad696211b54752792e771c2bb1ad2e7f3943` (`Release 0.2.1 without legacy
  compatibility shims`). Its annotated `0.2.0` tag preserves the exact
  pre-removal release at `f273a038d00f95e6a80167935c0e004dea566a5e`.
  Canonical 0.2.1 behavior remains numerically equivalent to 0.2.0 while the
  retired argument translations and aliases are gone.
  The published `jflows_md` 0.2.1 fallback remains at
  `a6e1948148d4b7628d2a329bd339e8be479f80fc` (`Add full-pool molecular
  training mode`): `pool_size=0` uses every current validation particle for
  SMC/training while KLXX creates an equally sized fresh-source QT population,
  and every bridge attempt starts from exact identity. The user subsequently
  and explicitly reopened **only** `jflows_md` for the e/r reconstruction.
  The verified reconstruction is now published at
  `9e1e42456f9cd9e8df488207d16a7708059e292a` (`Add molecular energy-distance
  regularization`) and protected by annotated tag `v0.3.0`. Generic `jflows`
  remains clean and intentionally unchanged at 0.2.1 for this checkpoint. Its
  later modernization is pending, not part of the molecular repair.
- **Done — local `jflows_md` e/r reconstruction:**
  `Molecular_Potential.regularized(energy_threshold_kj_mol,
  pair_distance_floor_nm=0.0)` now combines an exact-below-threshold,
  logarithmic energy-tail map with an optional pair-distance floor. The floor
  applies only to ordinary nonbonded pairs/exceptions; OBC1 and bonded terms
  retain their physical formulas. `physical_energy` remains the raw target,
  `regularized_energy` is the explicit training surrogate, and the floor-aware
  reference energy is recomputed consistently. The implementation handles
  exact collisions, small-distance OBC1 limits, float32 underflow/overflow,
  boolean/nonscalar arguments, and zero-subgradient floored distances without
  silently changing the physical endpoint. This surrogate is a diagnostic or
  training bridge, not an automatic claim of a normalized physical density.
- **Done — complete 9D--45D e/r-regularized KLXX alkane series:** the current
  `Molecular_BG/` tree contains methane, ethane, propane, n-butane, and
  n-pentane runs at dimensions 9/18/27/36/45. Every full run reached `t=1`,
  saved every pre-update batch ESS, and exited normally. Their `(e [kJ/mol],
  r [nm])` pairs and endpoint full-validation ESS are `(100, 0.1): 0.999992`,
  `(100, 0.1): 0.998610`, `(50, 0.1): 0.996787`,
  `(50, 0.15): 0.993094`, and `(50, 0.2): 0.829475`, respectively. Methane
  and ethane used 200,000 validation particles with batch 10,000; the three
  larger systems used 400,000 with batch 20,000. The complete run summaries
  report 19,500 per-step ESS rows in total. No molecular process is running.
- **Working scientific conclusion — e/r controls stability:** the user has
  identified the main regularization pair, energy threshold `e` and pair
  distance floor `r`, as the crucial factor governing molecular training
  stability. Batch size and validation population size are secondary and were
  not the decisive controls in this campaign. Keep optimizer `e_clip` separate
  from physical-potential `e`; do not substitute population scaling for tuning
  the e/r surrogate.
- **Done — explicit molecular identity initialization:** direct
  `train_*` functions accept `initialize_from_identity=False`, preserving the
  supplied flow by default. Molecular `boltzmann_*` functions default to
  `initialize_from_identity=True`; every stage/retry then starts from exact
  identity, while `False` deliberately enables stage-entry warm starts. The
  identity ESS comparator remains independent, stage acceptance remains based
  only on full-validation ESS, and checkpoint manifests now record schema 2
  plus the initialization choice on complete and incomplete paths.
- **Verified — reconstructed companion package:** two independent focused
  reviews returned PASS after all findings were repaired. The complete
  isolated `smoke/run_all.py` suite passes on `cuda:0`, including the new e/r,
  collision/OBC1, initialization, retry, checkpoint, and invalid-input cases.
  Stored OpenMM parity remains within `5.04e-8 kJ/mol` in energy and
  `4.48e-8 kJ/mol/nm` force RMSE across ADP, glycerol, and diethanolamine. The
  float32 glycerol compile path took `2.53 s` for energy+gradient and `5.27 s`
  for one MALA step. An isolated source-copy wheel builds successfully as
  `jflows_md-0.3.0-py3-none-any.whl`; neither `jflows` nor `jflows_md` is
  installed in `/home/xuda/.envs/jflows`. Commit
  `9e1e42456f9cd9e8df488207d16a7708059e292a`, `origin/main`, and the peeled
  local/remote `v0.3.0` tag target agree exactly. No production molecular
  training was launched.
- **Workspace archive ledger:** the former baseline/control trees
  `Molecular_BG_jflows_v1_nor/`, `Molecular_BG_zflows_v1_original/`,
  `Molecular_BG_zflows_v2_snapshot/`, and
  `Molecular_BG_zflows_v3_reverify/` now live below ignored `.archive/` as
  historical evidence. Git records their old tracked locations as deletions;
  the mirror backup, not Git, preserves the ignored archive payload. The active
  e/r series remains under `Molecular_BG/`. The manuscript directory was
  separately renamed from `Paper_Arxiv/` to `Paper/`.
- **Verified — reverified PyTorch control tests:** the renamed
  `.archive/Molecular_BG_zflows_v3_reverify/` suite now runs through pytest so
  fixtures and parameterized validation cases are exercised rather than
  called as zero-argument functions. All 31 tests pass in `1.38 s`, including both
  historical and c/r potential families, collision stress, explicit validation
  populations, adaptive bridge/sharpen endpoints, and local package layout.
  Generated diagnostic candidates/runs remain mirror-only through rename-safe
  `Molecular_BG*/...` ignore rules.
- **Done — fresh source-only `Codes/` rewrite:** `Codes/` now contains exactly
  20 Python source files and no historical figures, tables, arrays, logs,
  checkpoints, cache directories, manifests, hashes, provenance controllers,
  or nested Git metadata. The 2D scripts render their final figures directly;
  HD Product, Lattice Clock, and Phi4 write disposable run inputs below
  `artifacts/` and final figures/tables below `results/`. Neither output
  directory exists before a run. All 20 files parse, 99 live public `jflows`
  call sites bind to current signatures, 251 preserved literal scientific and
  training settings match the dated pre-rewrite backup, and three independent
  read-only reviews found no scientific or regeneration blocker. No model,
  reference, smoke, or result-producing script was run during the rewrite.
  The complete old tree remains in
  `/mnt/games/X-regularization_071226_231028`.
- **Done — current-head `Codes/` rerun and independent result audits:** the
  four 2D benchmarks, complete $d=2,\ldots,256$ HD Product sweep, both L=6/L=8
  Phi4 studies, and all five paired Lattice Clock configurations are finished.
  Independent reviewers recomputed the displayed values from the live raw
  arrays and found no remaining discrepancy. The fresh reports contain no
  historical-comparison framing. The 2D presentation contains only sample
  panels and final ESS/coverage, not its ESS-history curves. HD Product retains
  its requested $d=256$ ESS-history figure but drops the old total-variation
  occupancy analysis. Phi4 uses the distinct $(L,h)=(6,0.0257)$ and
  $(8,0.0144)$ targets and never bolds a collapsed high-ESS run.
- **Locked — `Codes/` numerical results:** the completed 2D Benchmark, HD
  Product, Lattice Phi4, and Lattice Clock raw result sets, reports, tables,
  and final figures are the accepted numerical checkpoint. Their independent
  raw-artifact audits are complete. Do not rerun, reseed, regenerate, or alter
  these numerical results unless the user explicitly reopens them; subsequent
  package maintenance is not authorization to change this locked checkpoint.
  The user's final `Paper/main.tex` layout adjustment is preserved, and
  the corresponding PDF was generated afterward: 34 pages, 5,274,119 bytes,
  with no unresolved-reference, overfull-box, or fatal-build marker.
- **Done — corrected Lattice Clock paired rerun and downstream rebuilds:** the
  repaired driver defines `t_hist` as the accepted KL level history, stores
  rejected trials only in `attempt_t_hist`, and enforces exact KL/KLXX schedule
  equality. At batch sizes 2000, 1000, 500, 250, and 125, KLXX improves the
  full-validation stage ESS on 36 of 37 shared levels and reduces
  $F=\prod_k\mathrm{ESS}_k^{-1}$ from
  `26.8707/53.2909/119.9999/342.3248/1058.3636` to
  `9.4661/20.0376/38.0382/90.7354/251.3791`. Direct composed-map and minibatch
  ESS are not presented as Clock results. A two-million-sample KLXX rebuild
  shows all six sectors. Fresh equal-work occupancy scaling gives slopes
  `-0.492` for KL and `-0.505` for KLXX; KLXX has lower mean bias at all nine
  particle counts, including `0.00106` versus `0.00161` at 2.56 million. The
  method-split occupancy jobs took 4267 s and 4306 s, each below two hours.
  Raw values, figures, report, and paper claims passed a final independent audit.
- **Done — c50 methane and ethane Boltzmann generators:** the canonical
  `.archive/Molecular_BG_jflows_v1_nor/methane_9d_c50` and `ethane_18d_c50`
  tests have
  minimal six-file bundles, one `parameters.py`, one `train.py`, ignored raw
  artifacts, and tracked `results/ess.md` plus `results/dihedrals.png`. Both
  full KLXX adaptive runs reached `t=1` using only full-validation ESS for
  stage acceptance. Their independent single-precision OpenMM references and
  all saved floating arrays are float32. No molecular process is running.
- **Diagnostic complete — identity-reset n-butane c20/c50 comparison:** full
  implicit-solvent 36D forward-KL and KLXX runs reached `t=1` at both caps.
  The reset removed the catastrophic stage-2 geometry/energy blow-up. Bare KL
  learned nontrivial maps at levels 1 and 3 and selected identity at most later
  levels. KLXX learned only level 1; every later attempt reported zero applied
  optimizer updates and was numerically identical to the identity fallback.
  At c20, KL finished in eight levels with ESS
  `0.421/0.432/0.382/0.365/0.327/0.382/0.338/0.834`; KLXX finished in eight
  levels with ESS `0.518/0.429/0.353/0.374/0.351/0.391/0.323/0.818`.
  Lowering c50 to c20 therefore did not repair the post-level-1 KLXX update
  failure. These are diagnostics, not promoted molecular benchmarks.
- **Control complete — historical glycerol 36D vacuum KLXX:** the standalone
  PyTorch `zflows_md` control reached `t=1` in five accepted sharpening levels
  at `t=0.1/0.24/0.436/0.828/1.0`. Full-600000-sample validation ESS is
  `0.7162/0.5541/0.6157/0.4940/0.9049`; every level stores a nonempty trained
  flow state. The first three levels were replayed from the early-stop
  checkpoint with ESS close to the originals before levels 4–5 were trained.
  The final checkpoint reports `complete=True`, and no molecular process is
  running.
- **🚨 DANGER — fifth severe accident: unauthorized public-package edit was
  started and immediately reverted:** while the authorized next comparison was
  confined to the local historical `Molecular_BG_zflows/zflows_md` control, an
  AI edit briefly changed `/data/projects/jflows_md/jflows_md/core/forcefield.py`.
  The edit was not requested, was never committed, and was reverted before any
  test or experiment. `git diff -- jflows_md/core/forcefield.py
  jflows_md/potential.py` was empty immediately after recovery. The five
  pre-existing full-pool/identity-reset files were separately reviewed, passed
  the complete smoke suite, and were later committed without any potential
  edit. The local-only scope rule in force at the time is preserved here as
  part of the accident record. It was later superseded by the user's explicit
  authorization to reconstruct `jflows_md` with the e/r surrogate and
  initialization API. That later authorization did not reopen generic
  `jflows`, did not authorize a production molecular run, and did not authorize
  a commit or push.
- **Pending — higher-dimensional JAX Molecular_BG:** implicit-solvent
  glycerol, ADP, and diethanolamine remain open. Propane/n-butane diagnostics
  and the successful PyTorch vacuum control have not yet established whether
  OBC1 solvent, potential tails, or a remaining `jflows_md` logic difference
  causes the JAX difficulty. The vacuum Hamiltonian is now a validated fallback
  for a term-by-term JAX port; no implicit-solvent result is promoted by this
  checkpoint.
- **Active environment:** `/home/xuda/.envs/jflows` is a pip-only Python 3.14.6
  virtual environment. The former Conda `jflows` environment and
  `/home/xuda/.envs/jax` are retired. The current resolver-selected stack is
  JAX/JAXlib/CUDA-13 plugin/PJRT 0.10.2, Equinox 0.13.8, OpenMM and
  OpenMM-CUDA-13 8.5.2, ParmEd 4.3.1, MDTraj 1.11.1.post2, NumPy 2.4.6, SciPy
  1.18.0, Matplotlib 3.11.0, h5py 3.16.0, scikit-learn 1.9.0, and the optional
  `ambertools-unofficial` 26.0.0 command-line toolchain. `pip check`
  is clean; JAX selects `cuda:0`; and OpenMM's Reference, CPU, CUDA, and
  OpenCL installation tests agree within tolerance.
- **Active old-framework control environment:** `/home/xuda/.envs/zflows` is a
  separate pip-only Python 3.14.6 environment. It intentionally contains no
  installed `zflows`, `zflows_md`, `jflows`, or `jflows_md`; local tests use
  `PYTHONPATH=/data/projects/X-regularization/.archive/Molecular_BG_zflows_v3_reverify`.
  Its
  dependency stack includes PyTorch 2.12.1+cu130 (matching the extant archived
  Conda environment's PyTorch release), Triton 3.7.1, OpenMM/OpenMM-CUDA-13
  8.5.2, ParmEd 4.3.1, and the scientific/figure stack. `pip check` passes and
  PyTorch sees the RTX 5090 through CUDA 13.0.
- Neither `jflows` nor `jflows_md` is installed in the local environment.
  Imports are intentionally absent without `PYTHONPATH`, while explicit roots
  resolve to the current repositories. Public readers may use editable pip
  installation; private runs continue to consume live source directly.
- **Done — minimal current bundle interface:** coordinate schema 2 and the
  rigid-motion-quotient measure remain unchanged. A bundle now contains exactly
  `manifest.json`, `system.json`, `coordinates.json`, `validation.json`,
  `system.xml`, and `reference.pdb`. The loader accepts only this current
  structure and coordinate measure; it does not translate another bundle or
  completed-run format.
- Bundle data remains deliberately outside the Python import package. Editable
  source checkouts support short-name lookup; built wheels are code-only and
  require an explicit external bundle path. The three public package bundles
  and the active methane/ethane experiment bundles use the same minimal
  six-file contract and contain no copied topology, construction transcript,
  or environment record.
- The optional public extra `jflows_md[bundles]` installs OpenMM, ParmEd, and
  `ambertools-unofficial`. The public builder takes a target, Amber topology,
  matching coordinate input, and output directory and writes only the six
  current runtime files. AmberTools is a preparation dependency, not a
  training dependency.
- **Verification passed:** the complete `jflows_md/smoke/run_all.py` suite
  passes from a fresh source copy on `cuda:0`; focused API, regularization,
  initialization, checkpoint, and controller tests also pass. Coverage includes
  all three OpenMM/JAX parity tests (maximum energy discrepancy
  `5.04e-8 kJ/mol`, maximum force RMSE `4.48e-8 kJ/mol/nm`), quotient
  Jacobians, ADP chirality, float32 execution, Mixed_NSF, MALA, SMC, score-free
  AIS, a tiny BG training stage, chunking, exact six-file bundle structure,
  current individual-flow persistence, and the glycerol compile path (`2.53 s`
  energy+gradient, `5.27 s` one-step MALA). Generic `jflows` remains at its
  previously verified clean 0.2.1 head.
- **Done — standard optimized-XLA 36D training smoke:** an isolated invocation
  of `.archive/Molecular_BG_jflows_v1_nor/glycerol_36d/train.py --smoke` ran
  on `cuda:0` without
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
- **Pending molecular diagnostic program:** the primary goal is an ordered,
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
- **🚨 DANGER — third severe accident: Lattice Clock paired-history contract
  violated; all current Clock outputs invalidated:** the fresh Clock driver fed
  KLXX the accepted KL coefficient sequence stored as `t_list`, but stored
  every adaptive KL attempt, including rejected coefficients, under `t_hist`.
  Consequently the paired KL and KLXX `t_hist` arrays were not identical even
  though their accepted `t_list` arrays were. This violates the user's required
  comparison contract that KLXX use exactly the same fixed `t_hist` as KL and
  makes the current Clock provenance ambiguous. The full occupancy analysis
  was killed during KLXX `N=10000` replication, before it saved its raw NPZ or
  final table/figure. Every pre-fix `Codes/Lattice_Clock/artifacts/` and
  `results/` output was invalidated and removed. The required repair is now
  implemented: `t_hist` is the accepted level history shared exactly by the
  pair; rejected/adaptive trials live in `attempt_t_hist`; the driver asserts
  exact schedule equality before and after KLXX training; and no old Clock
  artifact was reusable. The clean full rerun and downstream audits are now
  complete under that enforced contract.
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
  The active alkane drivers, OpenMM references, saved samples, ESS histories,
  energies, and flow parameters use float32. Offline diagnostic recomputation
  is separate from training.
- **🚨 DANGER — fourth severe accident: molecular bridge attempts omitted the
  required identity reset:** archived `zflows_md` executes `flow.zeros()`
  immediately before every KL/KLXX stage attempt, but the current molecular
  companion instead passed the last accepted incremental flow into the next
  bridge. In the 36D n-butane c50 KL diagnosis, that reused map already gave
  stage-2 validation ESS `1/400000` before the first optimizer update and
  collapsed a terminal H--H separation to `0.00414 nm`; both implicit-solvent
  and vacuum controls reproduced the failure. `jflows_md` now rebinds
  `attempt_flow = flow.zeros()` for every stage and retry, with that exact map
  also serving as the identity ESS fallback. A focused two-level/retry
  regression proves every trainer entry sees identity and passes on `cuda:0`.
  Full c50 and c20 n-butane KL/KLXX diagnostics now complete the repair check:
  the pre-update collision collapse is gone, while the independent post-level-1
  KLXX zero-update problem remains open.
- **Scope warning for locked `Codes/`:** every current and dated experiment
  driver constructs its initial NSF/NCSF with `.zeros()`. The 2D, HD Product,
  and Phi4 studies then make one direct trainer call per independently created
  flow. Lattice Clock is the sole multilevel BG use; its generic `jflows`
  controller starts level 1 from identity but warm-starts later incremental
  levels. The molecular repair does not modify frozen `jflows` or locked
  `Codes/`. Whether generic per-level warm starts should also be replaced must
  be decided by a separate paper/algorithm audit before any package edit or
  numerical rerun.

## Pending

- **Pending — apply controlled sharpening:** later experiments will adapt the
  sharpening technique in `../zflows_md`, where the e/r surrogate is annealed
  from soft to sharp and each stage records the Monte Carlo sharpening-reweight
  ESS. Port and audit that logic against current `jflows_md` before launching;
  preserve the raw physical target, per-step KLXX ESS, full-validation stage
  ESS, and the separate sharpening ESS. No sharpening implementation or run is
  authorized by this checkpoint.
- **Decision — tune e/r before population sizes:** treat the energy threshold
  and pair-distance floor as the primary stability parameters. Do not spend the
  next experiment cycle on batch-size or validation-population scaling unless
  an independent diagnostic specifically implicates finite-population error.
- **Preserved old-framework controls:** the original, snapshot, and reverified
  PyTorch workspaces now live under
  `.archive/Molecular_BG_zflows_v1_original/`,
  `.archive/Molecular_BG_zflows_v2_snapshot/`, and
  `.archive/Molecular_BG_zflows_v3_reverify/`. Their completed
  checkpoints/results are controls only and must not be overwritten by the
  future JAX e/r campaign.
- **Pending — generic `jflows` modernization after 0.2.1:** keep the current
  package and locked numerical results unchanged for now. A separate future
  design/reconstruction may add resumable Boltzmann runs, an explicit
  initialize-from-identity choice, and a simpler full-validation population
  interface that removes the present pool-size control. Audit the paper
  algorithm and Lattice Clock behavior before deciding exact stage warm-start
  semantics; none of these future features is part of `jflows_md` 0.3.0.

- **Done — fresh private 0.3.0 callers:** the methane and ethane drivers use
  descriptive e/r names and explicit identity initialization without
  retrofitting `.archive/Molecular_BG_jflows_v1_nor/`. Both full runs and their
  saved per-step ESS histories passed artifact-level consistency audits.
- **Done — alkane dimensional gates:** the full methane-through-pentane series
  is complete through 45D. Its tracked ESS tables/CSV histories and ignored raw
  particles, logs, and flow checkpoints are recovery-critical; the ignored
  payload must remain in the ext4 mirror.
- **Pending — `zflows_md` compilation engineering:** the archived PyTorch
  molecular implementation still has unresolved excessive compile latency and
  memory growth at realistic molecular sizes. Compare compilation boundaries,
  chunking, and wrapper granularity before any attempt to revive it; the
  successful `jflows_md` smoke compile does not resolve this separate issue.
- **Pending — new molecular targets:** every AmberTools-26 or otherwise changed
  small-molecule model must use a new descriptive bundle name and receive an
  explicit scientific/provenance review before promotion to the frozen registry.
- **Done — fixed-surrogate alkane scaling diagnosis:** CH4 through n-pentane
  have completed full-size KLXX runs. Any next alkane comparison should add the
  planned e/r sharpening path rather than repeat population-size scaling.
- **Pending — original-style vacuum glycerol (secondary):** extract the exact
  archived topology, `NoCutoff` Hamiltonian, coordinate convention, cap/floor,
  and hyperparameters; build a discrepancy ledger and run the same minimal
  diagnostic only after the CH4 gate is understood.
- **Done — high-dimensional fixed-surrogate stability gate:** choosing
  `e=50, r=0.15` stabilized n-butane 36D through endpoint ESS `0.993094`, and
  `e=50, r=0.2` carried n-pentane 45D through endpoint ESS `0.829475`. The next
  question is honest sharpening toward the physical target, not whether these
  fixed surrogates can finish training.
- **Pending:** train and evaluate ADP against the exact ff96/OBC1 bundle using
  MALA and optimizer-only finite-safe `e_clip`; retain honest physical target
  values for MCMC, SMC, ESS, and evaluation. Any sharpening bridge must first
  be specified and validated separately from the unchanged physical target.
- **Pending:** establish matched reference diagnostics for glycerol and neutral
  diethanolamine after the first production pipeline passes.
- **Pending:** decide whether the ignored FAB HDF5/NPZ reference data and ignored `Codes/` arrays/checkpoints need an external release artifact in addition to the mirror backup.
- **Pending:** integrate the completed JAX molecular backend into the paper only after BG sampling results pass the planned physical and distributional gates; then perform the final full-paper consistency audit.

## Timeline

### 2026-07-10T22:13:45-04:00 — Standalone recovery snapshot created

- Copied the complete project, including `.git`, untracked data, and ignored research artifacts, to `/mnt/backup/X-regularization_100726`.
- Verified matching source/snapshot HEAD, `git fsck --full --no-dangling`, equal payload inventories, and a checksum-clean `rsync` comparison.

### 2026-07-10T22:16:00-04:00 — JAX numerical tests adopted

- Copied `/data/projects/jflows/Codes` into the project as `Codes/` with matching checksums, 110 files, and 2,574,939,484 bytes.
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
  workstation-specific `PYTHONPATH=/data/projects/...` instructions from their
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
  maintained only in `/data/projects/X-regularization/status.md`.

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

### 2026-07-13T00:06:44-04:00 — Private numerical code rewrite completed

- Rebuilt `Codes/` as a clean, source-only current-API project: 20 Python files
  remain, with old figures, tables, data, logs, copied controllers, result
  narratives, caches, and presentation artifacts removed. The verified dated
  pre-rewrite snapshot remains at
  `/mnt/games/X-regularization_071226_231028`.
- Standardized current `jflows` names and stable log-space ESS evaluation while
  preserving the scientific targets, flow directions, objectives, random-key
  roles, MALA/QT/AIS behavior, and training parameters. Future temporary run
  data goes to `artifacts/`; final reproducible figures and tables go to
  `results/`, after which the temporary artifacts may be removed.
- Static verification passed without running a model: all 20 files parse, 99
  public calls bind to the live package signatures, 251 literal settings match
  the backup after canonical rename mapping, and `git diff --check` is clean.
  Independent reviews of 2D/HD, Lattice Clock, and Phi4 found no remaining
  scientific or regeneration blocker. The inherited Phi4 density label was
  corrected from `PT reference` to `mirror-MALA reference`.
- No result was regenerated. The complete current-head `Codes/` rerun and all
  Molecular_BG development remain explicitly pending.

### 2026-07-13T06:05:45-04:00 — 🚨 Lattice Clock paired-history accident stopped

- Stopped the full B=2000 occupancy-bias process during the KLXX `N=10000`
  row. It had completed the full KL scaling curve and 192 KLXX replicates, but
  had not yet written its raw NPZ, CSV, Markdown table, or figure.
- Verified all five fresh training pairs: their accepted `t_list` arrays were
  exactly equal, while their stored `t_hist` arrays differed because adaptive
  KL included rejected attempts and fixed KLXX did not. This is a provenance
  and experiment-contract failure even though the accepted numerical bridge
  coefficients were schedule-matched.
- Invalidated every current Lattice Clock artifact/result. The clean rerun may
  begin only after `t_hist` is made the exact shared accepted history,
  adaptive trials are renamed `attempt_t_hist`, and equality assertions cover
  both the saved pair and every downstream analysis.

### 2026-07-13T07:53:38-04:00 — Partial numerical rerun independently audited

- Checked the regenerated 2D logs against the dated logs rather than relying
  on the result narrative. All four targets preserve the archived
  collapse/full-coverage classification; the largest ESS and coverage changes
  are `0.0722` and `0.0297`. Visually inspected all four sample figures: the
  two support-local objectives still miss the hidden/far wells, while both
  coverage-pool objectives recover them.
- Audited all eight HD Product NPZ artifacts for finite arrays and compared the
  current and dated ESS tables. The equal-weight KLXX objective wins 6/8
  dimensions and every dimension d>=32; the only exceptions are d=4 and d=16,
  where another X objective leads by `0.0002` and `0.0012`. Every method still
  covers every strict mode. This is a more credible nuanced result than a
  universal-best claim and preserves the predicted high-dimensional benefit.
- Audited both Phi4 NPZ/CSV result sets and their figures. All 24 runs preserve
  their archived phase-coverage classification, every checked raw array is
  finite, and the largest change in a three-seed mean ESS is `0.010733`.
  Hence the fake-high-ESS collapse result and the two-phase recovery of the
  quench-and-temper losses are unchanged.
- Confirmed the repaired Clock runtime contract on the clean rerun. B=1000,
  B=500, and B=250 are complete with exact shared KL/KLXX `t_hist`; B=125 KL
  is active. Restored the archived presentation rule:
  only full-validation per-level ESS is reported and
  `F = product_k ESS_k^(-1)` is derived from those values. The direct composed
  ESS is excluded from the result table. At B=250 the resulting factor is
  `342.3` for KL versus `90.7` for KLXX.

### 2026-07-13T12:01:14-04:00 — Fresh numerical rerun and paper rewrite completed

- Completed all five exact-schedule Lattice Clock KL/KLXX pairs. KLXX raises
  full-validation ESS on 36/37 shared levels and reduces the staged
  propagation factor at every batch size.
- Rebuilt two million fresh KLXX clock samples and the full equal-work
  occupancy scaling from 10000 to 2560000 particles. The occupancy slopes are
  `-0.492` for KL and `-0.505` for KLXX; KLXX has lower bias at all nine sizes.
- Rewrote the per-experiment result reports and manuscript from live outputs.
  The 2D section presents sample panels and final ESS/coverage only; the HD
  section retains its ESS-history figure but no old TV analysis; Phi4 uses the
  correct size-dependent fields and collapse-aware emphasis.
- Deleted the stale paper-local figure tree and linked every completed panel to
  its live result. Unfinished molecular production panels remain explicit
  pending placeholders.
- Independent raw-artifact audits pass for 2D, HD Product, Phi4, and Clock.
  The final manuscript compiles to a visually inspected 35-page PDF with no
  missing reference/citation, overfull box, or fatal error.

### 2026-07-13T12:16:43-04:00 — Codes results locked for checkpoint

- Marked the complete current-head 2D Benchmark, HD Product, Lattice Phi4, and
  paired Lattice Clock reruns as finished. Their independently audited raw
  outputs, reports, tables, and figures are now locked and require explicit
  user authorization before any numerical regeneration or revision.
- Preserved the user's final manuscript layout adjustment. The PDF was built
  after the source change and is 34 pages and 5,274,119 bytes; the build log
  contains no unresolved-reference, overfull-box, fatal-error, or emergency-stop
  marker.
- Left two future workstreams explicitly open: removal of legacy `jflows`
  compatibility followed by its 0.2.1 release, and every Molecular_BG test
  and production result. Neither is included in this numerical-results
  checkpoint.

### 2026-07-13T13:36:42-04:00 — Public 0.2.1 API cleanup completed

- Preserved the exact `jflows_md` compatibility baseline as annotated tag
  `0.2.0` at `5db285664e32be4ece6e603ec0aad5b3ba18fe64`, then released and pushed
  version 0.2.1 at `220adaba22023fec9f9036fcc45049008f1d9070`.
- Removed retired callable wrappers, keyword translations, source-sampling
  aliases, stage-record aliases, and ordinary old-name bundle lookup. The
  package source, README, and metadata contain no retired-interface keywords;
  only private hash-gated historical-artifact readers remain. `smc` and `ais`
  are intentional current short names, not compatibility shims.
- Verified exact canonical numerical behavior against 0.2.0 over 128 saved
  arrays, passed the complete isolated GPU smoke suite and all three molecular
  OpenMM/JAX energy-force parity checks, and built a code-only wheel requiring
  `jflows>=0.2.1`. Three independent reviewers returned PASS for removal,
  cross-package consistency, smoke coverage, and equivalence.
- Verified local `main`, `origin/main`, GitHub, and the ext4 mirror at the same
  0.2.1 commit. No experiment or molecular training run was launched. Private
  unfinished molecular drivers require a canonical-name rewrite before their
  next authorized run.

### 2026-07-13T23:05:53-04:00 — Minimal bundle format and first two c50 alkane gates completed

- Simplified `jflows_md` to one current molecular bundle and flow-artifact
  format. Each bundle contains exactly six files; copied Amber topology,
  construction records, digest tables, completed-run readers, coordinate
  measure fallbacks, and migration paths are absent. The isolated 13-module GPU
  smoke suite passes, including all three OpenMM/JAX energy-force comparisons
  and the real float32 glycerol compile path. Public `jflows` remained frozen
  and clean at `21c5ad696211b54752792e771c2bb1ad2e7f3943`.
- Published the corresponding `jflows_md` cleanup at
  `2d015e16b4b0cad8e90df09e7da690480af40a94` (`Simplify molecular bundles and
  artifacts`); local `main`, `origin/main`, and GitHub agree.
- Completed the full methane 9D c50 KLXX run in three accepted levels. At
  `t=0.3/0.75/1.0`, full-200000-sample validation ESS is
  `0.958769/0.993007/0.990689`; the trained map wins every level. A 50000-frame
  single-precision OpenMM reference expanded to 1.2 million symmetry-equivalent
  torsion observations. The two marginal JS divergences are `0.013397` and
  `0.002313` bits.
- Completed the full ethane 18D c50 KLXX run in four accepted levels. At
  `t=0.2/0.5/0.95/1.0`, full-200000-sample validation ESS is
  `0.437133/0.735735/0.924835/0.992622`. The first three levels select trained
  maps; the final incremental level correctly selects identity (`0.992622`
  versus trained `0.972800`). A 50000-frame single-precision OpenMM reference
  expanded to 1.8 million five-torsion observations. Marginal JS divergences
  range from `0.018163` to `0.023436` bits.
- Both experiment folders now expose only `results/ess.md` and
  `results/dihedrals.png` as final tracked outputs. Raw trajectories, flow
  samples, per-attempt checkpoints, and logs live below ignored `artifacts/`
  and must remain in the ext4 mirror. Every saved floating array is float32;
  both figures were visually inspected.
- Stopped at ethane as instructed. Propane and all higher-dimensional molecule
  tests remain pending because validation ESS degradation and potential
  singularity are unresolved. A sharpening schedule has not been tested.

### 2026-07-14T09:14:00-04:00 — 🚨 Molecular identity-reset omission found and repaired

- Compared pure forward KL only against `/data/projects/zflows_md_backup`.
  Archived molecular KL resets the incremental flow to exact identity before
  every stage and retry; live `jflows_md` had instead reused the preceding
  accepted increment. The saved stage-2 probe proves the resulting collapse
  existed before optimization, so learning rate and optimizer batch size did
  not create it.
- Restored `attempt_flow = flow.zeros()` in the molecular controller and made
  the same exact identity the validation fallback. Replaced the warm-start
  smoke assertion with two-level and retry assertions that every attempt sees
  zero parameters. The isolated focused controller smoke passed with exit code
  zero using the live pip-only CUDA environment.
- Audited `Codes/` without rerunning it. All flows start from `.zeros()`, but
  Lattice Clock alone uses a multilevel generic `jflows` driver whose later
  levels are warm-started. Frozen `jflows` and the locked numerical results
  were not modified. The full implicit-solvent n-butane c50 KL rerun is the
  next verification step.

### 2026-07-14T11:53:29-04:00 — n-butane cap controls completed; old PyTorch control isolated

- Completed full implicit-solvent n-butane c50 and c20 runs for both bare KL
  and KLXX after restoring per-attempt identity initialization. All four
  ladders reached `t=1`, and none reproduced the former pre-update collision
  collapse. Bare KL learned selected nonidentity maps only at levels 1 and 3;
  KLXX learned level 1 but applied zero optimizer updates at every later level.
  The c20 softening did not resolve that KLXX-specific defect.
- Created `Molecular_BG_zflows/` as a standalone local copy of the historical
  PyTorch `zflows_md` package, Amber inputs, tests, documentation, and glycerol
  driver. Neither the dedicated pip-only environment nor the copied package
  imports or installs `zflows`, `jflows`, or `jflows_md`; runtime resolution is
  explicit through the local workspace `PYTHONPATH`.
- Two independent read-only audits confirmed the copied source and driver are
  byte-identical to the archived package. The early-stop vacuum configuration
  changes only `max_stages` to 3; explicit `max_retry=8` equals the old default.
  Vacuum `NoCutoff`, `r_floor=0.1`, geometric cap sharpening 100 to 200 kJ/mol,
  and delta-QT at 0.1 retain their original semantics. The auditors also noted
  that the historical driver is not literally config-only: method/raw/delta
  mode are CLI controls and 300 K/whitening-MD/compile choices are hardcoded.
  Preserve exact legacy behavior for the baseline rather than silently
  rewriting those controls.
- The first full-size launch reproduced the archived stage-1 SMC ESS `0.831`
  but PyTorch 2.13 exhausted 31.39 GiB while compiling the 12000-sample inverse
  before optimizer step 1. The new pip-only environment was aligned to PyTorch
  2.12.1+cu130, the exact release in the extant archived environment. A second
  launch was interrupted before training when the user required the package's
  smoke tests to pass first. No old result was overwritten and no control
  result has yet been claimed.

### 2026-07-14T13:30:37-04:00 — 🛟 Full vacuum control completed; solvent difficulty remains under test

- The historical glycerol 36D vacuum KLXX+delta-QT sharpening control reached
  `t=1` in five accepted levels. At `t=0.1/0.24/0.436/0.828/1.0`, SMC ESS is
  `0.8287/0.7896/0.8523/0.8479/0.9897`, full-600000-sample validation ESS is
  `0.7162/0.5541/0.6157/0.4940/0.9049`, and sharpening ESS is
  `0.9998/0.9761/0.6524/0.9730/1.0000`. The initial `t=0.3` second-level
  attempt missed the independent full-set gate, then the prescribed shrink to
  `t=0.24` passed; all other accepted levels used one attempt.
- The existing `zflows_md` resume algorithm replayed the first three saved
  levels over a fresh source population before continuing. Replay ESS
  `0.704/0.535/0.605` closely matches stored ESS `0.716/0.554/0.616`, providing
  a direct checkpoint/coordinate-chart consistency audit. The resume segment
  then trained levels 4–5, checkpointed after each, and exited normally with
  `complete=True` in 1,312 recorded seconds (2,738 seconds for the initial
  three-level segment). The read-back-verified 143,225,887-byte checkpoint,
  full 545,786-byte log, and config live in
  `Molecular_BG_zflows/Molecular_BG/glycerol_36d/`; every stage contains a
  nonempty flow state and no training process remains.
- Keep a deliberate route of retreat: if OBC1 implicit solvent proves to be the
  source of the extreme molecular-tail difficulty, reproduce the successful
  historical vacuum Hamiltonian, coordinate chart, cap sharpening, and KLXX
  logic in `jflows_md` and establish the JAX vacuum result first. There is no
  scientific reason to accept an intrinsically weaker JAX result when the old
  PyTorch logic can be ported and checked term by term.
- The preferred investigation remains to determine whether implicit solvent is
  the causal difficulty. FAB/OBC1 parity and the correct implicit-solvent
  glycerol comparison therefore remain future work, not silently abandoned;
  the vacuum path is a validated fallback and diagnostic control rather than a
  replacement claim.
- Treat the molecular rewrite as an engineering and experimental campaign, not
  an impossible benchmark. Stable and accurate performance may require careful
  controller reconciliation, potential/gradient parity checks, larger pools,
  sharpening schedules, and GPU-day-scale tuning. The project has viable exits
  and should remain active until those controlled comparisons identify the
  limiting factor.

### 2026-07-14T13:53:46-04:00 — 🚨 Unauthorized public-package edit reverted; local-only scope restored

- Began one unrequested change in the public `jflows_md` force-field source
  while the authorized comparison belonged only to the vendored historical
  `Molecular_BG_zflows/zflows_md` package. No test, experiment, commit, or push
  used that edit.
- Reverted the change immediately and verified that both public potential files
  have an empty Git diff. The five unrelated pre-existing public worktree
  modifications were preserved exactly.
- Froze the corrected scope: implement the reference-shifted c50 potential only
  in the local PyTorch `zflows_md`; retain the historical `e_cap/r_floor` path
  as a separate selectable control, and perform all vacuum/OBC1 comparisons
  locally without changing `jflows` or `jflows_md`.

### 2026-07-14T14:01:58-04:00 — Public jflows_md full-pool update verified, pushed, and frozen

- Audited the five pre-existing modified files and confirmed that they contain
  only the authorized `pool_size=0` full-pool controller path, the required
  per-attempt identity reset, and their focused documentation/tests. No
  potential, force-field, bundle, or other package file was modified.
- Passed the focused checkpoint/controller regression and the complete live
  `jflows_md` smoke suite on `cuda:0`. Coverage included full-pool KLXX with a
  fresh-source QT population, two-level identity-reset behavior, every current
  bundle/API/artifact check, OpenMM/JAX energy-force parity, and the float32
  glycerol compile path (`2.46 s` energy/gradient and `5.05 s` MALA).
- Committed and pushed exactly those five files at
  `a6e1948148d4b7628d2a329bd339e8be479f80fc`; local `main`, `origin/main`, and
  the clean worktree agree. Treat this as the frozen public `jflows_md` head.
  All subsequent c50/OBC1 work belongs only to the local vendored PyTorch
  `Molecular_BG_zflows/zflows_md` control.

### 2026-07-14T18:58:52-04:00 — jflows_md e/r reconstruction verified; molecular workspaces renamed

- Recorded that the user explicitly superseded the earlier local-only package
  restriction and authorized a focused `jflows_md` reconstruction. Generic
  `jflows` stayed frozen and clean. No production molecular experiment was
  launched.
- Replaced the retired cap-only public regularization interface with the
  descriptive e/r surrogate parameters `energy_threshold_kj_mol` and
  `pair_distance_floor_nm`. The raw physical energy remains available and
  unchanged; the optional distance floor is confined to ordinary nonbonded
  pairs/exceptions, while stable exact-collision and OBC1 small-distance logic
  prevents float32 NaNs and accidental gradients through a clamped pair.
- Added explicit initialization control. Direct molecular trainers preserve
  the supplied flow by default; molecular Boltzmann controllers default to an
  exact-identity start for every stage/retry. Schema-2 attempt metadata records
  the choice on successful, rejected, and incomplete paths, while
  full-validation ESS remains the sole stage acceptance quantity.
- Added and passed focused regularization/initialization/checkpoint tests and
  the complete isolated GPU smoke suite. Two independent final reviews report
  PASS. All three stored OpenMM energy/force parity checks remain within their
  float32 tolerances, and an isolated 0.3.0 wheel builds successfully. Published
  commit `9e1e42456f9cd9e8df488207d16a7708059e292a` to `origin/main` and created
  annotated tag `v0.3.0`; the local and remote branch/tag targets agree.
- Recorded the user-owned workspace moves:
  `Molecular_BG_jflows_v1_nor/`, `Molecular_BG_zflows_v1_original/`,
  `Molecular_BG_zflows_v2_snapshot/`, and
  `Molecular_BG_zflows_v3_reverify/`. The future regularized JAX campaign is
  reserved for `Molecular_BG_jflows_v2_r/`; that folder was not created and
  waits for the user's later instruction.
- Repaired the renamed reverified control's custom runner to honor pytest
  fixtures and parameterization. Its complete 31-test suite passes; rename-safe
  ignore rules keep generated candidate bundles and diagnostic runs out of Git.

### 2026-07-17T11:30:38-04:00 — Data paths migrated and methane e/r caller prepared

- Replaced active legacy-mount commands and provenance lookups with the live
  `/data/projects` roots across project documentation, current drivers, ignored
  operational configs, and the local `jflows` skill. Public `jflows` and
  `jflows_md` package repositories remain unmodified; historical captured
  stdout retains its original paths. The updated `jflows` skill passes its
  structural validator.
- Created `Molecular_BG/methane_9d/` without touching the successful
  `Molecular_BG_jflows_v1_nor/` baseline. Its six physical bundle files are
  byte-identical to the methane c50 source bundle, including matching SHA-256
  hashes.
- Added a standalone latest-interface full-size KLXX driver with primary
  potential constants `e=50.0` kJ/mol and `r=0.1` nm. It is designed to save
  every pre-update batch ESS by level, retry attempt, and optimizer step, plus
  the full-validation stage ESS records. Both Python files pass AST parsing and
  static live-signature checks. Per instruction, no training or other runtime
  execution was launched, and no `artifacts/` or `results/` directory exists.

### 2026-07-17T11:55:54-04:00 — Methane and ethane e/r KLXX runs completed

- Updated the main molecular energy threshold from the initial prepared value
  to the user-specified `e=100.0 kJ/mol`, retained `r=0.1 nm`, and completed the
  full methane 9D KLXX run. Its four accepted levels at
  `t=0.21/0.525/0.9975/1.0` have selected full-200000-particle ESS
  `0.978385/0.994428/0.998197/0.999992`; the first three select trained flows
  and the final endpoint selects identity.
- Created `Molecular_BG/ethane_18d/` from the same driver and an exact
  byte-for-byte copy of the legacy ethane six-file bundle. A machine comparison
  verified that all methane settings are unchanged except molecule identity,
  formula, dimension, bundle, and the requested smaller `t_safe`. The first
  `t_safe=0.25` pass completed, then its generated outputs were removed and
  overwritten as instructed by the final `t_safe=0.20` run.
- The final ethane run completed on `cuda:0` in `289.673 s`. Its accepted levels
  at `t=0.20/0.50/0.95/1.0` select trained flows with full-validation ESS
  `0.826795/0.965812/0.960052/0.998610`. All 2,000 per-step ESS values are
  finite and lie in `(0,1]`; 1,991 updates were applied and nine finite-safe
  updates were rejected. The consolidated CSV/NPZ histories exactly match all
  four monitor sidecars, all 200,000 x 18 final float32 particles are finite,
  and the schema-2 flow manifest is complete. Public `jflows` and `jflows_md`
  repositories remain clean and unmodified.

### 2026-07-17T16:59:19-04:00 — Full e/r alkane series completed and old workspaces archived

- Completed the current KLXX alkane sequence through propane 27D, n-butane
  36D, and n-pentane 45D. All five methane-through-pentane runs reached `t=1`;
  their endpoint full-validation ESS values are
  `0.999992/0.998610/0.996787/0.993094/0.829475`. The saved summaries contain
  19,500 per-step ESS rows in total, and no molecular training process remains.
- Recorded the working conclusion that the main potential regularization pair
  `(e, r)` is the crucial training-stability control, while optimizer batch
  size and validation population size are not the primary factors. The
  high-dimensional successful settings progressed from `e=50, r=0.1` at 27D
  to `e=50, r=0.15` at 36D and `e=50, r=0.2` at 45D.
- Set the next molecular direction: adapt and audit the per-stage e/r
  sharpening/reweighting technique from `../zflows_md`, retaining a separate
  sharpening ESS and the existing per-step/full-validation ESS records. No
  sharpening code was changed or launched in this checkpoint.
- Preserved the user's filesystem organization: the four old molecular test
  trees were moved below ignored `.archive/`, and `Paper_Arxiv/` was renamed to
  `Paper/`. The ignored archive and current molecular artifacts require the
  ext4 mirror for recovery; Git intentionally records only the old-path
  deletions and current tracked deliverables.
