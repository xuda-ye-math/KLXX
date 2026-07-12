# Project status

Last updated: 2026-07-12T06:56:00-04:00 (America/New_York)

## Current state

- Repository: `/mnt/projects/X-regularization`, branch `main`, HEAD
  `c7c47b1c18b73e138621c745fbf8c22f212f6923` (`Adopt pip workflow and canonical
  molecular bundles`), tracking `origin/main` at 0 ahead / 0 behind. The
  worktree has three modified tracked files
  (`Molecular_BG/glycerol_36d/{parameters.py,train.py}` and `status.md`) plus
  the untracked diagnostic directory
  `Molecular_BG/glycerol_36d/debug_loss_ess/`. Its ignored NPZ/EQX/log payload
  is recovery-critical and occupies about 290 MB.
- **Public/private boundary:** `/mnt/projects/jflows` and
  `/mnt/projects/jflows_md` remain the public package repositories. Their
  READMEs present a conventional pip-created `.venv`, `source` activation, and
  editable `pip install -e .` interface without workstation-specific paths.
  `X-regularization` remains the private experiment tree: every active command
  activates `/home/xuda/.envs/jflows`, then uses ordinary `python` plus explicit
  live-source `PYTHONPATH=/mnt/projects/jflows` or
  `/mnt/projects/jflows:/mnt/projects/jflows_md`. This is the sole project
  `status.md`; neither public package carries one.
- Public `jflows` is clean at pushed HEAD
  `616255f5d6212fcb0b03ccb3b08db0c339c9d0f2` (`Fix circular RQS seam
  selection`). The narrow change restores the learned circular-RQS derivative
  and log-Jacobian at the exact left seam while preserving all other knot
  selection, NSF boundary/tail behavior, public APIs, and Equinox
  serialization. Public `jflows_md` is clean at pushed HEAD
  `b8ed572ee8ad38e5452972eb3ca7255e0db3d739` (`Add molecular KLXX checkpoint
  selection`). It adds molecular KLXX, quench-and-temper support, accurate
  target-ratio monitoring, sparse flow snapshots and full-validation
  checkpoint selection, metadata, documentation, and regression coverage.
  Generic `jflows` behavior is unchanged, so existing `Codes/` numerical
  results require no rerun.
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
- **Verification passed:** `jflows/smoke/test_flow.py` and the complete live
  `jflows_md/smoke/run_all.py` suite passed on `cuda:0`; after strengthening
  the controller coverage, `smoke/test_boltzmann_checkpoints.py` passed again.
  Coverage includes all three OpenMM/JAX parity tests (maximum energy
  discrepancy `5.04e-8 kJ/mol`, maximum force RMSE `4.48e-8 kJ/mol/nm`),
  quotient Jacobians, ADP chirality, float32 execution, Mixed_NSF, MALA, SMC,
  score-free AIS, a tiny BG training stage, chunking, strict bundle closure,
  legacy artifact migration, wheel/external-data behavior, and the glycerol
  compile path (`2.46 s` energy+gradient, `5.14 s` one-step MALA in the latest
  smoke). Two independent final checkpoint/API reviewers returned PASS with no
  remaining high- or medium-severity issue; a separate gradient audit found no
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
- **Done — opt-in overtraining guard:** `jflows_md` now distinguishes the live
  target-pool ratio concentration from honest proposal ESS. A sparse
  `selection_steps` schedule compares exact identity (step -1), the accepted
  warm start (step 0), requested post-update flows, and the final flow on all
  validation particles before applying the unchanged `tau_ess` gate. The empty
  schedule preserves the original final-versus-identity behavior. Focused
  tests uniquely select a warm start and an intermediate checkpoint, rescue a
  nonfinite final flow, preserve multi-stage warm starts and trained-on-tie
  behavior, and enforce the 32-snapshot safety limit.

## Pending

- **Pending — `zflows_md` compilation engineering:** the archived PyTorch
  molecular implementation still has unresolved excessive compile latency and
  memory growth at realistic molecular sizes. Compare compilation boundaries,
  chunking, and wrapper granularity before any attempt to revive it; the
  successful `jflows_md` smoke compile does not resolve this separate issue.
- **Pending — new molecular targets:** every AmberTools-26 or otherwise changed
  small-molecule model must use a new descriptive bundle name and receive an
  explicit scientific/provenance review before promotion to the frozen registry.
- **Pending — next authorized molecular run:** launch the local
  `Molecular_BG/glycerol_36d` Boltzmann-generator training only after explicit
  user instruction. The completed smoke and interrupted diagnostic attempt are
  not production results; no GPU process is currently running.
- **Pending — ESS degeneration and identity selection:** the
  standard-compiler smoke proved that the complete 36D pipeline runs quickly
  enough, but identity won every accepted level and the production-size bare-KL
  diagnostic peaked early before overfitting its fixed 200000-particle pool.
  This remains the primary unresolved molecular-training problem. The next
  authorized experiment should retain sparse full-validation checkpoint
  selection, report independent holdout ESS, and compare KL+X or KLXX against
  bare KL before increasing step count. A run must select trained stages and
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
- Replaced the molecular trainer's misleading live `ESS` label by
  `target-ratio C`. Added opt-in sparse post-update flow snapshots and a
  full-validation stage selector that also tests identity and the accepted
  warm start, while preserving the original empty-schedule behavior and
  unchanged `tau_ess` gate. The private glycerol driver saves all checkpoint
  labels, steps, and ESS values.
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
