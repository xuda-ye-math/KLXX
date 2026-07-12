# Project status

Last updated: 2026-07-11T22:09:55-04:00 (America/New_York)

## Current state

- Repository: `/mnt/projects/X-regularization`, branch `main`, base HEAD before
  this debugging checkpoint
  `9b7553cb627f5b4f88f0c5f2545c685555842ceb` (`Extract molecular packages
  and add glycerol driver`), tracking `origin/main` at 0 ahead / 0 behind.
  The reviewed worktree has 5 modified tracked paths, 46 tracked deletions,
  and 3 untracked v2 bundle directories. Every change is under
  `Molecular_BG/` or in this status record: the v1 bundle trees are being
  replaced by v2 quotient-measure targets, the glycerol driver is upgraded to
  artifact schema 2, and no unrelated user work is mixed into the checkpoint.
- **Public/private boundary:** `/mnt/projects/jflows` and
  `/mnt/projects/jflows_md` are the public package repositories. Their public
  GitHub interface uses ordinary editable installation and ordinary Python
  imports, with no workstation-specific `/mnt/projects` command in README,
  example, smoke, or bundle-builder instructions. `X-regularization` remains
  the private/local experiment tree: commands under `Codes/` and
  `Molecular_BG/` deliberately use `conda activate jflows` plus absolute
  `PYTHONPATH=/mnt/projects/jflows` or the two-root molecular path. This is now
  the sole project `status.md`; the retired package-root status file was
  removed from `jflows`, and `jflows_md` has none.
- Public `jflows` is clean and pushed at
  `6910c3cedfd8fc74314e74a1b475c3caec0861d9`. Its behavior remains the one used
  by the existing `Codes/` runs, so those results do not require a rerun.
  Public `jflows_md` is clean and pushed at
  `deac775d09cbb8fa686c59e3c3696c769c999e49` (`Correct molecular quotient
  targets and persistence`), containing the reviewed scientific/debugging
  checkpoint described below.
- The Conda `jflows` environment is dependency-only for local work. Distribution
  metadata and imports for both local packages are absent without
  `PYTHONPATH`; explicit roots resolve imports to
  `/mnt/projects/jflows/jflows` and `/mnt/projects/jflows_md/jflows_md`.
- **Done — corrected standalone `jflows_md` core:** schema-2 coordinates now
  use the standard Cartesian configurational measure after quotienting global
  translation and rotation. The missing nonconstant anchor factor in the old
  gauge-slice Jacobian is restored; legacy schema-1 bundles remain explicitly
  readable but all built-in targets are v2. The pure-JAX Amber/OBC1 potential,
  L-ADP half-chart, mixed NSF, torus-correct MALA, potential-space SMC,
  score-free target-rejuvenated AIS, molecular training, and public facade
  remain self-contained in `/mnt/projects/jflows_md`, with low-level code under
  `jflows_md/core/`.
- `Molecular_BG/` now contains only local experiment inputs and outputs:
  `bundles/`, `glycerol_36d/`, `reference/`, and `JFLOWS_MD_PLAN.md`. It no
  longer carries copied `jflows`, `jflows_md`, smoke, or package documentation
  trees. The standalone public `jflows_md` repository retains its own matching
  outer `bundles/` tree; bundle data is not hidden inside the Python package.
- Three immutable v2 targets remain complete: FAB L-ADP uses Amber ff96/OBC1
  on `R^42 × T^18` (60D); glycerol uses GAFF2/AM1-BCC/OBC1 on
  `R^25 × T^11` (36D); explicitly neutral diethanolamine uses
  GAFF2/AM1-BCC/OBC1 on `R^33 × T^15` (48D). Their physical and provenance
  payloads are byte-identical to the retired v1 sources; the only coordinate
  change is schema/chart/measure metadata plus a hash-bound upgrade record.
  Built-in names are pinned to exact manifest hashes, and the builder rejects
  changed seeds, AmberTools data/version, or candidate output unless a new
  bundle version is created. The public and local `Molecular_BG/bundles/`
  trees are checksum-synchronized.
- **Verification passed:** with `conda activate jflows` and explicit
  `PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md`, the complete public
  smoke suite passed. This includes all three OpenMM energy/force parity tests
  (maximum energy discrepancy `5.04e-8 kJ/mol`, maximum force RMSE
  `4.48e-8 kJ/mol/nm`), independent quotient-Jacobian checks, ADP chirality,
  float32 execution, Mixed_NSF, MALA, SMC, score-free AIS, one tiny BG training
  stage, chunking, artifact reconstruction, strict bundle closure, manifest
  immutability, and an exact ADP rebuild. Three independent final reviewers
  returned PASS with no high- or medium-severity issue. No production molecule
  training was launched.

## Pending

- **Decision needed — environment split:** determine whether bundle preparation
  should use a dedicated Conda OpenMM/ParmEd/AmberTools scientific environment
  while training uses a pure JAX/math environment with no OpenMM runtime
  dependency and loads `Molecular_Potential` directly from frozen bundles.
  Preserve this as a later packaging/runtime-boundary question; no decision was
  requested in this update.
- **Pending — next authorized molecular run:** launch the local
  `Molecular_BG/glycerol_36d` Boltzmann-generator training only after explicit
  user instruction. No production molecule training has been launched.
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
