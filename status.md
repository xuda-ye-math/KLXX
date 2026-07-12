# Project status

Last updated: 2026-07-12T00:13:17-04:00 (America/New_York)

## Current state

- Repository: `/mnt/projects/X-regularization`, branch `main`, base HEAD
  `50cf110541b8dc81d9e1e0e96fddafabb92c4daa` (`Checkpoint corrected molecular
  quotient targets`), tracking `origin/main` at 0 ahead / 0 behind before this
  checkpoint is committed. The reviewed worktree has 36 logical changed files,
  including 49 path-level bundle renames, and no untracked paths or standalone
  deletions. Changes are limited to active launch documentation, the molecular
  plan/driver, the mirrored bundle tree, environment documentation, and this
  status record.
- **Public/private boundary:** `/mnt/projects/jflows` and
  `/mnt/projects/jflows_md` remain the public package repositories. Their
  READMEs present a conventional pip-created `.venv`, `source` activation, and
  editable `pip install -e .` interface without workstation-specific paths.
  `X-regularization` remains the private experiment tree: every active command
  activates `/home/xuda/.envs/jflows`, then uses ordinary `python` plus explicit
  live-source `PYTHONPATH=/mnt/projects/jflows` or
  `/mnt/projects/jflows:/mnt/projects/jflows_md`. This is the sole project
  `status.md`; neither public package carries one.
- Public `jflows` remains based on pushed HEAD
  `6910c3cedfd8fc74314e74a1b475c3caec0861d9`, with one reviewed README change.
  Public `jflows_md` remains based on pushed HEAD
  `deac775d09cbb8fa686c59e3c3696c769c999e49`, with 22 logical changed files
  including the same 49 bundle payload renames. Generic `jflows` behavior is
  unchanged, so existing `Codes/` numerical results require no rerun.
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
- **Verification passed:** `jflows/smoke/test_flow.py` and the final complete
  `jflows_md/smoke/run_all.py` suite passed from isolated copies on `cuda:0`.
  Coverage includes all three OpenMM/JAX parity tests (maximum energy
  discrepancy `5.04e-8 kJ/mol`, maximum force RMSE `4.48e-8 kJ/mol/nm`),
  quotient Jacobians, ADP chirality, float32 execution, Mixed_NSF, MALA, SMC,
  score-free AIS, a tiny BG training stage, chunking, strict bundle closure,
  legacy artifact migration, wheel/external-data behavior, and the glycerol
  compile path (`2.45 s` energy+gradient, `5.08 s` one-step MALA in the final
  smoke). Three independent final reviewers returned PASS with no remaining
  high- or medium-severity issue. No production molecule training was launched.

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
