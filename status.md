# Project status

Last updated: 2026-07-11T12:24:51-04:00 (America/New_York)

## Current state

- Repository: `/mnt/projects/X-regularization`
- Branch: `main`
- Current HEAD: `ecc2371dc00a2fa5882767db1223dc1b91f97f1f` — `Polish manuscript claims and retire old 2D benchmarks`; branch `main` tracks `origin/main`.
- Worktree before this status update: no staged paths, 2 modified tracked paths (`.gitignore`, `status.md`), 52 tracked deletions, and 8 untracked top-level/path entries reported by Git. The molecular reorganization accounts for 51 tracked deletions under the former `Molecular_BG/` tree plus `adp.jpeg`: the complete original tree is preserved as `Molecular_BG_1/`, while the clean replacement is `Molecular_BG/`. Existing user deletions and unrelated changes were preserved. Ignored research datasets, checkpoints, logs, and bytecode remain local.
- Canonical manuscript: `Paper_Arxiv/main.tex`; the compiled `Paper_Arxiv/main.pdf` is 39 pages and 7,893,661 bytes.
- Canonical numerical tests: `Codes/`, copied byte-for-byte from `/mnt/projects/jflows/Codes`; 110 files, approximately 2.4 GiB. Python drivers use public `jflows` APIs and contain no `zflows` imports. Equinox `*.eqx` checkpoints and `*.npz` arrays remain on disk but are ignored by Git.
- Sections 1--4 have received a first consistency and evidence pass against `Codes/style.md` and the current results. The paper now distinguishes the intended small/noisy-batch robustness from the benchmark-specific large-batch clock advantage, names the balanced method as `KL + X_μ + X_(μ̂ + ν̄)/2`, uses stage/step terminology consistently, and defines the QT melt scale as `m_e`. Molecular results are excluded from the abstract and Sections 1--4 framing for now.
- The score-free training annealing is now described as a biased target surrogate rather than exact AIS/SMC because its rejuvenation kernel targets the final distribution instead of each intermediate geometric bridge. Outer ladder selection remains a separate classical SMC procedure.
- The latest PDF build completed successfully with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`; it has no unresolved references or reported overfull/underfull boxes.
- Recovery snapshot: `/mnt/backup/X-regularization_100726`, verified at HEAD `55b4ce98324beca51f2862663628d0990260c79a`. It predates the later cleanup, `Codes/` copy, and Section 5 rewrite.
- Major tracked deletions are the user's cleanup of legacy numerical trees and the old `Paper/` tree: `Clock_Lattice/`, `HD_Product/`, `HD_Product_Ladder/`, `Phi4_Lattice_6/`, `Phi4_Lattice_8/`, `Poisson_Inverse/`, `Sensor_Array/`, and `Paper/` content.
- The canonical molecular workspace is now `Molecular_BG/`, a clean OpenMM + local-jflows rebuild. Its completed reference benchmark consists of two independent 20 ns-per-slot CUDA REMD chains with 21 replicas (300--1300 K), Amber ff96 with `amber96_obc.xml` OBC2, and a fixed 2 ns burn-in. The pooled artifact contains 36,000 saved 300 K phi/psi pairs and can be replotted without OpenMM. Each schema-v2 chain file contains native context checkpoints, exchange RNG state, walker history, configuration hashes, and samples, and both chains were successfully extended from 10 to 20 ns with `--resume`.
- Quantitative OBC2 evidence is in `Molecular_BG/data/analysis/metrics.json`: 36-bin cross-chain JS divergence is `0.01229` bits; per-chain JS divergence to the authors' one-million-frame FAB data is `0.01104` and `0.01172` bits. Every walker visited all 21 slots, minimum per-walker round trips are 461 and 471, and exchange-acceptance ranges are `0.543--0.882` and `0.544--0.879`. This establishes a visually close, internally reliable OBC2 benchmark, not identity with the FAB Hamiltonian.
- `Molecular_BG/adp_fab.png` is currently the raw 100 x 100 Ramachandran histogram of exactly 1,000,000 ground-truth FAB REMD configurations, with logarithmic color limits `1e-4`--`1` and no KDE. It is the only retained ADP PNG after the requested figure cleanup.
- Independent source and numerical audits established that FAB actually uses OpenMMTools ff96 + OBC1 (`igb=2`, `mbondi2`, ACE, dielectric 1.0/78.5), 300 K, `NoCutoff`, and no constraints. A PDB-built `amber96.xml + implicit/obc1.xml` system matches the canonical OpenMMTools system on 100 FAB frames to maximum energy error `1.90e-4 kJ/mol` and force RMSE `1.21e-3 kJ/mol/nm`; the prior OBC2 system differs by energy standard deviation `0.884 kJ/mol` and force RMSE `25.6 kJ/mol/nm`.
- The approved next-phase design is `Molecular_BG/JFLOWS_MD_PLAN.md`. It specifies a new pure-JAX `jflows_md` package and three targets: exact FAB L-ADP (60D, ff96/OBC1), glycerol (36D, GAFF2/AM1-BCC/OBC1), and explicitly neutral diethanolamine (48D, GAFF2/AM1-BCC/OBC1). Each target uses an immutable benchmark bundle rather than inferring physics from coordinates. The shared design provides mixed Euclidean/circular spline flows, domain-correct MALA, potential-space SMC, forward KL + X training, finite-safe `e_clip`, and no sharpening or target deformation.
- Three independent second-round audits of the final plan--force-field/model provenance, chirality/coordinate support, and `jflows_md` architecture--all returned `PASS — no main issues`. The converged design freezes `CoordinateSpec` with `SystemSpec`, treats PDB/Amber entry points as audited bundle builders, preserves full parity support for glycerol and neutral diethanolamine, gives the exact pinned GAFF2/AM1-BCC/mbondi2/OBC1 construction contract, and separates optimizer-only `e_clip` from honest SMC/ESS.
- Chirality is now evidence-backed: the current topology PDB coordinates are D (`c=-0.00262 nm^3`), the saved L minimum is positive, and all 1,000,000 FAB configurations are L. The plan therefore requires regenerating a canonical L PDB and parameterizes the L chiral torsion through a smooth half-chart rather than copying FAB's transform-specific posthoc filter.
- `Molecular_BG_1/` is the preserved original PyTorch result tree formerly tracked at `Molecular_BG/`. Its reported molecular results are historical/paper-era artifacts and may be physically wrong: the ADP run used a vacuum target and a small-molecule/OpenFF-style topology rather than the intended Amber96/OBC L-alanine system; its table reports full-target ESS `0.000`. Do not use it as the active implementation or trusted ADP benchmark.
- `Molecular_BG_2/` is the paused PyTorch correction attempt and a useful read-only reference implementation. It contains a self-contained `zflows_md/` package parallel in role to `jflows`, including molecular coordinates, force fields, potentials, flows, losses, Boltzmann/SMC utilities, and plotting helpers. It correctly diagnosed force-field, chirality, and solvent errors; generated a 34 ns Amber96/OBC OpenMM reference whose flat negative-phi basins were independently scored 8/10 against FAB; and passed its Torch-vs-OpenMM OBC energy gate with maximum total error `1.366e-05 kJ/mol`. The later PyTorch ASMC result was not a successful replacement: its saved 200,000-sample ensemble has `frac(phi>0)=0.577`. Active PyTorch development stopped when the project switched fully to JAX/jflows, but `Molecular_BG_2/zflows_md/` should be consulted for molecular design and implementation details rather than dismissed as disposable output.

## Pending

- **Pending:** Decide whether ignored `Codes/` arrays, Equinox checkpoints, and run logs need a tagged release or other external archival location beyond the mirror backup.
- **Pending:** Continue the Sections 1--4 polish, then update the conclusion, README, and environment/install instructions where they still describe the former `zflows` backend. The molecular section and its conclusions are intentionally treated as not yet part of the current paper narrative.
- **Done (OBC2 comparison benchmark):** Two independent production chains reached 20 ns per temperature slot in restartable 10 ns segments, producing 36,000 post-burn-in 300 K samples and a quantitatively FAB-like distribution. Preserve it as the explicitly labeled `amber96_obc.xml` OBC2 sensitivity result; do not call it the exact FAB target.
- **Qualification:** The major OBC2 modes and basin weights agree with FAB at the reported histogram resolutions, but the Hamiltonians differ and the 106 positive-phi frames do not establish convergence of rare-region fine topology.
- **Pending (next major task):** Implement `Molecular_BG/JFLOWS_MD_PLAN.md` in a new `jflows_md` package. Begin with the canonical L PDB, an immutable ff96/OBC1 ADP benchmark bundle carrying both `SystemSpec` and `CoordinateSpec`, and pure-JAX energy/force validation gates; then add separately validated glycerol and neutral-diethanolamine GAFF2/AM1-BCC/OBC1 bundles and matched REMD references. Do not start BG training before the applicable gates pass. A coordinate-only public dataset is evaluation/scaling data, not a sufficient potential definition.
- **Pending:** Add compatible JAX/Equinox dependencies to the Python 3.11 `jflows` Conda environment or use the documented two-process SystemSpec build/runtime split. OpenMM reference construction should use CPU/Reference and JAX alone should own the GPU during training.
- **Pending:** Keep `Molecular_BG_1/` as the committed historical PyTorch archive and `Molecular_BG_2/zflows_md/` as the read-only repair reference parallel to jflows. Do not resume the PyTorch backend unless explicitly requested.
- **Ignored/inactive:** `Molecular_BG/data/runs/dense100k_seed20260713/remd.npz` is a stopped 0.1 ns checkpoint with 1,000 high-cadence samples from the aborted dense-run request. No OpenMM project process is running; this file is not a production result.
- **Pending:** Perform a final full-paper consistency audit after the non-Section-5 backend text is updated.

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
