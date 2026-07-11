# Project status

Last updated: 2026-07-10T23:19:48-04:00 (America/New_York)

## Current state

- Repository: `/mnt/projects/X-regularization`
- Branch: `main`
- Current HEAD: `8607f8677ef6f6e53d37d6f855f975d713339d70` — `Use project relation macros in manuscript`; branch `main` tracks `origin/main`.
- Worktree before this status update: six tracked paths changed (four modified and two deleted), with no staged or untracked paths. The changes are the unified manuscript source/PDF, preamble and README updates, and retirement of `Paper_Arxiv/supp.tex` and `Paper_Arxiv/supp.pdf`. Ignored research data, checkpoints, logs, and bytecode remain local.
- Canonical manuscript: `Paper_Arxiv/main.tex`; the compiled `Paper_Arxiv/main.pdf` is 39 pages and 7,895,229 bytes.
- Canonical numerical tests: `Codes/`, copied byte-for-byte from `/mnt/projects/jflows/Codes`; 110 files, approximately 2.4 GiB. Python drivers use public `jflows` APIs and contain no `zflows` imports. Equinox `*.eqx` checkpoints and `*.npz` arrays remain on disk but are ignored by Git.
- Section 5 of `Paper_Arxiv/main.tex` is synchronized with `Codes/**/results.md` and `Codes/style.md`. The former Fisher–Rao supplement is now Appendix A, “Fisher–Rao gradient flow with log-ratio variation,” in the same file; Section 2 and the organization paragraph direct readers to the appendix. The molecular section remains outside the numerical-results rewrite.
- The latest PDF build completed successfully with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`; it has no unresolved references or reported overfull/underfull boxes.
- Recovery snapshot: `/mnt/backup/X-regularization_100726`, verified at HEAD `55b4ce98324beca51f2862663628d0990260c79a`. It predates the later cleanup, `Codes/` copy, and Section 5 rewrite.
- Major tracked deletions are the user's cleanup of legacy numerical trees and the old `Paper/` tree: `Clock_Lattice/`, `HD_Product/`, `HD_Product_Ladder/`, `Phi4_Lattice_6/`, `Phi4_Lattice_8/`, `Poisson_Inverse/`, `Sensor_Array/`, and `Paper/` content.
- The canonical `Codes/`, non-ignored `Molecular_BG_2/` content, `adp.jpeg`, `status.md`, and new Section 5 figure directories are now tracked. Ignored Molecular checkpoint/data artifacts remain protected by the mirror backup.

## Pending

- **Pending:** Decide whether ignored `Codes/` arrays, Equinox checkpoints, and run logs need a tagged release or other external archival location beyond the mirror backup.
- **Pending:** Update Sections 1--4, conclusions, README, and environment/install instructions where they still describe the former `zflows` backend; the completed rewrite was deliberately limited to Section 5.
- **Pending:** Audit the detailed contents and long-term organization of `Molecular_BG_2/`; this checkpoint intentionally included its non-ignored files without a content review. Decide whether `adp.jpeg` and the tracked stdout files should remain in their current locations.
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
