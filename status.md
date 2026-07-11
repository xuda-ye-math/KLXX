# Project status

Last updated: 2026-07-10T22:53:06-04:00 (America/New_York)

## Current state

- Repository: `/mnt/projects/X-regularization`
- Branch: `main`
- HEAD: `55b4ce98324beca51f2862663628d0990260c79a` — `README: point paper references to Paper_Arxiv (main + supp)`
- Worktree: dirty. After adding this file, the expected summary is 3 modified tracked files, 130 deleted tracked files, and 11 untracked path groups; nothing is staged.
- Canonical manuscript: `Paper_Arxiv/main.tex`; the compiled `Paper_Arxiv/main.pdf` is 34 pages and 7,853,628 bytes.
- Canonical numerical tests: `Codes/`, copied byte-for-byte from `/mnt/projects/jflows/Codes`; 110 files, approximately 2.4 GiB. Python drivers use public `jflows` APIs and contain no `zflows` imports. Equinox `*.eqx` checkpoints and `*.npz` arrays remain on disk but are ignored by Git.
- Section 5 of `Paper_Arxiv/main.tex` is synchronized with `Codes/**/results.md` and `Codes/style.md`. Sections 1--4 and the molecular section were intentionally left outside that numerical-results rewrite.
- The latest PDF build completed successfully with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`; it has no unresolved references or reported overfull/underfull boxes.
- Recovery snapshot: `/mnt/backup/X-regularization_100726`, verified at HEAD `55b4ce98324beca51f2862663628d0990260c79a`. It predates the later cleanup, `Codes/` copy, and Section 5 rewrite.
- Major tracked deletions are the user's cleanup of legacy numerical trees and the old `Paper/` tree: `Clock_Lattice/`, `HD_Product/`, `HD_Product_Ladder/`, `Phi4_Lattice_6/`, `Phi4_Lattice_8/`, `Poisson_Inverse/`, `Sensor_Array/`, and `Paper/` content.
- Important untracked paths are `Codes/`, `Molecular_BG_2/`, `adp.jpeg`, `status.md`, and new Section 5 figure directories under `Paper_Arxiv/figures/`.

## Pending

- **Pending:** Review and stage the 130 legacy deletions together with the replacement `Codes/` tree and new `Paper_Arxiv` figures.
- **Pending:** Decide which non-ignored `Codes/` results, logs, CSV/Markdown files, and figures belong in Git versus external release storage.
- **Pending:** Update Sections 1--4, conclusions, README, and environment/install instructions where they still describe the former `zflows` backend; the completed rewrite was deliberately limited to Section 5.
- **Decision needed:** Decide whether `Molecular_BG_2/` and `adp.jpeg` are permanent project inputs/outputs and should be tracked, moved, or ignored.
- **Pending:** Perform a final full-paper consistency audit after the non-Section-5 backend text is updated.
- **Pending:** Commit and push only after the deletion set and all new untracked content have been reviewed.

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
