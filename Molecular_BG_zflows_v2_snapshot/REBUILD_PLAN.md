# Rebuild plan (v3) — ADP Boltzmann generator (klxx: annealed-SMC/AIS + X + X), FAB-style rejection

> v3 supersedes v2 after an independent code-level review refuted v2's central premise. Corrections
> are marked **[CORRECTED v2→v3]**. See the "How klxx actually works" section — read it before Phase 0.

## Goal
A clean alanine-dipeptide Ramachandran from the `klxx` Boltzmann generator (X-regularized annealed
SMC/AIS with the X_μ + X_(μ̂+ν̄)/2 terms), whose flat φ<0 regions match the validated MD reference
(`ramachandran_final.png` / FAB Fig. 19). All work inside `Molecular_BG_2/` with the local editable
`Molecular_BG_2/zflows_md/`.

## How klxx ACTUALLY works [CORRECTED v2→v3 — this is the crux]
`klxx` is **energy-based annealed SMC/AIS, NOT maximum likelihood on MD frames.** Verified in code:
- The training/validation batch is drawn from the **source prior** `u0` (whitened-Gaussian
  bonds/angles + uniform torsions), not from data: `boltzmann.py:329  Y = u0.samples(n_valid)`.
- The batch weight is purely energy-based: `boltzmann.py:222  logw = u_prev.eval(xb) - u_next.eval(y) + ladj`.
- The "forward-KL" term is the **annealed bridge log-ratio** μ_{k-1}→μ_k over the AIS surrogate
  batch (`fused_stage_loss`/`fused_kl_loss`, `boltzmann.py:64–79`), **not** E_data[−log p].
- The genuine data-driven MLE `forward_KL_F` (`loss.py:34`) is **never called**. `md_frames` are
  consumed ONLY for whitening stats `mu_b/sig_b/mu_a/sig_a` (`boltzmann.py:703–709`); the driver's
  2000-frame `short_md` already supplies them (`train.py:138`).

**Consequences (kills two v2 assumptions):**
1. There is **no data anchor** — Ramachandran quality is driven by **(a) OBC energy correctness** and
   **(b) SMC/Langevin mixing on that energy**, nothing else. The flat-region match is *plausible*
   (the φ<0 basins C5/C7eq/αR have low inter-basin barriers, so annealed SMC reaches them) but **not
   guaranteed by construction**. The MD reference is the **validation target to compare against**,
   not training data.
2. Saving full-coordinate `X_ref (N,22,3)` frames "for forward-KL MLE" is **dropped** — no code path
   consumes them as a likelihood target.

## Core design change (corrected placement)
Replace **soft-core** (`r_floor` distance clamp + `softcap_energy`) with **FAB-style rejection** —
but placed in the **sampling machinery**, not only the loss, because the NaN/Inf is generated and
consumed *inside* the SMC/AIS loop, upstream of the loss. Sharpening is dropped (not the contribution).

---

## Phase 0 — Physics fixes (amber96, L-alanine, OBC) [CORRECTED: no X_ref frames]
Fix the three diagnosed bugs (see `EXPERIMENT_LOG.md`):
1. **Force field:** amber96, not the openff-2.1.0 small-molecule prmtop.
2. **Chirality:** L-alanine — `short_md` must start from the L structure (`position_min_energy.pt`),
   not the D PDB, so the whitening stats (and any comparison) are L.
3. **Environment:** OBC implicit. Add `GBSAOBCForce` at **BOTH** OpenMM build sites, not one:
   `build_system` (`forcefield.py:55–65`, currently `implicitSolvent=None`) **and** the driver's
   whitening `short_md` (`train.py:56`, currently vacuum `NoCutoff`) — needed for the Phase-1 gate
   and the Option-B fallback to be meaningful. (Whitening from vacuum MD would be acceptable since
   bond/angle stats are solvent-insensitive, but keep them consistent.)
- **DROPPED [v2]:** saving full-coordinate reference frames for MLE — no consumer exists.

## Phase 1 — OBC energy backend + HARD validation gate
- **Option A (primary):** add an OBC2-GBSA term to the torch `Amber_Force_Field` (Born-radii + GB
  pair energy + ACE surface; autograd → forces free), params from `GBSAOBCForce` (ε_solvent=78.3,
  ε_solute=1.0, surfaceAreaEnergy=2.25936 kJ/mol/nm², per-particle charge/radius/scale; offset
  0.009 nm; OBC2 α,β,γ=1.0,0.8,4.85).
- **HARD GATE — `validate_obc.py` (do not train until it passes):** per-frame |ΔE_total(torch−OpenMM)|
  < 1e-2 kJ/mol on a batch of physical frames, and E_GB ≈ −50.36 kJ/mol at the min-energy config.
- **Option B (fallback, exact):** OpenMM-in-the-loop energy autograd Function (boltzgen
  `OpenMMEnergyInterface` style); slower, but zero reimplementation risk; also use as a cross-check.

## Phase 2 — Sample-wise rejection, placed in the SAMPLING machinery [CORRECTED — was loss-only]
Order matters: **add finiteness handling to the sampler FIRST, then remove `r_floor`** — otherwise
the stack crashes on the first clash (`softcap_energy` does NOT sanitize NaN: `NaN>e0→False→returns
NaN`, `forcefield.py:43–49`; removing the `clamp_min(self.r_floor)` at `forcefield.py:212` restores
`1/r` → float32 `inf−inf = NaN` at `forcefield.py:213–216`).
1. **Guard every consumer of energy weights (the real crash points):**
   - `resample` (`utils.py:359–360`): mask/zero non-finite weights **before** `torch.multinomial`
     (a NaN weight currently raises "invalid multinomial distribution").
   - `langevin`/ULA (`utils.py:603+`): drop/clip non-finite drift so a clash grad can't poison coords.
   - `sequential_monte_carlo`, `lbfgs`, `compute_ESS_log` (`utils.py:130–132`): treat non-finite
     energy as reject (weight→0 / exclude), never NaN-propagate.
   - sharpening resample path (`boltzmann.py:502–514`) — same guard.
2. **Then remove `r_floor`** (`forcefield.py:212`, keep a `1e-12` div-by-zero guard only) and disable
   `softcap_energy` in the potentials (optionally adopt FAB's exact `regularize_energy` as the only
   regularizer; units: FAB cut on E/kBT vs yours on kJ/mol — convert via /(R·T)).
3. **Loss-layer mask + per-batch NaN/Inf step-skip + grad clip** = backstop, not primary guard.
4. **Log the rejection rate** each step (fraction of non-finite energies dropped in the sampler) to
   the status log — the observable that replaces the soft-core; must stay low/bounded.

## Phase 3 — Train the BG (klxx: annealed SMC/AIS + X_μ + X_(μ̂+ν̄)/2)
- `klxx` on the corrected implicit/L-alanine/amber96 target, **soft-core off / sampler-rejection on /
  sharpening off**. Quality = validated OBC energy + SMC mixing (no data anchor — see crux section).
- Per-molecule read-only `config.json`. Observability (hard rule): flushed timestamped status log —
  START (sizes/device), per-iteration (step/total, loss, **rejection rate**, ESS), DONE (result+path);
  no tqdm. Executor launches the GPU job via tracked background; verify training via `nvidia-smi`.

## Phase 4 — Reference comparison + QUANTITATIVE metric
- Plot **BG vs MD reference** Ramachandran, FAB Fig. 19 style (log-viridis, φ,ψ∈[−π,π], 100×100,
  light σ=1 smoothing), as NEW files (never overwrite/delete).
- **A number, not just a picture:** KL (or L1/L2) between BG and reference Ramachandran histograms,
  **reweighted vs unweighted**, and/or IS **ESS** vs the OBC target — reported **X-on vs X-off** to
  show X_μ / X_(μ̂+ν̄)/2 actually help (the plot alone can't, and there's no MLE that trivially matches).

## Phase 5 — Independent verification gates
- **Flat-region match:** independent fresh-context agent scores BG vs FAB Fig. 19 (score/10 +
  FLAT_REGIONS_MATCH / PARTIAL / NO), reading PNGs directly (as in `INDEPENDENT_REVIEW.md`).
- **Rejection soundness:** a short soft-core-off run must NOT crash `resample`/`langevin` and must
  stay finite (logged rejection rate bounded, loss not diverging) before the full run.
- **OBC correctness:** cite the passing Phase-1 `validate_obc.py` numbers.

## Caveats — what the plot will and will NOT show [CORRECTED]
- **No MLE guarantee.** Flat-region match rests on annealed-SMC mixing over the correct OBC energy,
  not on fitting data. If SMC under-mixes, basins can be mis-weighted — watch the reference metric.
- **αL mode loss inherited.** Annealed SMC at 300 K crosses the αL barrier as rarely as the MD walker
  did; φ>0 will be under-populated (accepted).
- **The plot alone doesn't prove the X contribution** — needs the Phase-4 reweighted-vs-unweighted metric.

## Risks / open decisions
- **OBC-torch correctness = make-or-break** — gated by `validate_obc.py`; Option-B fallback exact.
- **Sampler finiteness must land before `r_floor` removal** — else `torch.multinomial`/ULA crash.
- **SMC mixing, not data, sets basin populations** — the real quality risk; monitor the metric/ESS.
- **Rejection-rate stalls** — high early rate wastes SMC weight; monitor; consider keeping a very high
  (FAB-scale, kT) `regularize_energy` cap so clashes are finite-but-negligible rather than NaN.
- **Units** — e_cap/energy_cut kJ/mol (yours) vs kT (FAB); convert consistently.

## Files this touches (local copies only, in Molecular_BG_2/)
- `zflows_md/forcefield.py` — add OBC term; disable `r_floor` (l.212); implicit `build_system` (l.55–65).
- `zflows_md/utils.py` — finiteness guards in `resample` (l.359), `langevin` (l.603+),
  `sequential_monte_carlo`, `lbfgs`, `compute_ESS_log` (l.130) — the primary rejection layer.
- `validate_obc.py` (new) — Phase-1 hard gate (torch vs OpenMM energy match).
- `zflows_md/potential.py` — neutralize `softcap_energy`; rejection-friendly energy.
- `zflows_md/boltzmann.py` — loss-layer backstop mask + step-skip + grad clip + rejection-rate log;
  guard sharpening resample (l.502–514); OBC-aware `build`.
- `train.py`/`config.json` — amber96 + implicit + L-start `short_md`; BG-vs-ref plot + KL/ESS metric.
- **DROPPED:** any "save X_ref (N,22,3) frames" step (v2 artifact; no consumer).
