# Molecule_BG: X-regularized Boltzmann generator on a real molecule

The final test of the suite. Goal: train an X-regularized Boltzmann generator on
a real molecular force field in whitened internal coordinates, then apply the
coarse-training separation trick (Section 3.5 of the paper) by truncating the
stiff C--H degrees of freedom, and compare against the relevant baselines.

Conventions follow `STYLE.md` (parameter names, file layout, Algorithm-4
framework, failure fall-backs). The flow library is `zflows`; energies come from
a real force field parsed with OpenMM / ParmEd.

Two phases: (A) full internal-coordinate BG, to establish that KLXX samples the
conformers; (B) coarse-fine with C--H truncation, the new contribution.

---

## Stage 1 -- `PDB_Potential`: a zflows Potential from a PDB

A `zflows.potential.Potential` subclass that, given a PDB and a force field,
evaluates `U(xi)` and its gradient on a GPU batch of **whitened internal
coordinates** `xi`.

1. **Parameters and reference energies (OpenMM / ParmEd).** Load the PDB, assign
   a force field (e.g. Amber `ff14SB` for a peptide, or GAFF + AM1-BCC for a
   small molecule), and extract the bonded and non-bonded parameters with ParmEd
   / `openmm.app`: bond `(k, r0)`, angle `(k, theta0)`, torsion Fourier series,
   Lennard-Jones `(sigma, epsilon)`, partial charges, and the exclusion / 1--4
   scaling list. Keep an OpenMM `Context` for cross-checking energies.
2. **Differentiable batched force field (torch).** Re-implement the parsed
   force field as a torch module `ff_energy(x_cartesian) -> [N]`: bonded terms
   are cheap; non-bonded is an `O(M^2)` pair sum with the exclusion mask and 1--4
   scaling (fine for a small molecule, `M` atoms). This gives `U_cart` and
   `grad U_cart` batched on the GPU. (Alternative: `openmm-torch` `TorchForce`;
   the pure-torch route is preferred for transparent control of exclusions and
   for the C--H freezing of Stage 4.)
3. **Internal coordinates (BAT) with the exact Jacobian.** A differentiable
   bond--angle--torsion (z-matrix) map `internal <-> cartesian`. The target
   density in internal coordinates carries the BAT volume factor, so
   `U_internal(z) = beta * U_cart(x(z)) - log|det J_{x<-z}(z)|`, with the
   standard BAT Jacobian (product of `r^2 sin(theta)` factors). Global
   translation/rotation are removed (6 DOF) by fixing the root frame.
4. **Whitening + priors.** From a short OpenMM MD trajectory, take the mean and
   std of each internal coordinate; whiten `xi = (z - mu_z) / sigma_z` so each
   coordinate is `O(1)`. Priors: **torsions uniform** on `[-pi, pi]` (use an
   `NCSF` periodic flow), **bonds and angles Gaussian** `N(0, 1)` after
   whitening. The source `mu_0` is the product of these. `PDB_Potential.forward`
   returns `0.5|xi_bond,angle|^2 + (uniform const) + U_internal(z(xi))` in
   whitened coordinates.

Files: `potential.py` (`PDB_Potential`, BAT transform, torch force field,
`build()`), `parameters.py` (canonical block + molecule-specific tail: PDB path,
force-field name, temperature, whitening stats path).

---

## Stage 2 -- validate the potential (gate before any training)

- **Energy cross-check.** `PDB_Potential` energy must match the OpenMM `Context`
  energy on the same configuration to numerical tolerance (the single most
  important correctness test; a wrong exclusion list or BAT Jacobian shows up
  here).
- **Gradient check.** Finite-difference vs autograd `grad U` on a few configs.
- **Referee.** A long OpenMM run (or replica exchange / metadynamics on the slow
  torsions) gives the ground-truth conformer populations -- the analog of the
  PT-MALA / transfer-matrix referee in the other benchmarks.
- **Target molecule, smallest first.** Start with **alanine dipeptide** (the
  canonical BG benchmark: backbone torsions phi/psi, ~22 atoms), where the
  free-energy surface and conformer basins are textbook; then a slightly larger
  peptide once the pipeline is solid.

---

## Stage 3 -- full internal-coordinate BG with KLXX

Train the adaptive-temperature generator (Algorithm 4) on the full internal
coordinates, source = whitened prior, target = `U_internal`, loss = the
X-regularized stage loss `KL + X_mu + X_{(hat_mu+bar_nu)/2}` at balanced
hyperparameters.

- **Flow.** `NCSF` on the periodic torsion block and `NSF` on the bond/angle
  block (a mixed/product flow), per `STYLE.md`.
- **Where the X terms bite.** Multimodality lives in the torsions (metastable
  conformers); quench-and-temper discovers the conformers, and `X_hat_mu` /
  the mixture insure against collapsing onto a subset -- the same role as in the
  2D, sensor, and phi^4 tests.
- **Metrics.** Per-stage validation ESS along the ladder; conformer coverage and
  populations vs the MD referee (TV on the basin weights, as for the Poisson
  wells). Compare bare forward KL vs KLXX: the prediction is KLXX completes the
  ladder and matches conformer populations where bare KL collapses or misweights.
- **Fall-backs (STYLE.md).** Tight grad clip + halved LR if the stiff bond/angle
  curvature spikes the loss; chunk bulk passes; whitening already removes the
  worst scale separation.

---

## Stage 4 -- coarse training: truncate the C--H degrees of freedom

Apply Section 3.5. Split `x = (x_L, x_H)`: `x_H` = the **C--H bond lengths and
H-centered angles** (the stiffest, highest-frequency, most conformation-decoupled
DOFs); `x_L` = everything else (heavy-atom bonds/angles + all torsions).

- **Definition of `U_L`.** `U_L(x_L)` is the molecule's potential with the C--H
  bonds and angles **fixed at their whitened means** (their reference geometry).
  `U_H(x_H)` is the whitened C--H Gaussian, `mu_H` sampled directly. The
  separation `U ~ U_L + U_H` holds because C--H is nearly harmonic and barely
  couples to conformation (validated as benign by the universal practice of
  constraining bonds-to-hydrogen in MD).
- **Construction (Section 3.5).** Train the BG only on `x_L` (dimension reduced
  by ~the hydrogen count -- the cheap part), draw `x_H ~ mu_H` fresh at
  evaluation, and reweight by `w = mu / (mu_L mu_H)`, log-weight
  `z_L + (U_L + U_H - U)`. The reweight stays exact for any proposal; the C--H
  stiffness makes it efficient. (Unlike SHAKE, which removes the DOF and needs a
  Fixman correction, this keeps the DOF and corrects by reweighting, so it is
  exact.)
- **Truncation-ceiling diagnostic FIRST** (the phi^4 lesson). Draw `x_L` from the
  MD referee, draw `x_H` from `mu_H`, and measure the full-force-field reweight
  ESS. Gate the trick on a high ceiling; the danger is non-bonded sterics forcing
  the C--H conditional away from its prior.
- **Freezing ladder.** C--H first (stiffest, safest, biggest dimension cut); then
  optionally heavy-atom bonds; stop where the ceiling ESS starts to drop
  (heavy-atom angles, which do shift with conformation, likely stay in the flow).
  Methyl-rotor hydrogens: freeze their C--H bonds and intra-methyl angles, keep
  the methyl torsion in the flow.
- **Payoff.** Same conformer accuracy as Stage 3 at training cost scaling with
  the soft (torsion + heavy-atom) dimension only.

---

## Stage 5 -- comparison with related methods

On the same molecule and force field, at matched compute:

- **Standard internal-coordinate BG** (bare forward KL, no X) -- the within-method
  baseline; expect conformer collapse / misweighting where KLXX holds.
- **FAB** (flow AIS bootstrap) -- contrast per Remark `rem: fab`: FAB needs flow
  density gradients in every AIS transition and a replay buffer; KLXX is
  score-free in the flow and buffer-free.
- **iDEM** (iterated denoising energy matching) and other energy-based samplers
  as available.
- **MD / MCMC** at equal wall-clock -- the practitioner's reference.

Report: conformer populations vs the MD referee (TV), per-stage / final ESS,
coverage, and wall-clock; for Stage 4 additionally the training-cost reduction at
fixed accuracy. The two claims to demonstrate: KLXX gives correct, reproducible
conformer weights where the non-oracle losses fail; the coarse-fine truncation
delivers the same weights at lower training cost.

---

## Risks and mitigations

- **Batched differentiable energy is the engineering crux.** The pure-torch force
  field must reproduce OpenMM to tolerance (exclusions, 1--4 scaling, PME vs
  cutoff for a small gas-phase molecule -> plain Coulomb). Mitigate with the
  Stage-2 energy cross-check as a hard gate.
- **BAT Jacobian and the `r^2 sin(theta)` singularities.** Linear/near-linear
  angles and zero bond lengths blow up the Jacobian; choose the z-matrix tree to
  avoid near-linear reference angles, and clamp in the diagnostic.
- **Coupling defeats the truncation** (the phi^4 failure mode). The ceiling
  diagnostic gates Stage 4; if C--H alone already collapses the ESS, the molecule
  is too coupled and the trick is reported as the measured boundary, not forced.
- **Referee cost.** Slow torsions may need enhanced sampling (REMD / metadynamics)
  for trustworthy populations; budget for it before claiming TV numbers.
