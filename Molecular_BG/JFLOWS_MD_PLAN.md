# Plan: `jflows_md` three-molecule Boltzmann-generator benchmark

## Objective and fixed decisions

The first `jflows_md` suite contains three molecular Boltzmann-generator
targets: FAB alanine dipeptide (ADP, 60D), glycerol (36D), and neutral
diethanolamine (48D). ADP reproduces a published target; glycerol and
diethanolamine extend the same package to explicitly defined small-molecule
benchmarks. This phase designs the package and validation gates; it does not
implement or train the models yet.

Fixed decisions:

1. The ADP benchmark Hamiltonian is exactly FAB's 22-atom ACE--ALA--NME model:
   Amber ff96 with OBC1 GBSA (`igb=2`, `mbondi2` radii), ACE nonpolar term,
   dielectric 1.0/78.5, zero salt, `NoCutoff`, no constraints, and 300 K.
2. `amber96_obc.xml` is not used for the FAB target. It is OBC2 and defines a
   different Hamiltonian. It may remain a separately named sensitivity model.
3. The target is implemented as a pure-JAX potential. OpenMM is used to build
   the immutable parameter specification and as the independent reference.
4. The ADP state is an exact L-only internal-coordinate chart on
   `R^42 x T^18`, not an all-periodic 60-dimensional box.
5. There is no energy soft-cap, distance-floor anneal, post-stage sharpening,
   or soft chirality penalty. Annealing stops at the physical target `t=1`.
6. MALA is the rejuvenation kernel. `e_clip` is only a training-loss screen;
   it never changes MALA, SMC weights, final importance weights, or the target.
7. The million-frame FAB trajectory is evaluation data, not a likelihood
   training set. Short physical data may set coordinate scales only.
8. A target bundle freezes both the Cartesian Hamiltonian and the complete
   internal-coordinate support. Caller-supplied, unversioned coordinate
   transforms are not allowed in benchmark training.

## Three-target benchmark matrix

| Target | Atoms / dimension | Frozen Hamiltonian | Internal domain/support | Reference |
|---|---:|---|---|---|
| L-ADP | 22 / 60 | FAB ff96 + OBC1/ACE | `R^42 x T^18`, L by construction | Authors' intended FAB 1M REMD split |
| Glycerol | 14 / 36 | GAFF2 + AM1-BCC + OBC1/ACE | `R^25 x T^11`, full labeled parity support | New matched, mirror-paired REMD |
| Neutral diethanolamine | 18 / 48 | GAFF2 + AM1-BCC + OBC1/ACE | `R^33 x T^15`, both N-pyramid signs | New matched, mirror-paired REMD |

The archived glycerol and diethanolamine tests identify the molecules and
dimensions, but not a valid implicit-solvent Hamiltonian. Their new presets
must therefore be generated, audited, and versioned separately. The
diethanolamine target preserves the archived neutral `OCCNCCO` microstate for
continuity. A protonated microstate would be a distinct benchmark bundle with
its own parameters and reference data.

Glycerol's middle carbon has two identical `CH2OH` arms, and neutral
diethanolamine has two identical hydroxyethyl arms; neither molecule has a
configurational stereocenter to condition. They must not inherit ADP's
half-chart. Their references must demonstrate mixing between both relevant
signed-volume/pyramidal signs or combine independently equilibrated mirrored
starts with exactly symmetric weights.

## Why ff96/OBC1

FAB constructs `openmmtools.testsystems.AlanineDipeptideImplicit` with
`constraints=None`. The contemporaneous OpenMMTools implementation builds its
ff96 prmtop with `implicitSolvent=app.OBC1`. A direct audit on 100 FAB frames
found that

```python
ForceField("amber96.xml", "implicit/obc1.xml")
```

reproduces the canonical prmtop/OBC1 system to a maximum energy difference of
`1.90e-4 kJ/mol` and force RMSE `1.21e-3 kJ/mol/nm`. The prior
`amber96.xml + amber96_obc.xml` OBC2 model differs by a conformation-dependent
energy standard deviation of `0.884 kJ/mol` and force RMSE
`25.6 kJ/mol/nm`.

ff96/OBC1 is therefore reliable as a versioned machine-learning benchmark with
published reference samples. Zenodo labels the data only as ff96 with OBC GBSA
and does not ship its generation script, so OBC1 is established exactly for
FAB's target evaluator and is the authors' intended reference model, rather
than being independently encoded in the HDF5 metadata. It is not claimed to be
modern experimental molecular truth. A later modern model must be a separate
preset with its own MD reference, for example ff14SB + GBn2.

## Proposed package boundary

```text
jflows_md/
  system.py          # audited builders + immutable bundle/SystemSpec loading
  forcefield.py      # pure-JAX Amber bonded/nonbonded and OBC1/ACE energy
  coordinates.py     # bundle-defined BAT transform and exact Jacobians
  chirality.py       # target-specific signed-volume/support diagnostics
  domain.py          # R^p x T^q metadata and periodic operations
  potential.py       # Molecular_Potential facade and Cartesian reconstruction
  source.py          # Gaussian Euclidean x uniform torus source
  flow.py            # mixed Euclidean/circular spline coupling flow
  mcmc.py            # mixed-domain MALA
  smc.py             # thin molecular controller around generic jflows SMC
  validation.py      # OpenMM, geometry, sampler, and distribution gates
```

`jflows` remains the generic flow/training engine. Molecular physics,
coordinates, chirality, and diagnostics belong in `jflows_md`. Small generic
changes may be made upstream in `jflows`:

- accept an injected transition kernel instead of hard-coding Euclidean
  Langevin in trainers, generic potential-space SMC, and Boltzmann wrappers;
- make `e_clip` reductions finite-safe, relative to a fixed reference-energy
  origin, and report the optimizer kept fraction separately from honest ESS;
- expose a stable public spline primitive if `jflows_md` needs it without
  importing `jflows.core.*`.

`Molecular_Potential` is the sole public target class, is exported from
`jflows_md`, and subclasses `jflows.potential.Potential`. Generic resampling,
bridging, stage selection, and SMC remain in `jflows`; `jflows_md` supplies the
domain metadata, `Mixed_Domain_MALA`, and molecule-specific diagnostics rather
than forking the training algorithms.

## 1. Exact bundle-driven potential

The primary public facade loads a self-contained, versioned molecular bundle:

```python
target = Molecular_Potential.from_bundle(
    "fab_adp_ff96_obc1_v1/manifest.json",
)
```

A valid bundle contains the topology/PDB, serialized OpenMM `System.xml` (or
an equally complete prmtop plus explicit construction settings), temperature,
units, atom map, force-field and solvent provenance, package versions, and
cryptographic hashes. It also contains a hashed `CoordinateSpec`: root atoms,
deterministic placement/reference order, log-bond and logit-angle maps with
offsets/scales, periodic mask, stereocenter constraints, accepted support,
Jacobian convention, and chart version. `from_bundle` constructs
`target.coordinates` and `target.domain`; a chart override is accepted only as
a separately serialized, hash-bound spec. Reference coordinates are optional
and clearly labeled as scale/evaluation data. A coordinate-only trajectory is
rejected as an energy specification.

System-building helpers emit an auditable bundle; they are not alternative
training-time target loaders:

```python
build_bundle_from_pdb(
    "alanine_dipeptide_L.pdb",
    preset="fab_adp_ff96_obc1_v1",
    output="fab_adp_ff96_obc1_v1/",
)

build_bundle_from_amber(
    "molecule.prmtop",
    "molecule.rst7",
    build_spec=amber_build_spec,
    coordinate_spec=coordinate_spec,
    output="molecule_v1/",
)
```

The complete `build_spec` fixes temperature, OBC/ACE model, radii, dielectric,
salt, cutoff, constraints, and every other OpenMM construction option. A PDB
builder may resolve an already frozen preset and verify its atom map, but a PDB
does not reliably encode small-molecule bond order, formal charge,
protonation, or parameterization graph; those come from mapped SDF/MOL2 and
Amber artifacts.

`fab_adp_ff96_obc1_v1` is a proposed code-owned registry entry in
`jflows_md`; it does not come from fitting or reading the FAB trajectory. It is
constructed once from the audited FAB/OpenMMTools code path and canonical
ff96/OBC1 Amber system, then checked into the package as immutable parameters
and metadata. The public FAB trajectory remains independent evaluation data.

For the audited ADP preset, the PDB contributes verified atom records, ordering,
and initial geometry; the canonical Amber artifacts remain authoritative for
bonds and parameters. The preset supplies every Hamiltonian choice that a PDB
cannot encode: ff96, OBC1/ACE, radii, dielectric and salt settings, cutoff and
constraints, plus expected parameter/System hashes. It expands to the exact
force-field files and OpenMM settings above.
At build time it serializes and hashes the PDB, resolved atom mapping, complete
`CoordinateSpec`, OpenMM System XML, force-field specification, OpenMM version,
and extracted numerical arrays. The jitted target contains only JAX arrays; it
contains no OpenMM Context or host callback.

Coordinates alone, even the full million-frame FAB split, cannot uniquely
identify off-support energies, forces, temperature, solvent, or chirality
conditioning. A density or score model fitted only to those coordinates is an
approximate learned surrogate, not the benchmark potential: it cannot provide
exact MALA acceptance probabilities, SMC weights, or final importance weights.
Such a surrogate may later be exposed under a deliberately different API, but
is not accepted as `Molecular_Potential`.

### Molecules without a published benchmark bundle

For a new molecule, first choose and version a chemically applicable
parameterization, then generate MD/REMD reference data from that exact frozen
system. Reference data may estimate coordinate scales and test mode coverage;
it never defines the energy.

The archived `Molecular_BG_1/glycerol_36d` and
`Molecular_BG_1/diethanolamine_48d` examples illustrate the distinction. Their
14-atom glycerol (`OCC(O)CO`) and 18-atom neutral diethanolamine (`OCCNCCO`)
prmtops were generated using OpenFF 2.1.0/SMIRNOFF with AM1-BCC charges and
used in vacuum with `NoCutoff`. Despite an old generator docstring saying
“GAFF,” the executable code used OpenFF. Their stored implicit-solvent
parameters are not a validated OBC1 specification, so OBC1 must not simply be
inferred from the ADP preset.

For the new implicit-solvent benchmarks, define separate bundles
`glycerol_gaff2_am1bcc_obc1_v1` and
`diethanolamine_neutral_gaff2_am1bcc_obc1_v1`. Their pinned AmberTools build is

```text
explicit-H, atom-mapped 3D SDF/MOL2
  -> antechamber -at gaff2 -c bcc -nc 0 -m 1
  -> parmchk2 -s gaff2
  -> tleap: source leaprc.gaff2; set default PBRadii mbondi2
  -> prmtop + rst7
```

Freeze the canonical mapped SMILES, explicit-H input and atom order, formal
charge and multiplicity, input geometry, generated MOL2 charges/types,
`frcmod`, LEaP script/log, SQM/antechamber logs, prmtop/rst7, and System XML.
Pin the AmberTools version and hashes of `gaff2.dat`, `BCCPARM.DAT`, and
`ATOMTYPE_GFF2.DEF`; AM1-BCC output can depend on tool version and geometry.

For GAFF2 the prmtop is authoritative. Build its OpenMM reference exactly as

```python
AmberPrmtopFile(prmtop).createSystem(
    nonbondedMethod=NoCutoff,
    constraints=None,
    implicitSolvent=OBC1,
    soluteDielectric=1.0,
    solventDielectric=78.5,
    implicitSolventSaltConc=0 * molar,
    sasaMethod="ACE",
)
```

Do not reconstruct a GAFF2 molecule from a PDB plus Amber protein XML files.
Require the expected formula/atom count and net charge, no unresolved `ATTN`
or missing parameters, the expected `RADIUS_SET`, positive radii, finite
nonzero screens, finite minimized energy/forces, and an identical atom map
through SDF/MOL2/prmtop/rst7/SystemSpec. This shares FAB's OBC1 solvent
convention while using a small-molecule force field rather than ff96, which
does not parameterize arbitrary organics. These are deliberately frozen new
algorithmic benchmarks with their own REMD references, not FAB targets or
claims of state-of-the-art physical modeling.

For Cartesian coordinates `x`, the JAX force field contains:

- harmonic bonds;
- harmonic angles;
- periodic torsions;
- Amber Coulomb/Lennard-Jones terms and all exceptions;
- OBC1 descreening and Born radii, using OpenMM notation
  `rho_i=r_i-0.009 nm`, `psi_i=I_i*rho_i`, and
  `B_i^-1=rho_i^-1-r_i^-1*tanh(0.8 psi_i+2.909125 psi_i^3)`;
- GB polarization and ACE nonpolar solvation.

OBC descreening and GB pair energies have no bonded exclusions. Amber
`NonbondedForce` exceptions and 1--4 scalings apply only to the Coulomb/LJ
term and must never be reused as a GB pair mask.

The potential returns the reduced internal-coordinate energy

```text
U(q) = beta E_bundle(x(q)) - log|J_BAT(q)| - log|J_chart(q)|,
beta = 1/(k_B T).
```

No soft-cap, finite replacement-energy sentinel, or physical-energy
deformation appears in this definition. Stable float64 evaluation is primary;
consumers handle failures without changing the target: a nonfinite MALA
proposal is rejected, a nonfinite SMC/final target weight becomes `-inf`, and
a nonfinite optimizer sample is screened from that optimizer reduction only.

## 2. Bundle-defined mixed-domain coordinates

For `N` atoms, BAT gives `N-1` bonds, `N-2` angles, and `N-3` torsions. The old
linear whitening is not globally valid: Gaussian tails can create negative
bonds or angles outside `(0, pi)`. Every bundle instead freezes

- `N-1` offset/scaled log-bond variables on `R`;
- `N-2` offset/scaled logit-angle variables on `R`;
- ordinary torsions on `T = [-pi, pi)`, except any explicitly conditioned
  stereochemical torsion replaced by one Euclidean half-chart variable.

Thus an unrestricted target uses `R^(2N-3) x T^(N-3)`. ADP replaces one of
its 19 BAT torsions--it does not add a 61st degree of freedom--and therefore
uses `R^42 x T^18`. Glycerol and neutral diethanolamine remain unrestricted on
`R^25 x T^11` and `R^33 x T^15`.

All log, logit, scaling, stereochemical, and BAT Jacobians are included in
`U(q)`. The maps are bijective almost everywhere; collinear placement
references and planar boundaries remain measure-zero coordinate singularities.
No MCMC coordinate is clamped.

The bundle's `CoordinateSpec` fixes root atoms, BAT placement order and
reference overrides, scales/offsets, periodic mask, support, and convention.
An automatic fallback is not accepted until it passes conditioning,
round-trip, and hash-stability tests.

## 3. Chirality by construction

Amber energy is parity invariant, so the energy cannot distinguish L and D.
Indeed, reflecting audited configurations changes the OpenMM energy only at
roundoff. FAB's posthoc magic-index filter is transform-specific and is not a
definition of a conditional density.

Use the invariant signed volume

```text
c(x) = dot(N6-CA8, cross(C14-CA8, CB10-CA8)).
```

The desired L component has `c(x)>0`. All 1,000,000 downloaded FAB frames are
positive, with signed-volume range `[0.0013915, 0.0034028] nm^3`.

Choose the CB placement references `(CA8, C14, N6)` so that its placement
torsion is `tau_chi = dih(N6, C14, CA8, CB10)`. With the ported zflows
convention all one million L frames satisfy

```text
tau_chi in [-2.61918, -1.81033] subset (-pi, 0).
```

Parameterize the complete L half-chart smoothly as

```text
tau_chi = -pi * sigmoid(eta),  eta in R,
log|d tau_chi/d eta| = log(pi) + log(sigmoid(eta))
                       + log(1-sigmoid(eta)).
```

This `eta` replaces `tau_chi`; it is not an additional coordinate.
The implementation infers and verifies the accepted half from the L reference
rather than silently hard-coding a sign convention. The logit Jacobian makes
the planar boundaries infinite-energy limits without a hard penalty. Source,
flow, MALA, and SMC samples are therefore L by construction. A parity-folded
flow is a fallback only if the half-chart fails a bijectivity gate.

The million observed FAB frames establish empirical support but do not alone
prove the whole half-chart. Randomized geometric tests over the valid chart
must show that `tau_chi in (-pi,0)` always has the selected signed-volume sign,
the other half has the opposite sign, and the forward/inverse placement is
unique away from the planar boundary.

The current `Molecular_BG/assets/system/alanine_dipeptide.pdb` coordinates are
D (`c=-0.00262 nm^3`), whereas `l_minimum_positions_nm.npy` is L. Before
implementation, generate a canonical L PDB with the same topology and atom
ordering; never use the current PDB coordinates as the chirality oracle.

Glycerol and neutral diethanolamine have no chemical stereocenter constraint.
Their `CoordinateSpec`s retain all torsions on the torus and bundle construction
must fail if automatic stereocenter detection restricts either target. For
diagnostics, track both signs of a determinant around glycerol's central carbon
and both three-neighbor pyramidal signs around diethanolamine's nitrogen;
verify parity-energy equality and balanced full-support reference coverage.

## 4. Mixed spline flow

Current `jflows.NCSF` is invalid here because it wraps and circularly embeds
every coordinate. Implement an identity-initialized mixed spline coupling
flow:

- ordinary RQS with identity tails for the bundle's `p` Euclidean variables;
- circular C1 RQS only for its `q` torus variables;
- conditioner inputs use normalized raw Euclidean values and `(cos, sin)` for
  periodic values;
- alternating/random coupling masks mix Euclidean and torsion information;
- periodic shifts act only on torsions;
- no unrestricted LU/linear mixing across different domain types;
- Gaussian source on `R^p` and uniform source on `T^q`.

A coupling flow is preferred to an autoregressive MAF because both directions
are fast at these dimensions. A first smoke architecture may follow FAB
structurally (12 blocks, 8 bins, width 256), but those are tuning values rather
than scientific constants.

## 5. Exact mixed-domain MALA and singularity handling

Do not clamp Euclidean coordinates or naively wrap a Euclidean MALA proposal.
The mixed MALA kernel must:

- propose in the tangent coordinates;
- wrap only the bundle's `q` periodic torsions;
- use ordinary Gaussian proposal terms for `R^p`;
- use the wrapped-normal forward/reverse proposal density for `T^q`;
- include the complete asymmetric MALA correction;
- map every nonfinite proposal `log_alpha` to `-inf` and reject it;
- report acceptance, seam crossings, high-energy rejection, and chirality.

The wrapped-normal density uses an exact theta-function expression or an image
sum whose omitted-tail error is bounded from the actual proposal covariance;
a fixed unverified three-image sum is not accepted as exact MALA.

`e_clip` is a separate optimizer-only mechanism. Each bundle stores and hashes
a fixed reduced-energy origin `U_ref`; a kept optimizer sample satisfies
`isfinite(U) & (U-U_ref <= delta_e_clip)`. Masked reductions must use
`where(keep, value, 0)`, not `keep*value`, because `0*inf` is NaN. All-screened
batches trigger a skipped/retried step. Report the optimizer kept fraction and
optional clipped-batch training ESS separately. Stage selection and acceptance
use honest full-target ESS; only genuinely invalid/nonfinite log weights become
`-inf`. A clipped ESS can never accept a stage. The clipped fraction is logged
and must fall during training.

## 6. Training without sharpening

Use a potential-space bridge

```text
U_t = (1-t) U_0 + t U_1,  0=t_0 < ... < t_K=1.
```

At each stage:

1. Select the next `t_k` from incremental-weight ESS on held-out particles.
2. Generate target-stage particles with potential-space SMC and mixed MALA at
   the actual intermediate `U_t`.
3. Train an incremental mixed flow with forward KL + X regularization
   (`KLX`) using those energy-generated particles.
4. Compare trained-increment ESS with the identity/SMC fallback.
5. Accept only if the held-out ESS and mode gates pass; otherwise shrink the
   step and retry.
6. Advance the particle population by exact reweight/resample + mixed MALA.

Do not initially use KLXX, L-BFGS/quench pools, aggressive energy caps, or any
post-stage sharpening. They can obscure whether the potential, geometry, and
basic BG are correct. The current flow-proposal AIS routine rejuvenates at the
final target on nominal intermediate rungs; it is not used as the molecular
bridge. `e_clip` screens only optimizer losses.

## 7. Implementation sequence and hard gates

First validate mixed Euclidean/torus flows and MALA on analytic toy targets.
Then build and energy/force-audit all three bundles, using authoritative ADP to
validate OBC1 first. Exercise the full BG path in increasing difficulty:
glycerol (36D, unrestricted), neutral diethanolamine (48D, unrestricted), then
L-ADP (60D, conditioned chirality and rare modes). No target begins BG training
until its Gates A--F pass.

### Gate A: per-target system and bundle identity

- Every manifest freezes SystemSpec, CoordinateSpec, topology, atom map,
  microstate/net charge, temperature, units, software/data provenance, and
  hashes; loading verifies them before JIT compilation.
- ADP has a canonical L PDB and matches the OpenMMTools 0.21.5 ff96/OBC1
  system. Anchor the exact upstream commit, prmtop/CRD/LEaP hashes, and local
  System XML in the bundle.
- Glycerol is exactly `C3H8O3`, 14 atoms, charge 0; neutral diethanolamine is
  exactly `C4H11NO2`, 18 atoms with secondary N--H, charge 0. Their pinned
  AmberTools build has no `ATTN`/missing parameters, all GB radii/screens pass,
  finite minimized energies/forces pass, and the atom map is invariant across
  every artifact.

### Gate B: per-target JAX force field

- Per-term and total energies are compared with OpenMM Reference on minima,
  physical reference frames, perturbed frames, mirrored frames, and finite
  high-energy frames for every bundle.
- Float64 maximum total-energy error is `<1e-2 kJ/mol`, force RMSE is
  `<1e-2 kJ/mol/nm`, and maximum absolute force-component error is
  `<5e-2 kJ/mol/nm`.
- OBC1 descreening, GB pairs without bonded exclusions, and ACE are tested
  independently; no OBC2 coefficients or Coulomb/LJ exception masks leak in.

### Gate C: per-target coordinates and Jacobians

- Cartesian/internal round trip passes modulo rigid motion; torsions round trip
  modulo `2pi`; chart behavior is documented as almost-everywhere bijective.
- Analytic BAT/chart log-Jacobian agrees with autodiff on reduced systems.
- Cartesian and transformed energy gradients agree by the chain rule.
- The loaded domain counts/masks exactly match the benchmark matrix and the
  bundle hash; Euclidean variables are never wrapped or clamped.

### Gate D: target-specific stereochemical support

- ADP's million-frame signed-volume/torsion checks pass; randomized geometry
  proves the selected half-chart/sign equivalence and uniqueness away from the
  boundary. The source produces zero D configurations and mirrored ADP lies
  outside the L chart while retaining parity-equal Cartesian energy.
- Glycerol and neutral diethanolamine have no half-chart. Their sources cover
  both central-carbon/N-pyramid determinant signs, parity energy equality
  passes, and construction fails if either full support is accidentally
  restricted.

### Gate E: mixed flow and MALA

- Forward/inverse reconstruction and opposite log-Jacobians pass for all three
  domain masks; torsion density/Jacobian are seam-continuous.
- Gaussian x von-Mises toys recover analytic moments and pass seam-crossing
  detailed-balance flux tests and nonfinite-proposal rejection.
- Wrapped-normal evaluation is exact or meets a declared tail-error bound at
  every proposal covariance.
- Arbitrary flow/MALA outputs respect each bundle's support: ADP remains L;
  glycerol and diethanolamine retain both diagnostic signs.

### Gate F: sampler before flow

- Generic potential-space SMC with injected `Mixed_Domain_MALA` reaches `t=1`
  independently for each bundle without any `e_clip` in weights or gates.
- At least two independent SMC seeds reproduce each target's named modes and
  agree within predeclared divergence gates.
- Per-rung honest full-target ESS, MALA acceptance, nonfinite rejection,
  support diagnostics, and unchanged bundle/System hashes are saved.

### Matched references for the two new targets

Before production, freeze a manifest for each small molecule: two independent
seeds for each mirrored start, 21 replicas at 300, 350, ..., 1300 K, 1 fs
unconstrained dynamics, exchanges every 200 steps, frames every 1000 steps,
2 ns fixed burn-in, and 20 ns per slot in restartable segments below two hours.
Any protocol change after calibration creates a new reference version.

Require all walkers to visit both ladder ends, report per-edge exchange and
round trips, pass cross-seed torsion-histogram JS `<0.10` bits on a frozen
36-bin estimator, and reproduce named torsional mode occupancies. Demonstrate
both determinant/pyramidal signs through actual mixing or combine the two
mirrored ensembles with exactly equal declared weights. Save coordinates,
topology, Hamiltonian, protocol, RNG, and analysis hashes.

## 8. BG acceptance criteria and artifacts

Run at least two independent BG seeds per target. Save before plotting:

- model and optimizer states;
- complete configuration plus bundle, CoordinateSpec, and software hashes;
- raw internal and Cartesian samples;
- exact target/proposal log densities and importance weights;
- all named torsions/support diagnostics, honest stage ESS, MALA acceptance,
  nonfinite rejection, and optimizer-only clip fractions.

Report raw, importance-reweighted, resampled, and post-MALA results separately.
For every target report normalized importance-sampling ESS, cross-seed
divergence, all `d` coordinate marginals, named torsional-mode probabilities,
and reference divergence at predeclared resolutions.

For ADP additionally report the identical 100-bin Ramachandran plot, JS/TV at
24/36/48 bins, named basins including positive phi, and L fraction. For
glycerol and neutral diethanolamine report both diagnostic determinant signs
and proper-torsion marginal/mode agreement with their matched REMD references.

Initial success for a target means both seeds reach `t=1`, retain the exact
bundle hash/support, recover every predeclared mode, and pass cross-seed and
reference gates. Only ADP must remain 100% L. Success establishes an algorithmic
benchmark for the declared Hamiltonian, not modern experimental molecular
truth.

## 9. Environment strategy

The current environments are split: the `jflows` Conda environment has
OpenMM/ParmEd/MDTraj but not JAX, while the existing JAX environment lacks the
molecular stack. The preferred implementation path is Python 3.11 in the
`jflows` Conda environment, adding compatible JAX CUDA, Equinox, and test
dependencies.

System construction/reference validation runs as a short CPU/Reference process
that writes an immutable bundle containing `SystemSpec` and `CoordinateSpec`;
JAX training runs as a separate process that reads these pure-array
specifications. This avoids simultaneous OpenMM and JAX CUDA contexts and makes
the training target reproducible without OpenMM at runtime.

## Primary evidence

- FAB target: `/mnt/projects/fab-torch/fab/target_distributions/aldp.py`
- FAB configuration: `/mnt/projects/fab-torch/experiments/aldp/config/fab_buff.yaml`
- FAB chirality filter (reference only; not adopted):
  `/mnt/projects/fab-torch/fab/utils/aldp.py`
- Old molecular implementation:
  `/mnt/projects/X-regularization/Molecular_BG_2/zflows_md/`
- OpenMMTools 0.21.5 test system:
  `https://github.com/choderalab/openmmtools/blob/0.21.5/openmmtools/testsystems.py`
- Exact upstream anchors for the ADP bundle: OpenMMTools commit
  `e7e5847a677a9ecc83fcb284d0f120960416353f`; canonical prmtop SHA-256
  `2ce81216c7e18fd4d354fac44e22ba3843d89e297884bd6389a4cd57c74ecf6e`;
  CRD SHA-256
  `b8a151fd35b909de52b7f50b09f4a0c9fc5221d0b423c6a9ec3133bf867c4954`;
  LEaP input SHA-256
  `b26fb2aada22000a031768d23b1160b4702654467f26f83ad83bc1ef428c2397`.
- OpenMM implicit-solvent documentation:
  `https://docs.openmm.org/latest/userguide/application/02_running_sims.html#implicit-solvent`
