# Plan: `jflows_md` three-molecule Boltzmann-generator benchmark

## Objective and fixed decisions

The first `jflows_md` suite contains three molecular Boltzmann-generator
targets: FAB alanine dipeptide (ADP, 60D), glycerol (36D), and neutral
diethanolamine (48D). ADP reproduces a published target; glycerol and
diethanolamine extend the same package to explicitly defined small-molecule
benchmarks. The package core and validation gates are now implemented. A full
glycerol training attempt compiled and ran normally, but its per-step ESS stayed
near the identity value instead of improving. Molecular flow training is now
paused while the target singularity is isolated with sampler-only diagnostics.

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
5. The canonical `Molecular_Potential` is always the exact physical target and
   has no energy cap, distance floor, or soft chirality penalty. Explicitly
   named regularized surrogate potentials may be used for diagnostics and
   training initialization, but every accepted sharpening path must terminate
   at the unchanged physical target.
6. MALA is the rejuvenation kernel. `e_clip` is only a training-loss screen;
   it never changes MALA, SMC weights, final importance weights, or the target.
7. The million-frame FAB trajectory is evaluation data, not a likelihood
   training set. Short physical data may set coordinate scales only.
8. A target bundle freezes both the Cartesian Hamiltonian and the complete
   internal-coordinate support. Caller-supplied, unversioned coordinate
   transforms are not allowed in benchmark training.
9. The paper algorithm and generic `jflows` adaptive controller are normative.
   `jflows_md` changes are limited to the molecular potential, mixed
   Euclidean/periodic domain, and the minimum execution changes those require.

### Controller safety rule and incident history

Two severe controller/interface regressions occurred on 2026-07-12. First, an
AI-created `target-ratio C` replaced the user-required per-step ESS monitor.
Second, an AI-created `zero optimizer updates` branch shrank the bridge and
automatically retried before computing full-validation proposal ESS. Both runs
were killed before any stage or final result was promoted, and both behaviors
are removed.

The prevention rule is now explicit:

1. Per-step batch ESS is the optimizer-loop monitor. Update-applied and kept
   histories are diagnostics only.
2. The original potential-space SMC ESS gate may pre-select a bridge candidate.
   After a training attempt, compare only exact identity and the final trained
   flow on all validation particles. The better proposal's ESS is the sole
   stage accept/reject criterion. No diagnostic may shrink or retry a stage.
3. Zero optimizer updates still proceed to full-validation ESS. A nonfinite
   trained flow is assigned zero proposal ESS and excluded, while finite
   identity remains eligible.
4. Before any molecular controller edit, compare the path line by line with
   generic `jflows`, document each unavoidable molecular deviation, and add a
   focused regression for it. A new scientific control signal requires explicit
   user authorization; it cannot be introduced as defensive engineering.

The live generic `jflows` source, Git HEAD/origin, and dated ext4 snapshot were
audited after the second incident. All four adaptive drivers use the same
trained-versus-identity full-set ESS comparison and contain no zero-update-like
stage gate. An unchanged-flow controller probe accepted attempt 1 in all four
drivers, so the molecular shrink incident itself did not invalidate established
generic runs. The later direct-first forward-AIS correction is separate; its
affected private `Codes/` consumers have a controlled rerun pending.

## 2026-07-12 priority: diagnose the singular target before more BG training

This section supersedes the earlier instruction to avoid all sharpening. The
failed experiment did not show a compilation problem: the ordinary JAX policy
compiled and trained quickly. It showed a scientific problem: at the full
glycerol workload (`N_BATCH=50000`) the learned proposal did not improve the
only standard training monitor, per-step ESS. No further BG run is launched
until the following sampler-only diagnostic is complete.

For a physical Cartesian energy `E`, bundle reference energy `E_ref`, cutoff
`c`, scale `s`, and optional linear-tail fraction `rho`, define

```text
d = E - E_ref
E_(c,s,rho) = E                                      if d <= c,
E_(c,s,rho) = E_ref + c
                + (1-rho) s log(1 + (d-c)/s)
                + rho (d-c)                         if d > c,
U_(c,s,rho)(q) = beta E_(c,s,rho)(x(q)) - log J(q).
```

The shift by `E_ref` makes the cutoff portable across Hamiltonians. Only the
Cartesian energy is deformed; the coordinate Jacobian remains exact. The
physical `Molecular_Potential` is never mutated. `e_clip` remains an
optimizer-only sample screen and is unrelated to this surrogate family.

The first partial goal is a three-panel `glycerol_36d/dihedrals.png` for the
current **implicit-solvent** GAFF2/AM1-BCC/OBC1 target. It compares saved
sampler populations at several energy cutoffs for O-C-C-O `(0,1,2,3)`,
C-C-C-O `(1,2,4,5)`, and C-C-O-H `(1,2,3,10)`. Use `s=50 kJ/mol` and initially
probe `c = 50, 100, 200, 400, 800 kJ/mol`, followed by the exact target if an
honest adaptive bridge reaches it. The plotted subset may be smaller to keep
the modes readable, but all sampled levels are saved.

Sampling requirements:

1. Start from the bundle source and use potential-space SMC with mixed-domain
   MALA. Select each bridge increment from honest full-particle incremental
   ESS; resample and rejuvenate at the actual intermediate potential.
2. Run two independent seeds. A small pilot checks compilation and file paths
   only; it is never distributional or ESS evidence.
3. Save after every requested cutoff: raw internal coordinates, the three raw
   unsymmetrized torsions, exact physical and deformed energies, cap-active
   fraction, per-level ESS, MALA acceptance, seed/configuration, bundle hash,
   and source hashes. Sampling is restartable at endpoint boundaries.
4. Replot from the saved HDF5 file only. Estimate periodic marginals by a fixed
   circular histogram and wrapped Gaussian smoothing; do not rerun dynamics to
   tune the figure.
5. Require both seeds to retain the same named torsional modes before treating
   a curve as evidence. Record cross-seed histogram divergence and adjacent-cap
   reweighting ESS. The exact endpoint is mandatory before any physical claim.

The old `Molecular_BG_1/glycerol_36d/dihedrals.png` is context only. It used a
vacuum Hamiltonian plus the jointly deformed `e_cap=200`, `r_floor=0.1 nm`
target; it neither isolates the effect of `e_cap` nor validates the current
implicit-solvent distribution. Accordingly, differences in the new curves are
expected and must not be cosmetically forced to match the archive.

The archived lin-log form motivates the first cutoff scan, especially the
observed 100--200 kJ/mol transition, but not its final choice. A cutoff is
acceptable for training initialization only if it retains all stable modes,
has adequate adjacent-level ESS, and admits a reliable adaptive path to the
exact target. A separate collision-distance regularizer may be studied later,
but it is not mixed into this first experiment. The `rho=0` logarithmic tail
also needs a normalizability/coercivity audit before production use; a positive
linear tail is the fallback.

Two alternatives remain pending rather than combined with this scan: train a
separate generator at 600 K or 1200 K and cool it to 300 K, or regularize
individual collision terms. A temperature bridge must use
`beta_T E(x(q)) - log J(q)`; scaling the complete reduced potential would
incorrectly scale the Jacobian.

### First diagnostic outcome

The 2026-07-12 **exploratory** run completed under `~/.envs/jflows` with two
independent 100000-particle seeds and ordinary JAX compilation. It reached
every requested cutoff and the exact potential in about 82 seconds. The saved
artifact is `glycerol_36d/dihedrals_samples.h5`; the figure is reconstructed
from it by `glycerol_36d/dihedrals.py replot`. This run establishes the visible
mode structure and the location of the useful cutoff transition. It is below
the full `N_VALID=1000000` standard and is not equilibrium or production ESS
evidence.

The source-to-`c=50` transition required 13 adaptive levels in both seeds; its
direct ESS was only `1.77e-5` and `1.12e-5`. At the sampled `c=50` endpoint,
70.8% and 83.9% of frames were still above the cutoff, and the worst torsion
cross-seed JS divergence was `0.1065` bits. Thus `c=50` is visibly deformed and
not yet an accepted equilibrium reference, although both seeds retain the
same three named modes.

The `c=50 -> 100` transition required two levels. At `c=100`, only 0.020% and
0.018% of frames activated the cap. The direct `c=100 -> 200` ESS was
`0.999992` and `0.999998`; all later `200 -> 400 -> 800 -> exact` transitions
had ESS `1.000000`, with no sampled frame above 200 kJ/mol excess energy.
Exact-potential mixed-MALA acceptance was about 0.857. Exact cross-seed JS values
for O-C-C-O, C-C-C-O, and C-C-O-H were `0.0228`, `0.0102`, and `0.0341` bits,
and all three exact marginals retain clear three-mode structure. Aggregate
`c=100` versus exact marginal JS values are at most `0.0003` bits.

The marginals hide incomplete joint mixing. At the exact-potential endpoint,
the two seeds have 12-by-12-by-12 selected-torsion joint JS `0.2246` bits,
coarse 3-by-3-by-3 rotamer JS `0.0460` bits, maximum single-coordinate JS over
all 11 internal torsions `0.1150` bits, central determinant-positive fractions
`0.5320` versus `0.3160`, and a median excess-energy difference of about
`1.63 kJ/mol`. At `c=100`, the corresponding joint JS is `0.2650` bits and the
determinant-positive fractions are `0.5341` versus `0.3163`. Thus the gray
curve is an exact-**potential** population, not an accepted exact-equilibrium
reference. High acceptance with persistent population differences points to
insufficient intermode movement rather than MALA rejection.

Consequently `c=100 kJ/mol` above `E_ref` is only the leading
**next-diagnostic candidate**. Conditional on each already discovered basin
mixture, `c=100 -> exact` has ESS `0.999992` and `0.999998` and negligible
marginal change; this does not prove the basin weights. Before BG training,
run a fresh direct source-to-`c=100`-to-exact diagnostic with all 11 torsions,
joint rotamers, determinant signs, longer/tuned rejuvenation, and the full
`N_VALID=1000000` population. Do not simply reuse the exploratory MALA setting
(`step=1e-3`, 20 transition iterations, 50 endpoint iterations): its high
acceptance together with persistent joint disagreement indicates insufficient
movement. Tune the step and/or increase rejuvenation in a bounded pilot first.

There is also little coercivity margin in the pure logarithmic tail. For a
collective dilation of glycerol's 13 bonds, normalizability requires
`s > (39/2) k_B T = 48.64 kJ/mol` at 300 K; `s=50` clears this by only
`1.36 kJ/mol`. Prefer a positive `rho` or a materially larger `s` in the next
run and repeat the cutoff comparison. The driver therefore defaults future,
new-path diagnostics to `N=1000000` and `rho=0.01`; these settings have not yet
been run or validated. The HDF5 now records `E_ref`, geometry,
deformation-spec hashes, all-torsion/joint diagnostics, and both the original
sampling-script hash and later audit-script hash. Sampling resume refuses a
script-hash mismatch; this exploratory artifact itself predates that strict
guard and must not be extended with a changed sampler.

## Implementation update: molecular compilation and test tiers

Current status: generic scientific controller semantics remain authoritative in
the single `jflows` checkout, while the mixed-domain execution boundary below
is implemented in `jflows_md`. The complete bounded smoke suite passes, and
an isolated 18-cell RTX 5090 benchmark covers flow maps, both public one-step
trainers, the real glycerol potential/gradient, and a fixed-shape molecular
MALA chunk. All benchmark cells pass; the largest observed peak is below
1.64 GB host RSS and 340 MB backend-reported GPU use. A later full glycerol
KLXX run also compiled promptly under the ordinary JAX policy, but its
per-step ESS stayed near the identity baseline; that scientific failure is the
reason for the regularization diagnostic above.

The molecular controller must not put chunk or ladder-level Python loops
inside one JIT. Two scaled glycerol attempts demonstrated that this boundary
is not viable: the default XLA policy spent more than 15 minutes in the first
SMC compilation and grew toward 53 GB host memory; retrying with
`JAX_DISABLE_MOST_OPTIMIZATIONS=1` reduced memory to about 20 GB but still did
not finish the first compilation in comparable time. Neither attempt reached
the first SMC-selection log line, so optimizer batch size, optimizer steps,
and the flow-training scan were not the cause of that initial stall.

Lowering the actual glycerol potential isolates the problem. The mixed-MALA
StableHLO grew from approximately 0.79 million characters at `chunk=1` to
5.52 million at `chunk=8`, with one copied scan/force-field path per chunk.
With `ladder=6` and `chunk=128`, the current outer JIT exposes 768 chunk paths
before XLA optimization. This also defeats the intended memory meaning of
`chunk`: XLA may overlap independent in-graph chunks. The live `jflows`
chunk regression test and the old `zflows_md` implementation both require an
eager Python chunk controller around a compiled fixed-shape kernel.

The API-preserving compilation plan is:

1. Make `mixed_mala` an eager chunk controller. Compile one private
   single-chunk trajectory kernel containing the `mc_iters` `lax.scan`; call
   it once per physical chunk and concatenate outside JIT.
2. Keep SMC and AIS level, reweighting, resampling, and chunk loops in eager
   Python. Compile only fixed-shape per-chunk weight and MALA kernels. Classical
   potential-space SMC rejuvenates at each actual intermediate bridge. The
   flow-proposal training AIS is intentionally different: its nominal
   incremental weights are followed by final-target rejuvenation at every
   level, keeping MCMC score-free in the flow at the cost of a biased target
   surrogate.
3. Make molecular Boltzmann identity weights, trained weights, flow inverse,
   and validation passes eager chunk wrappers around private no-chunk JIT
   kernels. This is a self-contained molecular implementation and does not
   import private helpers from `jflows`.
4. The molecular trainer retains its bounded packed optimizer `lax.scan`; it
   consumes an already prepared SMC pool and nests no sampling controller.
   Generic `jflows` trainers retain their original whole-stage compiled
   `lax.scan`, including their sampling path. The different compilation
   treatments are intentional and are not a compatibility requirement.
5. As a secondary optimization, consider fusing molecular energy and gradient with
   `vmap(value_and_grad(single_energy))` so each MALA endpoint does not repeat
   the primal force-field evaluation. Do not change MALA acceptance or the
   physical target.
6. Use the normal JAX compiler policy for the redesigned small kernels.
   `JAX_DISABLE_MOST_OPTIMIZATIONS=1` remains a diagnostic fallback, not the
   primary fix. Enable a persistent compilation-cache directory only after a
   bounded compile succeeds; leave its size unlimited (`-1`).

Testing proceeds in two tiers and never jumps directly to production:

| Tier | N_VALID | N_POOL | N_BATCH | STEPS | LADDER | MC_ITERS | CHUNK |
|---|---:|---:|---:|---:|---:|---:|---:|
| compile/path smoke | 2048 | 1024 | 32 | 2 | 2 | 1 | 8 |
| scaled compile smoke | 60000 | 12000 | 120 | 100 | 6 | 20 | 128 |
| current glycerol run | 1000000 | 200000 | 50000 | 100 | 12 | 100 | 8 |

The bounded component/full-path suite now exercises SMC, score-free AIS, mixed
MALA, both trainers, full-set weights, resampling, one adaptive BG stage, all
three molecular potentials, and default-float32 checkpointing. Its real
glycerol energy/gradient and one-step MALA compiles finish in about 3 and 5
seconds, respectively. A side-by-side backup comparison gives bitwise-identical
G-AIS, SMC, two-step KLX parameters, and ESS; F-AIS differs only by
`1.67e-6` from compiled arithmetic reassociation. The scaled smoke is only a
compile/path test: its ESS is too noisy to assess a training method. Scientific
ESS evidence uses the full workload. No individual run may exceed two hours.

ESS evaluation uses all `N_VALID` particles at every stage decision. The
selected full-validation proposal ESS is the sole post-training acceptance
quantity. Save separately the optimizer `ess_history`, trained validation ESS,
identity validation ESS, selected map, kept fraction, and update-applied
history; the latter two never control acceptance. ESS is stochastic and need
not rise at every optimizer step. Report, per stage,
the first/last value, first-20 versus last-20 median, linear trend, fraction of
positive adjacent changes, and whether the trained or identity map won. The
aggregate evidence for improvement is a positive robust trend and higher
late-window median in a majority of trained stages, not strict monotonicity.

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

## Implemented package boundary

```text
jflows_md/
  system.py          # immutable molecular-bundle loading and verification
  artifacts.py       # versioned source hashes and verified flow loading
  potential.py       # Molecular_Potential facade and Cartesian reconstruction
  source.py          # Gaussian Euclidean x uniform torus source
  flow.py            # mixed Euclidean/circular spline coupling flow
  train.py           # mixed-domain molecular KL+X stage trainer
  boltzmann.py       # adaptive molecular Boltzmann-generator ladder
  utils.py           # mixed MALA, classical SMC, and score-free AIS
  core/
    forcefield.py    # pure-JAX Amber bonded/nonbonded and OBC1/ACE energy
    coordinates.py   # bundle-defined BAT transform and exact Jacobians
    chirality.py     # target-specific signed-volume/support diagnostics
    domain.py        # R^p x T^q metadata and periodic operations
    flow.py          # low-level mixed spline coupling transforms
    validation.py    # OpenMM/geometry validation helpers
```

`jflows` remains the generic flow/training engine. Molecular physics,
coordinates, chirality, and diagnostics belong in `jflows_md`. The verified
generic fixes live only in the authoritative `/mnt/projects/jflows` checkout;
the molecular workspace does not contain a shadow copy. Possible later
generic extensions are:

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
    "adp_ff96_obc1/manifest.json",
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
    preset="adp_ff96_obc1",
    output="adp_ff96_obc1/",
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

`adp_ff96_obc1` is a code-owned registry entry in
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
`glycerol_gaff2_am1bcc_obc1` and
`diethanolamine_gaff2_am1bcc_obc1`. Their pinned AmberTools build is

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
deformation appears in this definition. Training and molecular evaluation use
JAX's default float32 dtype; neither the package nor the active driver enables
x64. Consumers handle failures without changing the target: a nonfinite MALA
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

The obsolete D-valued exploratory PDB has been removed. The canonical
`bundles/adp_ff96_obc1/reference.pdb` and first stored validation frame
are L and share the frozen topology and atom ordering. Bundle construction and
smoke tests verify the accepted determinant sign; an arbitrary PDB must never
be used as the chirality oracle.

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
batches skip that optimizer update. They do not trigger a stage shrink or
retry. Report the optimizer kept fraction and
optional clipped-batch training ESS separately. Stage selection and acceptance
use honest full-target ESS; only genuinely invalid/nonfinite log weights become
`-inf`. A clipped ESS can never accept a stage. The clipped fraction is logged
and must fall during training.

## 6. Regularized initialization and exact sharpening

After the sampler-only cutoff scan passes, use a two-axis potential-space
construction. The first axis trains at a declared regularized endpoint; the
second sharpens that endpoint to the exact physical target. Within either
fixed pair of endpoint potentials use

```text
U_t = (1-t) U_0 + t U_1,  0=t_0 < ... < t_K=1.
```

At each bridge level:

1. Select the next `t_k` from incremental-weight ESS on held-out particles.
2. Generate target-level particles with potential-space SMC and mixed MALA at
   the actual intermediate `U_t`.
3. Once the sampler diagnostic is accepted, train an incremental mixed flow
   with forward KL + X regularization using those energy-generated particles.
4. Compare trained-increment ESS with the identity/SMC fallback.
5. Accept only if the selected proposal's full-validation ESS clears
   `tau_ess`; otherwise shrink the step and retry. Mode diagnostics may stop a
   scientific experiment outside the controller, but cannot trigger a
   controller shrink or acceptance.
6. Advance the particle population by exact reweight/resample + mixed MALA.

Cutoff increases are adaptive and determined by full-particle reweighting ESS,
not by an arbitrary geometric schedule. Each transition reweights with the
complete difference between the two declared potentials, resamples, and uses
mixed MALA invariant for the new intermediate target. Save the adjacent-cutoff
ESS and reject a transition that destroys a named torsional mode. The final
endpoint is the exact `Molecular_Potential`, never a largest finite cap.

Do not initially combine the energy cutoff with a distance floor, temperature
change, delta-QT surrogate, or another hidden deformation. KLXX and the
wide-coverage pool can be reconsidered only after the target ladder itself is
validated. The current flow-proposal AIS routine rejuvenates at the final
target on nominal intermediate ladder levels; it is not used as the molecular
potential bridge. `e_clip` screens only optimizer losses.

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
- Per-level honest full-target ESS, MALA acceptance, nonfinite rejection,
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

The active environment is the pip-only virtual environment
`~/.envs/jflows`. It contains the latest compatible CUDA-13 JAX/Equinox stack,
OpenMM/ParmEd/MDTraj, and the plotting/scientific dependencies. The former
Conda environment and `~/.envs/jax` are retired. Neither `jflows` nor
`jflows_md` is installed locally; experiment commands select both live source
trees explicitly with `PYTHONPATH`.

System construction/reference validation runs as a short CPU/Reference process
that writes an immutable bundle containing `SystemSpec` and `CoordinateSpec`;
JAX training runs as a separate process that reads these pure-array
specifications. This avoids simultaneous OpenMM and JAX CUDA contexts and makes
the training target reproducible without invoking OpenMM at runtime. The
checked-in bundles are sufficient for ordinary training and evaluation.
AmberTools is an optional provenance dependency only for rebuilding the two
GAFF2 small-molecule bundles and is discovered through `AMBERHOME` or `PATH`.

## 10. Immediate molecular-training diagnostic pivot

The unsuccessful glycerol c50 KLXX attempts are halted.  Before another
glycerol optimization, the current trainer must pass a deliberately ordered
small-molecule diagnosis.  This diagnostic phase does not silently change the
public `jflows` or `jflows_md` packages: a proven package defect produces a
minimal patch specification and a separate authorization gate.

### Primary goal: soft methane-to-butane series

Use explicit-H labeled methane, ethane, propane, and n-butane, with internal
dimensions 9, 18, 27, and 36.  Freeze one small `Mixed_NSF` architecture and
one set of sample counts, key roles, and diagnostics for the complete series.
For each molecule compare the exact physical target with declared c50
regularizations.  The primary soft target is coercive (`cut=50 kJ/mol`,
`scale=50 kJ/mol`, positive residual slope `rho=0.01`); the old pure-log
`rho=0` form is a separately named diagnostic and stops at n-butane because
its tail-normalizability margin fails for the next homologue.

No CH4 optimization starts before all of the following pass:

1. an independently specified shifted-Gaussian/von-Mises mixed-domain oracle;
2. CH4 topology, charge, chart round trip, support, and permutation audits;
3. OpenMM/JAX energy-and-force parity plus finite float32 energy gradients;
4. direct formula checks for the KL/KLXX loss, masking, parameter update, and
   both target-side and proposal-side importance-weight conventions; and
5. disjoint selection and never-selected audit populations.

Per-step loss and batch ESS are diagnostics.  A trained result is judged on a
frozen, never-selected audit population, using saved exact log weights and a
paired comparison with identity.  Scientific finite-sample improvement
requires `Delta ESS >= 0.02`, a paired block-bootstrap 95% lower bound above
zero, and improvement in at least three of four independent blocks.  The
Gaussian Euclidean source and identity tails of `Mixed_NSF` do not match the
exponential chart-boundary tails of the molecular target, so finite-N ESS must
not be presented as a positive asymptotic chi-square-overlap claim.

Proceed CH4 -> ethane -> propane -> n-butane and stop at the first unreconciled
failure.  Replicate the first failing molecule and its adjacent passing member.
Save bundles, pools, keys, per-step raw arrays, parameter deltas, masks, exact
log weights, nested-N ESS, block estimates, environment, and code hashes so
every plot can be regenerated without rerunning molecular sampling.

### Secondary goal: 36D vacuum glycerol controls

After the CH4 gate is understood, run two clearly separated vacuum controls:

1. **Causal solvent ablation:** keep the current GAFF2 glycerol topology,
   chart, source, flow, particles, and regularization fixed, but construct the
   Cartesian vacuum energy before regularization by removing the complete
   GB/ACE contribution.  Validate it against an OpenMM `NoCutoff`,
   unconstrained, no-implicit-solvent System.
2. **Archived original-style Hamiltonian:** use the preserved OpenFF Sage 2.1
   / AM1-BCC vacuum PRMTOP and RST7 from `Molecular_BG_2/zflows_md/data` with a
   newly frozen current coordinate specification.  Record that this is not an
   exact rerun: the old stochastic whitening arrays, gauge-slice measure,
   clamped source revision, and PRNG state are unavailable.  The historical
   all-circular NCSF, `r_floor=0.1 nm`, 100-to-200 energy cap, and ULA settings
   are optional labeled ablations, not solvent evidence.

The causal pair must differ only by the solvent energy.  Results from the
archived OpenFF Hamiltonian must never be pooled with that comparison.

The recovery-critical preregistration, exact thresholds, and outcome table
live in `.aris/experiments/molecular_training_diagnostics/EXPERIMENT_PLAN.md`.
The detailed read-only `zflows`/`jflows` replanting audit is complete; its
direct-first forward-AIS repair is frozen in the package commits recorded by
the experiment environment snapshot.

## 11. BG histories and recoverable attempt flows

Molecular training is paused while the generic and molecular BG interfaces
gain a recovery-grade stage record.  This is a diagnostic/interface change;
it must not alter losses, random keys, optimization, SMC/AIS, the
trained-versus-identity gate, or any acceptance decision.

Every committed stage records compact, attempt-aligned fields:

- `t_hist`: bridge coefficients that reached training and full validation;
- `batch_ess_hist`: optimizer-minibatch ESS, with shape
  `[attempt, train_step]`;
- `valid_trained_ess_hist` and `valid_identity_ess_hist`: honest ESS on the
  complete validation population for every trained attempt; and
- `attempt_status_hist` and `trained_flow_path_hist`: the disposition and
  recoverable trained-flow file corresponding one-to-one with those arrays.

The committed scalar summary is `t`, `valid_selected_ess`,
`valid_trained_ess`, `valid_identity_ess`, `selected`, and `flow`, with
`selected_flow_path` naming its durable file.  The identity map is never
called "trained".  The old scalar `imp_history` is derivable and is retired;
temporary read compatibility does not make it a canonical field.

Each trained candidate is serialized atomically outside compiled code as soon
as full validation finishes, including rejected candidates and candidates that
lose to identity.  Stable `stage_NNNN/attempt_NNNN` directories contain the
flow, per-step arrays, and metadata; a run manifest preserves terminal failed
stages that correctly do not enter the composable returned stage list.  Paths
stored in records are relative so the complete experiment directory can move.
Saving is optional at package level and mandatory in molecular experiment
drivers.  Tests require saving-on/off numerical identity, rejected-flow reload,
history/path alignment, and exact selected-flow reconstruction.

The accompanying naming cleanup uses `dt` only for an integration step size
and `steps` only for a count.  Composite controls therefore use `mc_dt` /
`mc_steps`, `opt_alpha` / `opt_steps`, and `train_steps`; sizes use
`batch_size` and `pool_size`.  `ladder` remains the annealing-level count.
Existing public keyword spellings remain compatibility aliases so archived
`Codes` scripts retain exactly the same numerical behavior and require no
scientific rerun from this interface-only change.

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
