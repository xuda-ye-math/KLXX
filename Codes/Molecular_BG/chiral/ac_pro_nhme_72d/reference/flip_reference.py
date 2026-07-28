"""Independent equilibrium reference for Ac-Pro-NHMe by basin-jump Monte Carlo.

The ACE-PRO amide barrier is 60-65 kJ/mol, about 26 kT at 300 K.  No unbiased
local sampler crosses it: in a 4.5 ns pilot, every one of 3,722 basin changes
came from the jump move below and not one from the dynamics.  Every earlier
attempt at this molecule bought crossings by deforming the target -- parallel
tempering, deposited metadynamics hills, a frozen analytic bias -- and then paid
for it, in reweighting variance, in a residual barrier, or in needing a seed
drawn from the very generator the run was meant to check.

This driver pays nothing.  The cis and trans basins sit exactly pi apart in
omega, so rotating the proline side of the ACE-PRO amide bond rigidly by pi maps
one basin onto the other.  Write that map T.  It is

  * an involution -- applying it twice is a 2*pi rotation, the identity, so
    T(T(x)) = x exactly; and
  * volume preserving -- it is orthogonal with det = +1, a proper rotation, so
    the Jacobian is exactly 1 and chirality is untouched.

The rotation axis is fixed by atoms 4 and 6.  Atom 6 is *inside* the moved
block -- it seeds the block search -- so the volume argument cannot rest on the
axis atoms being unmoved.  What it rests on is that both are *fixed points* of
the map, for two different reasons: atom 4 lies outside the block, and atom 6 is
the rotation origin, so R(x6 - x6) + x6 = x6 identically, and bit-exactly in
floating point.  Ordering the coordinates as (fixed atoms, atom 6, the rest of
the block) therefore makes dT/dx block *lower* triangular with diagonal blocks
I, I and R: the dependence of R on the configuration lives entirely in the
strictly lower blocks and cannot contribute.  det = det(R)^k = 1 exactly,
since R = 2 n n^T - I has eigenvalues (+1, -1, -1) in three dimensions.

Getting this reason right matters: an edit that moved the axis onto a genuinely
mobile atom would destroy exactness while still passing every check in
verify_move.

A symmetric volume-preserving proposal accepted with min(1, exp(-dU/kT)) obeys
detailed balance, so the *jump* is exact.  There is no weight, no reweighting
and no tempering: every retained sample counts once, at the target temperature,
and the estimator is a plain average.  The run needs no input from any earlier
calculation -- it starts from the bundle's own reference geometry.

**What "exact" does and does not cover.**  The jump is exact; the dynamics is
not.  LangevinMiddleIntegrator is an unadjusted discretization whose
configurational averages carry an O(dt^2) error, and a composition of a kernel
preserving exp(-U/kT) with one preserving the discrete stationary law preserves
neither exactly.  So the honest statement is: no importance weights, no
tempering, no reweighting, and the only bias is the integrator's O(dt^2) term.
At dt = 0.5 fs the stiffest mode is the X-H stretch, omega*dt = 0.31, which
inflates that bond's configurational variance by about 2.4 per cent; but the
observable is a ratio of basin populations, bond and angle frequencies are
alike in cis and trans, so that leading error cancels, and the soft modes that
distinguish the basins have omega*dt < 0.02 and relative errors below 1e-4
against a statistical error near 4e-4.  That is an estimate, not a measurement:
--timestep-fs 0.25 reruns the same chain with the bias cut fourfold, and
agreement within the error bar is what would bound it empirically.

The composite kernel is invariant but *not* reversible -- each factor is
reversible, but Langevin^n composed with the jump is not -- so results should be
quoted as invariance, never as detailed balance of the whole sweep.

The move acts on positions only.  The target factorizes as exp(-U(x)/kT) times
exp(-K(v)/kT), so leaving the velocities alone preserves the joint density and
the acceptance ratio needs the potential energy alone.

**Why the checks below are absolute, not relative.**  An earlier version of this
code computed dihedrals with the wrong sign on b0, which offsets every dihedral
by exactly pi.  That error is invisible to relative tests: the value still lies
on (-pi, pi], the flip still shifts omega by pi, and chi1 is still invariant.
It only swaps which basin is called cis, and it turned a correct cis = 0.14 into
a reported 0.86.  So verify_observables compares absolute dihedrals against
OpenMM's own CustomTorsionForce -- the same torsion the landscape figures use --
rather than trusting the formula.

Run:
    python flip_reference.py                 # production, 500 ns
    python flip_reference.py --samples 20000 # short check
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import openmm as mm
from openmm import unit

from jflows_md import Molecular_Bundle
from jflows_md.openmm import OpenMM_Potential


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "bundle"
SERIES = HERE / "flip_reference_series.npz"
FRAMES = HERE / "flip_reference_frames.npz"
MANIFEST = HERE / "flip_reference.json"
LOG = HERE / "flip_reference.log"

TEMPERATURE_KELVIN = 300
FRICTION_PER_PS = 1
PLATFORM = "CUDA"
SEED = 20260727

# Real masses, no hydrogen mass repartitioning, short step.  The jump acceptance
# does not depend on the step; the Langevin discretization error does, and 0.5 fs
# keeps it far below the statistical error this run is chasing.
TIMESTEP_FS = 0.5

# One jump attempt per this many dynamics steps.  Each attempt costs one energy
# evaluation and one host sync.  Measured in the pilot: 0.25 ps gives 21%
# acceptance, 827 basin changes per ns and an integrated autocorrelation time of
# 0.37 ps for the cis indicator.  The estimator is exact at any interval -- the
# 2.5 ps pilot agreed to within its error bar -- so this is a variance knob.
STEPS_PER_ATTEMPT = 500

DISCARD_SAMPLES = 4_000          # 1 ns
PRODUCTION_SAMPLES = 2_000_000   # 500 ns
REPORT_SAMPLES = 4_000           # one report per ns
# Checkpoint every this many reports.  Rewriting a two-million-sample compressed
# series once per nanosecond would spend more wall clock on zlib than on
# dynamics, so the log stays per-nanosecond while the arrays land every 20 ns
# and always on the final sample.
CHECKPOINT_REPORTS = 20
# Total: 2,004,000 attempts x 500 steps = 1,002,000,000 steps = 501 ns.

# Cartesian frames retained for the landscape comparison, thinned evenly out of
# the production samples; the full torsion series is kept at every attempt.
FRAME_TARGET = 100_000

# Identical to plot_conformational_landscape.py, so the reference and the
# generator are read through the same definitions.
PHI_ATOMS = (4, 6, 16, 18)
PSI_ATOMS = (6, 16, 18, 20)
OMEGA_ATOMS = (1, 4, 6, 16)
CHI1_ATOMS = (6, 16, 13, 10)
RING_ATOMS = (6, 16, 13, 10, 7)  # pyrrolidine; must lie inside the moved block
FLIP_BOND = (4, 6)               # rotation axis: C_ACE -> N_PRO

KB_KJ_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def dihedral(positions: np.ndarray, atoms: tuple[int, ...]) -> float:
    """Return the signed a-b-c-d dihedral in radians on (-pi, pi].

    b0 points from p1 *back* to p0.  Taking it the other way round negates both
    arctan2 arguments and shifts the result by exactly pi, silently swapping the
    cis and trans labels; verify_observables guards against that.
    """

    p0, p1, p2, p3 = (positions[i] for i in atoms)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def moved_block(bonds: np.ndarray, bond: tuple[int, int]) -> np.ndarray:
    """Return the atoms on the ``bond[1]`` side once ``bond`` is removed.

    If the bond lay on a ring the graph would stay connected and every atom
    would be reachable, so the "rigid" rotation would tear the ring open.  That
    case raises rather than silently producing nonsense.
    """

    first, second = bond
    neighbours: dict[int, set[int]] = {}
    for a, b in bonds:
        a, b = int(a), int(b)
        if {a, b} == {first, second}:
            continue
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    seen = {second}
    stack = [second]
    while stack:
        node = stack.pop()
        for nxt in neighbours.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if first in seen:
        raise ValueError(
            f"bond {bond} lies on a ring: removing it leaves the graph "
            "connected, so no rigid block exists"
        )
    return np.array(sorted(seen), dtype=np.int64)


def flip(positions: np.ndarray, block: np.ndarray, bond: tuple[int, int]) -> np.ndarray:
    """Rotate ``block`` by pi about the axis through ``bond``.

    For a unit axis n the pi rotation is R = 2 n n^T - I, which satisfies
    R @ R = I identically and has det R = +1.  Atom ``bond[0]`` is outside the
    block and atom ``bond[1]`` is the rotation origin, so both are fixed points
    of the map even though the second one belongs to the block; the axis is
    therefore recomputed identically on the second application, which is what
    makes the map an involution rather than merely close to one.
    """

    origin = positions[bond[1]]
    axis = positions[bond[1]] - positions[bond[0]]
    axis = axis / np.linalg.norm(axis)
    rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
    out = positions.copy()
    out[block] = (positions[block] - origin) @ rotation.T + origin
    return out


def verify_observables(positions: np.ndarray, n_atoms: int) -> None:
    """Check every dihedral against OpenMM on randomly perturbed geometries.

    This is the absolute test the relative ones cannot replace: a constant pi
    offset cancels in every difference but inverts the cis/trans assignment.
    """

    probe = mm.System()
    for _ in range(n_atoms):
        probe.addParticle(1.0)
    collective = mm.CustomCVForce("phi")
    names = ("phi", "psi", "omega", "chi1")
    groups = (PHI_ATOMS, PSI_ATOMS, OMEGA_ATOMS, CHI1_ATOMS)
    for name, atoms in zip(names, groups, strict=True):
        torsion = mm.CustomTorsionForce("theta")
        torsion.addTorsion(*atoms)
        collective.addCollectiveVariable(name, torsion)
    probe.addForce(collective)
    context = mm.Context(
        probe, mm.VerletIntegrator(1e-6), mm.Platform.getPlatformByName("Reference")
    )
    rng = np.random.default_rng(0)
    worst = dict.fromkeys(names, 0.0)
    for trial in range(64):
        sample = positions if trial == 0 else positions + rng.normal(
            0.0, 0.05, positions.shape
        )
        context.setPositions(sample * unit.nanometer)
        reference = collective.getCollectiveVariableValues(context)
        for name, atoms, value in zip(names, groups, reference, strict=True):
            gap = abs(
                (dihedral(sample, atoms) - value + np.pi) % (2.0 * np.pi) - np.pi
            )
            worst[name] = max(worst[name], gap)
    report = "  ".join(f"{name} {gap:.2e}" for name, gap in worst.items())
    log(f"  dihedral convention vs OpenMM, max disagreement: {report} rad")
    for name, gap in worst.items():
        if gap > 1e-6:
            raise ValueError(
                f"{name} disagrees with OpenMM by {gap:.6f} rad "
                f"({gap / np.pi:.3f} pi): the basins would be mislabelled"
            )

    # The check above compares the same atom indices on both sides, so it
    # cannot catch a wrong *selection* -- substituting the ACE carbonyl oxygen
    # for atom 1 would offset omega by about pi, invert the labels, and still
    # pass.  This is the independent physical anchor: the bundle's reference
    # geometry is built trans, so omega must sit near +/-pi, not near 0.
    reference_omega = dihedral(positions, OMEGA_ATOMS)
    log(
        f"  reference geometry omega = {reference_omega:+.6f} rad "
        f"({np.degrees(reference_omega):+.2f} deg), must be trans"
    )
    if abs(reference_omega) <= np.pi / 2.0:
        raise ValueError(
            f"the bundle reference geometry reads as cis "
            f"(omega = {reference_omega:+.4f} rad). It is built trans, so "
            "either the dihedral convention or OMEGA_ATOMS is wrong"
        )


def verify_move(positions: np.ndarray, block: np.ndarray) -> None:
    """Check the structural properties the jump must have."""

    ring = set(RING_ATOMS)
    log(f"  block: {len(block)} of {len(positions)} atoms move {block.tolist()}")
    if not ring <= set(block.tolist()):
        raise ValueError(
            f"pyrrolidine ring {sorted(ring)} is not contained in the moved "
            f"block: the rotation would tear the ring"
        )
    log(f"  ring {sorted(ring)} lies wholly inside the moved block: OK")

    axis = positions[FLIP_BOND[1]] - positions[FLIP_BOND[0]]
    axis = axis / np.linalg.norm(axis)
    rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
    determinant = float(np.linalg.det(rotation))
    log(f"  rotation determinant {determinant:+.15f} (proper rotation, +1)")
    if abs(determinant - 1.0) > 1e-9:
        raise ValueError(
            f"rotation has determinant {determinant}: an improper rotation "
            "would invert the stereocentre"
        )

    # det(R) = +1 is a statement about a 3x3 matrix, not about the map on all
    # 3N coordinates -- and the Metropolis ratio is only correct if the full
    # Jacobian is 1.  Because the axis is recomputed from the configuration,
    # that is a claim about a nonlinear map and it has so far only been argued,
    # never measured.  Here it is measured, by central differences at a
    # perturbed geometry so the test does not sit on a symmetric special case.
    probe = positions + np.random.default_rng(7).normal(0.0, 0.03, positions.shape)
    n_coords = probe.size
    jacobian = np.empty((n_coords, n_coords))
    step = 1e-6
    for column in range(n_coords):
        shift = np.zeros(n_coords)
        shift[column] = step
        plus = flip((probe.ravel() + shift).reshape(probe.shape), block, FLIP_BOND)
        minus = flip((probe.ravel() - shift).reshape(probe.shape), block, FLIP_BOND)
        jacobian[:, column] = (plus.ravel() - minus.ravel()) / (2.0 * step)
    full = float(np.linalg.det(jacobian))
    log(f"  full {n_coords}x{n_coords} Jacobian determinant {full:+.9f} (must be +1)")
    if abs(full - 1.0) > 1e-5:
        raise ValueError(
            f"the map has Jacobian determinant {full}, not 1: the Metropolis "
            "ratio would need a Jacobian factor and the chain would be biased"
        )

    once = flip(positions, block, FLIP_BOND)
    twice = flip(once, block, FLIP_BOND)
    residual = float(np.abs(twice - positions).max())
    log(f"  involution: max|T(T(x)) - x| = {residual:.3e} nm")
    if residual > 1e-9:
        raise ValueError(f"move is not an involution: residual {residual:.3e} nm")

    # omega must move by pi; the other three must not move at all.  phi is
    # invariant for a structural reason worth stating: atoms 4 and 6 both lie on
    # the rotation axis, so the rotation maps all four of phi's atoms, and a
    # dihedral is invariant under a rigid rotation of all four.  psi and chi1 are
    # invariant because all four of their atoms sit inside the moved block.
    for name, atoms, expected in (
        ("omega", OMEGA_ATOMS, np.pi),
        ("phi", PHI_ATOMS, 0.0),
        ("psi", PSI_ATOMS, 0.0),
        ("chi1", CHI1_ATOMS, 0.0),
    ):
        before, after = dihedral(positions, atoms), dihedral(once, atoms)
        shift = abs((after - before + np.pi) % (2.0 * np.pi) - np.pi)
        log(
            f"  {name:5s}: {before:+.6f} -> {after:+.6f} rad, "
            f"|shift| = {shift:.3e}, expected {expected:.4f}"
        )
        if abs(shift - expected) > 1e-6:
            raise ValueError(
                f"{name} shifted by {shift:.6f} rad, expected {expected:.6f}"
            )

    # Both axis atoms must be fixed points.  Atom 6 belongs to the moved block,
    # so this is not implied by the block partition; it holds because atom 6 is
    # the rotation origin.  The whole volume argument rests on it.
    for atom in FLIP_BOND:
        moved = float(np.abs(once[atom] - positions[atom]).max())
        log(f"  axis atom {atom} fixed point: displacement {moved:.3e} nm")
        if moved > 0.0:
            raise ValueError(
                f"axis atom {atom} moved by {moved:.3e} nm; the rotation axis "
                "is not a function of fixed coordinates and the Jacobian "
                "argument fails"
            )

    for name, group in (
        ("moved", block),
        ("fixed", np.setdiff1d(np.arange(len(positions)), block)),
    ):
        before = np.linalg.norm(
            positions[group][:, None, :] - positions[group][None, :, :], axis=-1
        )
        after = np.linalg.norm(
            once[group][:, None, :] - once[group][None, :, :], axis=-1
        )
        drift = float(np.abs(after - before).max())
        log(f"  {name} block rigid: max distance change {drift:.3e} nm")
        if drift > 1e-9:
            raise ValueError(f"{name} block is not rigid: {drift:.3e} nm")


def verify_internal_invariance(
    positions: np.ndarray, block: np.ndarray, system: dict
) -> None:
    """Check that the move changes torsions about the flip bond and nothing else.

    This is the internal-coordinate half of the measure argument.  The Cartesian
    to bond/angle/torsion Jacobian is a product over bond lengths and bond
    angles and never involves a torsion, so if the move leaves every bond and
    every angle invariant it is a pure translation of the torsions about the
    flip bond and has unit Jacobian in that chart too.  Intra-block rigidity does
    not imply this on its own: the bonds and angles that *span* the flip bond are
    the ones at risk, and they are exactly the ones this checks.
    """

    once = flip(positions, block, FLIP_BOND)

    bonds = np.asarray(system["bond_idx"], dtype=np.int64).reshape(-1, 2)
    before = np.linalg.norm(positions[bonds[:, 0]] - positions[bonds[:, 1]], axis=1)
    after = np.linalg.norm(once[bonds[:, 0]] - once[bonds[:, 1]], axis=1)
    drift = float(np.abs(after - before).max())
    log(f"  all {len(bonds)} bond lengths invariant: max change {drift:.3e} nm")
    if drift > 1e-9:
        raise ValueError(f"a bond length changed by {drift:.3e} nm")

    angles = np.asarray(system["angle_idx"], dtype=np.int64).reshape(-1, 3)

    def values(coords: np.ndarray) -> np.ndarray:
        u = coords[angles[:, 0]] - coords[angles[:, 1]]
        v = coords[angles[:, 2]] - coords[angles[:, 1]]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return np.arccos(np.clip(np.sum(u * v, axis=1), -1.0, 1.0))

    drift = float(np.abs(values(once) - values(positions)).max())
    log(f"  all {len(angles)} bond angles invariant: max change {drift:.3e} rad")
    if drift > 1e-7:
        raise ValueError(f"a bond angle changed by {drift:.3e} rad")


def blocking_curve(indicator: np.ndarray) -> list[dict]:
    """Return the standard error of the cis fraction against block length.

    The samples are correlated, so the naive binomial error understates the
    uncertainty.  Cutting the series into equal blocks and taking the spread of
    the block means converges to the correct error once the block is long
    compared with the autocorrelation time.
    """

    curve = []
    for blocks in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        size = indicator.size // blocks
        if size < 50:
            break
        means = np.array(
            [indicator[b * size : (b + 1) * size].mean() for b in range(blocks)]
        )
        curve.append(
            {
                "blocks": blocks,
                "block_samples": int(size),
                "mean": float(means.mean()),
                "sem": float(means.std(ddof=1) / np.sqrt(blocks)),
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=PRODUCTION_SAMPLES)
    parser.add_argument("--discard", type=int, default=DISCARD_SAMPLES)
    parser.add_argument("--timestep-fs", type=float, default=TIMESTEP_FS)
    parser.add_argument("--steps-per-attempt", type=int, default=STEPS_PER_ATTEMPT)
    parser.add_argument("--frames", type=int, default=FRAME_TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard an unfinished run recorded in the manifest and start over",
    )
    args = parser.parse_args()
    if args.samples <= 0 or args.discard < 0:
        parser.error("samples must be positive; discard nonnegative")
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text("utf-8"))
        status = previous.get("status")
        if status == "complete":
            raise SystemExit(
                f"{MANIFEST.name} records a complete run; refusing to overwrite. "
                "Move it aside to start again."
            )
        # A run killed part way leaves status "running".  Restarting silently
        # would overwrite its series at the first checkpoint and lose hours of
        # sampling; this driver has no resume, so the destruction has to be
        # asked for explicitly.
        if status == "running" and not args.restart:
            raise SystemExit(
                f"{MANIFEST.name} records an unfinished run of "
                f"{previous.get('nanoseconds', 0):g} ns "
                f"({previous.get('retained_samples', 0):,} retained samples). "
                "There is no resume: starting again overwrites it. "
                "Pass --restart to discard it, or move it aside to keep it."
            )

    rng = np.random.default_rng(args.seed)
    bundle = Molecular_Bundle.load(BUNDLE)
    bonds = np.asarray(bundle.system["bonds"], dtype=np.int64)
    # An incomplete bond list would orphan atoms into the complement, leaving
    # both blocks internally rigid and the ring check satisfied, with near-zero
    # acceptance as the only symptom.  This molecule has one ring, so a
    # connected graph on n atoms has exactly n bonds.
    if len(bonds) != int(bundle.system["n_atoms"]):
        raise ValueError(
            f"expected {int(bundle.system['n_atoms'])} bonds for a "
            f"single-ring molecule, found {len(bonds)}"
        )

    potential = OpenMM_Potential.from_bundle(BUNDLE)
    system = potential.create_system()
    n_atoms = system.getNumParticles()
    start = np.asarray(potential.reference_positions_nm, dtype=np.float64)
    if start.ndim == 3:
        start = start[0]

    total_samples = args.discard + args.samples
    attempt_ps = args.steps_per_attempt * args.timestep_fs / 1000.0
    total_steps = total_samples * args.steps_per_attempt
    total_ns = total_steps * args.timestep_fs / 1e6

    log(
        f"bundle={potential.bundle_name} atoms={n_atoms} unregularized "
        f"target={TEMPERATURE_KELVIN} K seed={args.seed}"
    )
    log(
        f"real masses, unconstrained bonds | timestep {args.timestep_fs} fs | "
        f"platform={PLATFORM}"
    )
    log(
        f"move: rigid rotation of the proline side of bond {FLIP_BOND} by pi -- "
        "an involution with unit Jacobian, accepted with min(1, exp(-dU/kT)); "
        "the chain samples exp(-U/kT) exactly, unweighted, at the target "
        "temperature"
    )
    log(
        "start: the bundle's own reference geometry; no bias, no tempering, and "
        "no input from any earlier run"
    )

    log("verifying observables and move before sampling:")
    verify_observables(start, n_atoms)
    block = moved_block(bonds, FLIP_BOND)
    verify_move(start, block)
    verify_internal_invariance(start, block, bundle.system)
    log("verified")

    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_KELVIN * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(args.seed)
    # Mixed precision: the two energies that form dU are differenced against
    # kT = 2.494 kJ/mol, and single-precision noise there perturbs acceptance.
    # The effect is tiny, but for 26 atoms the accuracy is nearly free.
    context = mm.Context(
        system,
        integrator,
        mm.Platform.getPlatformByName(PLATFORM),
        {"Precision": "mixed"},
    )
    context.setPositions(start * unit.nanometer)
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=500)
    context.setVelocitiesToTemperature(
        TEMPERATURE_KELVIN * unit.kelvin, args.seed + 1
    )

    masses = np.array(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(n_atoms)
        ],
        dtype=np.float64,
    )
    kt = KB_KJ_MOL_K * TEMPERATURE_KELVIN
    log(
        f"plan: one chain, discard {args.discard:,} then retain {args.samples:,} "
        f"attempts every {args.steps_per_attempt:,} steps ({attempt_ps:g} ps) = "
        f"{total_steps:,} steps, {total_ns:g} ns"
    )
    stride = max(1, args.samples // max(1, args.frames))
    log(
        f"frames: every {stride:,}th retained attempt kept as Cartesian "
        f"coordinates, about {args.samples // stride:,} frames"
    )

    phi_series = np.empty(total_samples, dtype=np.float32)
    psi_series = np.empty(total_samples, dtype=np.float32)
    omega_series = np.empty(total_samples, dtype=np.float32)
    chi1_series = np.empty(total_samples, dtype=np.float32)
    accepted_series = np.zeros(total_samples, dtype=np.int8)
    delta_series = np.empty(total_samples, dtype=np.float32)
    frames = np.empty((args.samples // stride + 1, n_atoms, 3), dtype=np.float32)
    frame_energy = np.empty(frames.shape[0], dtype=np.float64)

    started = time.time()
    n_cis = 0
    n_accept = 0
    n_accept_kept = 0
    kept = 0
    n_frames = 0
    block_cis = 0
    block_accept = 0
    block_size = 0

    def energy_of(positions: np.ndarray) -> float:
        context.setPositions(positions * unit.nanometer)
        return (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )

    for index in range(total_samples):
        integrator.step(args.steps_per_attempt)
        state = context.getState(getPositions=True, getEnergy=True)
        positions = np.asarray(
            state.getPositions().value_in_unit(unit.nanometer), dtype=np.float64
        )
        # Recentre on the centre of mass.  The system is non-periodic with no
        # CMMotionRemover, so the centre of mass performs a free random walk:
        # with D = kT/(M*gamma) = 0.0147 nm^2/ps it would reach a few hundred nm
        # over 500 ns.  Nothing about the physics changes -- the potential and
        # every internal coordinate are translation invariant, so U, dU and all
        # four torsions are untouched, and translation is a symmetry of the
        # target -- but absolute coordinates that large destroy the float32
        # frames written for the landscape comparison and inject rounding noise
        # into the energy difference that decides acceptance.  Recentring keeps
        # the molecule at the origin for the whole run.
        positions -= np.average(positions, axis=0, weights=masses)
        current = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        proposal = flip(positions, block, FLIP_BOND)
        energy = energy_of(proposal)
        delta = energy - current
        # Symmetric, volume-preserving proposal: the Metropolis ratio is the
        # Boltzmann factor of the energy difference and nothing else.  A
        # nonfinite delta is rejected outright: +inf and NaN already fail the
        # test below, but -inf would pass "delta <= 0" and accept a broken
        # geometry, so finiteness is required explicitly rather than relied on.
        if np.isfinite(delta) and (
            delta <= 0.0 or rng.random() < np.exp(-delta / kt)
        ):
            accepted_series[index] = 1
            n_accept += 1
            if index >= args.discard:
                n_accept_kept += 1
            block_accept += 1
            positions = proposal      # the context already holds the proposal
            current = energy
        else:
            context.setPositions(positions * unit.nanometer)

        omega = dihedral(positions, OMEGA_ATOMS)
        phi_series[index] = dihedral(positions, PHI_ATOMS)
        psi_series[index] = dihedral(positions, PSI_ATOMS)
        omega_series[index] = omega
        chi1_series[index] = dihedral(positions, CHI1_ATOMS)
        delta_series[index] = delta
        is_cis = abs(omega) < np.pi / 2.0
        # The chunk accumulators run through the discard too, so every report
        # carries a cis value; only the retained ones feed the estimate.
        block_size += 1
        if is_cis:
            block_cis += 1
        if index >= args.discard:
            kept += 1
            if is_cis:
                n_cis += 1
            if (index - args.discard) % stride == 0 and n_frames < len(frames):
                frames[n_frames] = positions
                frame_energy[n_frames] = current
                n_frames += 1

        if (index + 1) % REPORT_SAMPLES == 0 or index + 1 == total_samples:
            done = index + 1
            elapsed = time.time() - started
            ns_done = done * attempt_ps / 1000.0
            phase = "discard " if done <= args.discard else "retained"
            estimate = n_cis / kept if kept else float("nan")
            # Median over this chunk only: cheap at two million samples, and a
            # drifting chunk median is the readable diagnostic, not a cumulative
            # one that stops moving.
            chunk = delta_series[done - block_size : done]
            finite = chunk[np.isfinite(chunk)]
            log(
                f"{done:>9,}/{total_samples:,} ({100.0 * done / total_samples:5.1f}%) "
                f"{phase} | {ns_done:8.2f} ns | cis={estimate:.4f} "
                f"(chunk {block_cis / block_size:.4f}) | "
                f"jumps {n_accept:,} acc={n_accept / done:.4f} "
                f"(chunk {block_accept / block_size:.4f}) | "
                f"median dU={np.median(finite):7.1f} kJ/mol | "
                f"{ns_done / elapsed * 3600:.1f} ns/h | "
                f"eta {(total_samples - done) / done * elapsed / 3600:.2f} h"
            )
            block_cis = 0
            block_accept = 0
            block_size = 0
            reports_done = done // REPORT_SAMPLES
            if reports_done % CHECKPOINT_REPORTS and done != total_samples:
                continue
            np.savez_compressed(
                SERIES,
                phi_rad=phi_series[:done],
                psi_rad=psi_series[:done],
                omega_rad=omega_series[:done],
                chi1_rad=chi1_series[:done],
                accepted=accepted_series[:done],
                delta_kj_mol=delta_series[:done],
                discard_samples=np.asarray(args.discard),
                attempt_ps=np.asarray(attempt_ps),
            )
            if n_frames:
                np.savez_compressed(
                    FRAMES,
                    positions_nm=frames[:n_frames],
                    energy_kj_mol=frame_energy[:n_frames],
                    stride=np.asarray(stride),
                    attempt_ps=np.asarray(attempt_ps),
                )
            cis_indicator = (
                np.abs(omega_series[args.discard : done]) < np.pi / 2.0
            ).astype(np.float64)
            curve = blocking_curve(cis_indicator) if cis_indicator.size >= 100 else []
            # Quote one error bar rather than leaving the reader to choose.
            # The SEM of a B-block estimate is itself uncertain by about
            # 1/sqrt(2(B-1)) -- 71 per cent at B = 2 -- while small blocks
            # cannot see correlations longer than the block.  B = 32 is the
            # compromise: 13 per cent uncertainty on the error bar, with blocks
            # long enough to expose a slow mode coupled to the basin populations.
            selected = min(
                (row for row in curve if row["blocks"] <= 32),
                key=lambda row: abs(row["blocks"] - 32),
                default=None,
            )
            MANIFEST.write_text(
                json.dumps(
                    {
                        "bundle": potential.bundle_name,
                        "atoms": int(n_atoms),
                        "potential": "unregularized physical",
                        "sampler": (
                            "Langevin at 300 K plus a Metropolis rigid pi "
                            "rotation of the proline side of the ACE-PRO amide "
                            "bond; involution, unit Jacobian, unweighted. The "
                            "jump is exact; the integrator is unadjusted, so "
                            "the only bias is its O(dt^2) discretization error"
                        ),
                        "start": "bundle reference geometry",
                        "temperature_kelvin": TEMPERATURE_KELVIN,
                        "timestep_fs": args.timestep_fs,
                        "steps_per_attempt": args.steps_per_attempt,
                        "attempt_ps": attempt_ps,
                        "phi_atoms": list(PHI_ATOMS),
                        "psi_atoms": list(PSI_ATOMS),
                        "omega_atoms": list(OMEGA_ATOMS),
                        "chi1_atoms": list(CHI1_ATOMS),
                        "flip_bond": list(FLIP_BOND),
                        "moved_block": block.tolist(),
                        "discard_samples": args.discard,
                        "retained_samples": kept,
                        # Both denominators are named: the first counts every
                        # attempt including the discard, the second only the
                        # retained ones that feed cis_fraction.
                        "accepted_jumps_all": int(n_accept),
                        "acceptance_all_attempts": n_accept / done,
                        "accepted_jumps_retained": int(n_accept_kept),
                        "acceptance_retained": (
                            n_accept_kept / kept if kept else None
                        ),
                        # None, not NaN: bare NaN is not valid JSON and only
                        # Python's own parser accepts it.
                        "cis_fraction": (
                            estimate if np.isfinite(estimate) else None
                        ),
                        "cis_standard_error": (
                            selected["sem"] if selected else None
                        ),
                        "cis_standard_error_blocks": (
                            selected["blocks"] if selected else None
                        ),
                        "cis_blocking_curve": curve,
                        "frames": int(n_frames),
                        "frame_stride": int(stride),
                        "nanoseconds": ns_done,
                        "seed": args.seed,
                        "status": "complete" if done == total_samples else "running",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    log(
        f"done: cis={n_cis / kept:.4f} over {kept:,} retained attempts "
        f"({total_ns:g} ns), {n_accept:,} accepted jumps "
        f"({n_accept / total_samples:.4f}), {n_frames:,} frames saved"
    )


if __name__ == "__main__":
    main()
