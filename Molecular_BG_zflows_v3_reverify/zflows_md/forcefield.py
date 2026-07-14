# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Differentiable AMBER force fields and explicit regularizers in pure Torch.

`Amber_Force_Field` reads parameters straight from an OpenMM System and reproduces
OpenMM energies (kJ/mol, nm) to numerical tolerance. This is an INTERNAL helper
for `zflows_md.potential.PDB_Potential` (the public, internal-coordinate target) —
never a public Cartesian potential.  The historical e/r regularity
(`r_floor`, `softcap_energy`) and the reference-shifted c regularity
(`c_regularize_energy`) are separate, coexisting choices.
"""
import math
import numpy as np
import torch
from torch import nn
import openmm as mm
import openmm.app as app
import parmed as pmd
from openmm import unit
# Boltzmann constant in kJ/mol/K (OpenMM/CODATA).
KB_KJ = 0.00831446261815324

# OpenMM Coulomb constant 1/(4 pi eps0) in kJ/mol * nm / e^2.
ONE_4PI_EPS0 = 138.9354576

# Soft-core distance floor (nm) for the nonbonded pair term. Physical nonbonded
# contacts are >= ~0.18 nm (all 1-2/1-3 pairs are excluded), so this is INACTIVE
# for real configurations (the OpenMM energy match is preserved to ~1e-8), but it
# replaces the r->0 LJ/Coulomb singularity by a finite plateau when positive.
# r_floor=0 is the exact raw-distance member and retains the collision
# singularity.  A positive floor changes the surrogate inside the plateau and
# must be sharpened/corrected before claiming the raw target.
R_FLOOR = 0.10

# Smooth energy cap for the Boltzmann-generator targets. Physical ADP energies
# are all <~ -10 kJ/mol, so with E_CAP0 = 100 the cap is INACTIVE for real
# configs (Stage-1/2 OpenMM match preserved). Above E_CAP0 the energy is
# log-compressed so the enormous clash wall (random-torsion proposals reach
# ~1e8 kJ/mol) does not dominate the temperature-ladder bridging variance and
# stall the adaptive step at t~0. Clash configs keep ~zero Boltzmann weight, so
# the capped target is indistinguishable from the true one where it has mass.
E_CAP0 = 100.0     # kJ/mol threshold (cap inactive below this)
E_CAP_SCALE = 50.0  # kJ/mol log-compression scale above the threshold

# Reference-shifted c regularization used by the current molecular controls.
# These defaults are the c50 target: identity through 50 kJ/mol above E_ref,
# logarithmic scale 50 kJ/mol, with the established pure-log tail (rho=0).
C_CUT0 = 50.0
C_SCALE = 50.0
C_TAIL_FRACTION = 0.0

GB_COULOMB = 138.935485
GB_OFFSET_NM = 0.009
ACE_COEFFICIENT = 28.3919551


def _validate_r_floor(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("r_floor must be finite and nonnegative")
    return value


def softcap_energy(U: torch.Tensor, e0: float = E_CAP0,
                   scale: float = E_CAP_SCALE) -> torch.Tensor:
    """Identity for U <= e0; smooth (C1) log-compression e0 + scale*log1p((U-e0)/scale)
    above.  Finite inputs remain finite and grow only logarithmically, but this
    map neither repairs an upstream infinity nor removes the residual 1/r
    force singularity of a pure-log-compressed repulsive collision wall."""
    over = (U - e0).clamp_min(0.0)
    return torch.where(U > e0, e0 + scale * torch.log1p(over / scale), U)


def c_regularize_energy(
    U: torch.Tensor,
    reference_energy: torch.Tensor,
    c: float | torch.Tensor = C_CUT0,
    scale: float | torch.Tensor = C_SCALE,
    tail_fraction: float | torch.Tensor = C_TAIL_FRACTION,
) -> torch.Tensor:
    """Reference-shifted C1 lin-log regularization of a Cartesian energy.

    With ``d = U - reference_energy``, the map is the identity for ``d <= c``
    and otherwise returns

    ``E_ref + c + (1-rho) s log(1 + (d-c)/s) + rho (d-c)``.

    Unlike the historical e/r path, this function does not alter pair
    distances and its cutoff is relative to a declared reference geometry.
    """

    c = torch.as_tensor(c, dtype=U.dtype, device=U.device)
    scale = torch.as_tensor(scale, dtype=U.dtype, device=U.device)
    rho = torch.as_tensor(tail_fraction, dtype=U.dtype, device=U.device)
    reference_energy = reference_energy.to(dtype=U.dtype, device=U.device)
    excess = U - reference_energy
    over = (excess - c).clamp_min(0.0)
    log_tail = scale * torch.log1p(over / scale)
    compressed_log = torch.where(
        1.0 - rho == 0.0,
        torch.zeros_like(log_tail),
        (1.0 - rho) * log_tail,
    )
    compressed_linear = torch.where(
        rho == 0.0,
        torch.zeros_like(over),
        rho * over,
    )
    compressed = c + compressed_log + compressed_linear
    return reference_energy + torch.where(excess > c, compressed, excess)


# ──────────────────────────────────────────────────────────────────────
# build the OpenMM system (gas-phase AMBER, no cutoff, no constraints)
# ──────────────────────────────────────────────────────────────────────
def build_system(prmtop: str, crd: str, environment: str = "vacuum"):
    """Create a matched vacuum or OBC1/ACE OpenMM system.

    The historical default remains the ParmEd vacuum construction.  The OBC1
    branch uses OpenMM's Amber reader because it attaches the complete OBC1
    polarization and ACE nonpolar terms; ParmEd omits ACE for this input.
    Both branches retain ``NoCutoff``, no constraints, and no COM remover.
    """

    if environment not in {"vacuum", "implicit"}:
        raise ValueError("environment must be 'vacuum' or 'implicit'")
    struct = pmd.load_file(prmtop, crd)
    if environment == "vacuum":
        system = struct.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            implicitSolvent=None,
            removeCMMotion=False,
        )
    else:
        topology = app.AmberPrmtopFile(prmtop)
        system = topology.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            implicitSolvent=app.OBC1,
            soluteDielectric=1.0,
            solventDielectric=78.5,
            implicitSolventSaltConc=0.0 * unit.molar,
            sasaMethod=app.ACE,
            removeCMMotion=False,
        )
    return struct, system


# ──────────────────────────────────────────────────────────────────────
# Amber_Force_Field — batched differentiable energy on Cartesian coords (nm)
# ──────────────────────────────────────────────────────────────────────
class Amber_Force_Field(nn.Module):
    """Re-implements an OpenMM gas-phase AMBER System as a batched torch module.

    Parameters are read directly out of the OpenMM force objects (Harmonic bond/
    angle, periodic torsion, nonbonded with per-particle LJ + exceptions), so the
    energy is identical to OpenMM by construction (Lorentz-Berthelot combining,
    1-2/1-3 exclusions and 1-4 scaling carried by the NonbondedForce exceptions).

    forward(x): x is [N, M, 3] Cartesian (nm) -> energy [N] (kJ/mol).
    energy_terms(x) returns the per-term breakdown for validation.
    """

    def __init__(self, system: mm.System, dtype: torch.dtype = torch.float64,
                 r_floor: float = R_FLOOR):
        super().__init__()
        r_floor = _validate_r_floor(r_floor)
        self.M = system.getNumParticles()
        # soft-core nonbonded distance floor (nm) as a 0-d BUFFER (not a python float) so an
        # r_floor anneal can fill_() it in place and the COMPILED forward picks up the new value
        # with no retrace -- mirrors the e_cap buffer (set_r_floor; cf. PDB_Potential.set_regularization).
        self.register_buffer("r_floor", torch.tensor(r_floor, dtype=dtype))
        forces = {f.__class__.__name__: f for f in system.getForces()}

        # ---- bonds: 0.5 k (r - r0)^2 ----
        bf = forces["HarmonicBondForce"]
        bidx, br0, bk = [], [], []
        for i in range(bf.getNumBonds()):
            a, b, length, k = bf.getBondParameters(i)
            bidx.append([a, b])
            br0.append(length.value_in_unit(unit.nanometer))
            bk.append(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))

        # ---- angles: 0.5 k (theta - theta0)^2 ----
        af = forces["HarmonicAngleForce"]
        aidx, at0, ak = [], [], []
        for i in range(af.getNumAngles()):
            a, b, c, angle, k = af.getAngleParameters(i)
            aidx.append([a, b, c])
            at0.append(angle.value_in_unit(unit.radian))
            ak.append(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))

        # ---- torsions: k (1 + cos(n phi - phase)) ----
        tf = forces["PeriodicTorsionForce"]
        tidx, tn, tphase, tk = [], [], [], []
        for i in range(tf.getNumTorsions()):
            a, b, c, d, n, phase, k = tf.getTorsionParameters(i)
            tidx.append([a, b, c, d])
            tn.append(float(n))
            tphase.append(phase.value_in_unit(unit.radian))
            tk.append(k.value_in_unit(unit.kilojoule_per_mole))

        # ---- nonbonded: per-particle (charge, sigma, eps) + exceptions ----
        nb = forces["NonbondedForce"]
        q, sig, eps = [], [], []
        for i in range(nb.getNumParticles()):
            c, s, e = nb.getParticleParameters(i)
            q.append(c.value_in_unit(unit.elementary_charge))
            sig.append(s.value_in_unit(unit.nanometer))
            eps.append(e.value_in_unit(unit.kilojoule_per_mole))
        q = np.array(q); sig = np.array(sig); eps = np.array(eps)

        # exceptions override specific pairs (1-2/1-3 zeroed, 1-4 scaled)
        exc = {}
        eidx, eqq, esig, eeps = [], [], [], []
        for i in range(nb.getNumExceptions()):
            a, b, qq, s, e = nb.getExceptionParameters(i)
            key = frozenset((a, b))
            exc[key] = True
            eidx.append([a, b])
            eqq.append(qq.value_in_unit(unit.elementary_charge**2))
            esig.append(s.value_in_unit(unit.nanometer))
            eeps.append(e.value_in_unit(unit.kilojoule_per_mole))

        # normal pairs = all i<j not handled by an exception
        nidx, nqq, nsig, neps = [], [], [], []
        for a in range(self.M):
            for b in range(a + 1, self.M):
                if frozenset((a, b)) in exc:
                    continue
                nidx.append([a, b])
                nqq.append(q[a] * q[b])
                nsig.append(0.5 * (sig[a] + sig[b]))
                neps.append(np.sqrt(eps[a] * eps[b]))

        def L(x):  # long buffer
            return torch.tensor(x, dtype=torch.long)

        def F(x):  # float buffer in the chosen dtype
            return torch.tensor(np.asarray(x, dtype=np.float64), dtype=dtype)

        self.register_buffer("bond_idx", L(bidx))
        self.register_buffer("bond_r0", F(br0))
        self.register_buffer("bond_k", F(bk))
        self.register_buffer("ang_idx", L(aidx))
        self.register_buffer("ang_t0", F(at0))
        self.register_buffer("ang_k", F(ak))
        self.register_buffer("tor_idx", L(tidx))
        self.register_buffer("tor_n", F(tn))
        self.register_buffer("tor_phase", F(tphase))
        self.register_buffer("tor_k", F(tk))
        self.register_buffer("npair", L(nidx) if nidx else L([[0, 0]])[:0])
        self.register_buffer("npair_qq", F(nqq))
        self.register_buffer("npair_sig", F(nsig))
        self.register_buffer("npair_eps", F(neps))
        self.register_buffer("epair", L(eidx))
        self.register_buffer("epair_qq", F(eqq))
        self.register_buffer("epair_sig", F(esig))
        self.register_buffer("epair_eps", F(eeps))

    # -- geometry helpers (batched) --
    @staticmethod
    def _dist(x, idx):
        d = x[:, idx[:, 0]] - x[:, idx[:, 1]]
        return d.norm(dim=-1)

    @staticmethod
    def _angle(x, idx):
        v1 = x[:, idx[:, 0]] - x[:, idx[:, 1]]
        v2 = x[:, idx[:, 2]] - x[:, idx[:, 1]]
        denom = (v1.norm(dim=-1).clamp_min(1e-12) * v2.norm(dim=-1).clamp_min(1e-12))
        cos = (v1 * v2).sum(-1) / denom
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos)

    @staticmethod
    def _dihedral(x, idx):
        # Standard OpenMM convention phi = atan2((n1 x n2) . b2hat, n1 . n2),
        # result in [-pi, pi]. (Earlier (n1 x b2hat).n2 form gave -phi, which only
        # matches OpenMM's k(1+cos(n phi - delta)) for phases delta in {0, pi}.)
        p0 = x[:, idx[:, 0]]; p1 = x[:, idx[:, 1]]
        p2 = x[:, idx[:, 2]]; p3 = x[:, idx[:, 3]]
        b1 = p1 - p0; b2 = p2 - p1; b3 = p3 - p2
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        x_ = (n1 * n2).sum(-1)
        y_ = (torch.cross(n1, n2, dim=-1) * b2n).sum(-1)
        return torch.atan2(y_, x_)

    def _pair_nb(self, x, idx, qq, sig, eps):
        if idx.shape[0] == 0:
            return x.new_zeros(x.shape[0])
        r = (x[:, idx[:, 0]] - x[:, idx[:, 1]]).norm(dim=-1).clamp_min(self.r_floor)
        inv = 1.0 / r
        # Evaluate only nonzero interactions.  In particular, an excluded
        # zero-coefficient pair at an exact collision must contribute exactly
        # zero rather than the undefined floating-point product 0 * inf.
        coul_pair = ONE_4PI_EPS0 * qq * inv
        coul_pair = torch.where(
            (qq != 0).unsqueeze(0), coul_pair, torch.zeros_like(coul_pair)
        )
        coul = coul_pair.sum(-1)
        sr6 = (sig * inv) ** 6
        # sr6 * (sr6 - 1), unlike sr6**2 - sr6, has the correct +inf
        # limit at an exact collision instead of producing inf - inf = NaN.
        lj_pair = 4.0 * eps * sr6 * (sr6 - 1.0)
        lj_pair = torch.where(
            (eps != 0).unsqueeze(0), lj_pair, torch.zeros_like(lj_pair)
        )
        lj = lj_pair.sum(-1)
        total = coul + lj
        # At exact/overflowing collisions the r^-12 repulsion dominates a
        # simultaneous signed r^-1 Coulomb infinity.  Preserve the physical
        # +infinity rather than allowing +inf + -inf to become NaN.
        return torch.where(torch.isposinf(lj), lj, total)

    def set_r_floor(self, r_floor):
        """Set the soft-core nonbonded distance floor in place (fills the 0-d buffer): the compiled
        forward picks up the new value with no retrace, like set_regularization for the e_cap."""
        self.r_floor.fill_(_validate_r_floor(r_floor))

    def energy_terms(self, x: torch.Tensor) -> dict:
        r = self._dist(x, self.bond_idx)
        E_bond = (0.5 * self.bond_k * (r - self.bond_r0) ** 2).sum(-1)
        th = self._angle(x, self.ang_idx)
        E_ang = (0.5 * self.ang_k * (th - self.ang_t0) ** 2).sum(-1)
        phi = self._dihedral(x, self.tor_idx)
        E_tor = (self.tor_k * (1.0 + torch.cos(self.tor_n * phi - self.tor_phase))).sum(-1)
        E_nb = (self._pair_nb(x, self.npair, self.npair_qq, self.npair_sig, self.npair_eps)
                + self._pair_nb(x, self.epair, self.epair_qq, self.epair_sig, self.epair_eps))
        return {
            "bond": E_bond, "angle": E_ang, "torsion": E_tor,
            "nonbonded": E_nb, "total": E_bond + E_ang + E_tor + E_nb,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, M, 3] Cartesian (nm) -> energy [N] (kJ/mol)."""
        return self.energy_terms(x)["total"]


class Amber_OBC_Force_Field(Amber_Force_Field):
    """Historical Amber terms plus the exact OpenMM OBC1/ACE contribution."""

    def __init__(
        self,
        system: mm.System,
        dtype: torch.dtype = torch.float64,
        r_floor: float = R_FLOOR,
    ):
        super().__init__(system, dtype=dtype, r_floor=r_floor)
        forces = {force.__class__.__name__: force for force in system.getForces()}
        try:
            gb = forces["CustomGBForce"]
        except KeyError as exc:
            raise ValueError("OBC force field requires one CustomGBForce") from exc
        names = [
            gb.getPerParticleParameterName(index)
            for index in range(gb.getNumPerParticleParameters())
        ]
        if names != ["charge", "or", "sr"]:
            raise ValueError(f"unsupported OBC per-particle parameters: {names}")
        computed = [
            gb.getComputedValueParameters(index)[1]
            for index in range(gb.getNumComputedValues())
        ]
        energy = [
            gb.getEnergyTermParameters(index)[0]
            for index in range(gb.getNumEnergyTerms())
        ]
        if not any("0.8*psi+2.909125*psi^3" in expression for expression in computed):
            raise ValueError("CustomGBForce is not the expected OBC1 model")
        if not any("28.3919551" in expression for expression in energy):
            raise ValueError("CustomGBForce is missing the ACE nonpolar term")
        parameters = np.asarray(
            [
                list(map(float, gb.getParticleParameters(index)))
                for index in range(gb.getNumParticles())
            ],
            dtype=np.float64,
        )
        if parameters.shape != (self.M, 3):
            raise ValueError("OBC particle table has the wrong shape")
        self.register_buffer("gb_charge", torch.tensor(parameters[:, 0], dtype=dtype))
        self.register_buffer("gb_or", torch.tensor(parameters[:, 1], dtype=dtype))
        self.register_buffer("gb_sr", torch.tensor(parameters[:, 2], dtype=dtype))

    def _gb_energy(self, x: torch.Tensor) -> torch.Tensor:
        difference = x[:, :, None, :] - x[:, None, :, :]
        distance_squared = (difference * difference).sum(dim=-1)
        eye = torch.eye(self.M, dtype=torch.bool, device=x.device).unsqueeze(0)
        integral_distance = torch.sqrt(
            torch.where(eye, torch.ones_like(distance_squared), distance_squared)
        )
        radius_i = self.gb_or[None, :, None]
        scaled_j = self.gb_sr[None, None, :]
        upper = integral_distance + scaled_j
        lower = torch.maximum(radius_i, torch.abs(integral_distance - scaled_j))
        inv_lower, inv_upper = 1.0 / lower, 1.0 / upper
        integral = 0.5 * (
            inv_lower
            - inv_upper
            + 0.25
            * (integral_distance - scaled_j * scaled_j / integral_distance)
            * (inv_upper * inv_upper - inv_lower * inv_lower)
            + 0.5 * torch.log(lower / upper) / integral_distance
        )
        valid = (integral_distance + scaled_j - radius_i >= 0.0) & (~eye)
        born_integral = torch.where(valid, integral, 0.0).sum(dim=2)
        psi = born_integral * self.gb_or[None, :]
        full_radius = self.gb_or + GB_OFFSET_NM
        tanh_argument = 0.8 * psi + 2.909125 * psi**3
        born = 1.0 / (
            1.0 / self.gb_or[None, :]
            - torch.tanh(tanh_argument) / full_radius[None, :]
        )
        born_pair = born[:, :, None] * born[:, None, :]
        f = torch.sqrt(
            distance_squared
            + born_pair * torch.exp(-distance_squared / (4.0 * born_pair))
        )
        charge_pair = self.gb_charge[None, :, None] * self.gb_charge[None, None, :]
        prefactor = -0.5 * GB_COULOMB * (1.0 - 1.0 / 78.5)
        polarization = prefactor * (charge_pair / f).sum(dim=(1, 2))
        ace = ACE_COEFFICIENT * (
            (full_radius[None, :] + 0.14) ** 2
            * (full_radius[None, :] / born) ** 6
        ).sum(dim=1)
        return polarization + ace

    def energy_terms(self, x: torch.Tensor) -> dict:
        terms = super().energy_terms(x)
        gb = self._gb_energy(x)
        terms["gb"] = gb
        terms["total"] = terms["total"] + gb
        return terms
