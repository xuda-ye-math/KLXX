# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Differentiable AMBER force field (Cartesian) + soft regularity, in pure torch.

`Amber_Force_Field` reads parameters straight from an OpenMM System and reproduces
OpenMM energies (kJ/mol, nm) to numerical tolerance. This is an INTERNAL helper
for `zflows_md.potential.PDB_Potential` (the public, internal-coordinate target) —
never a public Cartesian potential. The regularity options (`r_floor` soft-core,
`softcap_energy`) cap the r->0 clash singularity for BG proposal samples.
"""
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

# GB/OBC2 (implicit solvent) constants: 1/(4 pi eps0) exactly as spelled in
# OpenMM's customgbforces energy-term string, and the OBC dielectric offset (nm).
ONE_4PI_EPS0_GB = 138.935485
GB_OFFSET = 0.009

# Soft-core distance floor (nm) for the nonbonded pair term. Physical nonbonded
# contacts are >= ~0.18 nm (all 1-2/1-3 pairs are excluded), so this is INACTIVE
# for real configurations (the OpenMM energy match is preserved to ~1e-8), but it
# caps the r->0 LJ/Coulomb singularity to a large FINITE value for the atom
# clashes that pervade random-torsion (Boltzmann-generator proposal) samples,
# preventing inf/NaN energies that crash multinomial resampling. Clashed configs
# have ~zero Boltzmann weight, so capping them does not bias sampling.
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


def softcap_energy(U: torch.Tensor, e0: float = E_CAP0,
                   scale: float = E_CAP_SCALE) -> torch.Tensor:
    """Identity for U <= e0; smooth (C1) log-compression e0 + scale*log1p((U-e0)/scale)
    above. Bounds the clash wall to a finite, slowly-growing value with a finite
    gradient everywhere (unlike a hard clamp, which zeros the escape force)."""
    over = (U - e0).clamp_min(0.0)
    return torch.where(U > e0, e0 + scale * torch.log1p(over / scale), U)


def regularize_energy(energy: torch.Tensor, energy_cut: float, energy_max: float) -> torch.Tensor:
    """Boltzgen/FAB linlog energy regularization on the REDUCED energy (E/kBT).
    Identity below `energy_cut` (so the physical basin is the TRUE, un-softened energy),
    log-compressed above it, hard-capped at `energy_max`, and non-finite -> energy_max.
    With FAB's defaults (cut=1e8, max=1e20 kT) it is inactive except for pathological
    clashes -- the accurate alternative to the aggressive soft-core softcap/r_floor."""
    energy = torch.where(torch.isfinite(energy), energy, torch.full_like(energy, energy_max))
    over = (energy - energy_cut).clamp_min(0.0)
    energy = torch.where(energy > energy_cut, energy_cut + torch.log1p(over), energy)
    return energy.clamp(max=energy_max)


# ──────────────────────────────────────────────────────────────────────
# build the OpenMM system (gas-phase AMBER, no cutoff, no constraints)
# ──────────────────────────────────────────────────────────────────────
def build_system(prmtop: str, crd: str):
    """Load the AMBER prmtop/crd with ParmEd and create a vacuum OpenMM System
    matching the reference energy. Returns (parmed_structure, openmm_system)."""
    struct = pmd.load_file(prmtop, crd)
    system = struct.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        implicitSolvent=None,
        removeCMMotion=False,   # CMMotionRemover is energy/force-neutral; drop it for clarity
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
        self.M = system.getNumParticles()
        # soft-core nonbonded distance floor (nm) as a 0-d BUFFER (not a python float) so an
        # r_floor anneal can fill_() it in place and the COMPILED forward picks up the new value
        # with no retrace -- mirrors the e_cap buffer (set_r_floor; cf. PDB_Potential.set_regularization).
        self.register_buffer("r_floor", torch.tensor(float(r_floor), dtype=dtype))
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

        # ---- GB/OBC2 implicit solvent (present iff the System carries a GBSAOBCForce) ----
        # Native GBSAOBCForce == Amber igb=5 (OBC2): tanh(psi - 0.8 psi^2 + 4.85 psi^3).
        # Per-particle (charge, FULL radius, screen scale) and the solvent/solute
        # dielectrics + surface tension are read straight off the OpenMM force, so the
        # torch energy reproduces OpenMM by construction (validated in validate_obc.py).
        gbf = forces.get("GBSAOBCForce")
        self.has_gb = gbf is not None
        if self.has_gb:
            eps_solvent = gbf.getSolventDielectric()
            eps_solute = gbf.getSoluteDielectric()
            sa = gbf.getSurfaceAreaEnergy().value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer**2)
            gq, grad, gsc = [], [], []
            for i in range(gbf.getNumParticles()):
                c, rr, sc = gbf.getParticleParameters(i)
                gq.append(c.value_in_unit(unit.elementary_charge))
                grad.append(rr.value_in_unit(unit.nanometer))
                gsc.append(float(sc))
            gq = np.array(gq); grad = np.array(grad); gsc = np.array(gsc)
            gor = grad - GB_OFFSET          # offset radius rho = radius - offset
            self.register_buffer("gb_charge", F(gq))
            self.register_buffer("gb_or", F(gor))
            self.register_buffer("gb_sr", F(gsc * gor))     # scaled offset radius
            self.register_buffer("gb_radius", F(grad))      # full radius = rho + offset
            self.gb_pre = -0.5 * ONE_4PI_EPS0_GB * (1.0 / eps_solute - 1.0 / eps_solvent)
            self.gb_sa_coeff = 4.0 * np.pi * sa             # ACE nonpolar prefactor (kJ/mol/nm^2 * 4pi)

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
        coul = ONE_4PI_EPS0 * qq * inv
        sr6 = (sig * inv) ** 6
        lj = 4.0 * eps * (sr6 * sr6 - sr6)
        return (coul + lj).sum(-1)

    def set_r_floor(self, r_floor):
        """Set the soft-core nonbonded distance floor in place (fills the 0-d buffer): the compiled
        forward picks up the new value with no retrace, like set_regularization for the e_cap."""
        self.r_floor.fill_(float(r_floor))

    def _gb_energy(self, x: torch.Tensor) -> torch.Tensor:
        """OBC2 generalized-Born + ACE nonpolar solvation energy [N] (kJ/mol),
        matching OpenMM's GBSAOBCForce. x: [N, M, 3] (nm). Ported verbatim from
        OpenMM's customgbforces energy-term strings (igb=5): descreening integral
        I_i (over all non-excluded ordered pairs) -> Born radius B_i -> polarization
        double sum + ACE. OBC2's tanh keeps B_i strictly positive and bounded, so
        the term is finite for every r>0 (r clamped at 1e-6, below any bonded pair)."""
        M = self.M
        # pairwise distances via the Gram identity |xi-xj|^2 = |xi|^2+|xj|^2-2 xi.xj,
        # avoiding an [N,M,M,3] difference tensor (keeps GB memory at O(N*M^2)).
        x2 = (x * x).sum(-1)                                     # [N, M]
        r2 = (x2[:, :, None] + x2[:, None, :] - 2.0 * (x @ x.transpose(1, 2))).clamp_min(1e-12)
        r = r2.sqrt().clamp_min(1e-6)                           # [N, M, M]
        or_i = self.gb_or.view(1, M, 1)                          # offset radius of atom i (row)
        sr_j = self.gb_sr.view(1, 1, M)                          # scaled offset radius of atom j (col)
        U = r + sr_j
        D = (r - sr_j).abs()
        L = torch.maximum(or_i.expand_as(r), D)
        invL, invU = 1.0 / L, 1.0 / U
        term = 0.5 * (invL - invU + 0.25 * (r - sr_j * sr_j / r) * (invU * invU - invL * invL)
                      + 0.5 * torch.log(L / U) / r)
        eye = torch.eye(M, dtype=torch.bool, device=x.device).view(1, M, M)
        valid = ((r + sr_j - or_i) >= 0) & (~eye)               # step(...) and drop self (i==j)
        I = torch.where(valid, term, term.new_zeros(())).sum(dim=2)   # [N, M]
        psi = I * self.gb_or.view(1, M)
        radius = self.gb_radius.view(1, M)                      # full radius
        tanharg = psi - 0.8 * psi * psi + 4.85 * psi * psi * psi
        B = 1.0 / (1.0 / self.gb_or.view(1, M) - torch.tanh(tanharg) / radius)   # [N, M] > 0
        # polarization double sum (self term is the diagonal, where f -> B_i)
        BiBj = B[:, :, None] * B[:, None, :]
        f = torch.sqrt(r * r + BiBj * torch.exp(-r * r / (4.0 * BiBj)))
        qq = self.gb_charge.view(1, M, 1) * self.gb_charge.view(1, 1, M)
        E_pol = self.gb_pre * (qq / f).sum(dim=(1, 2))
        E_sa = self.gb_sa_coeff * ((radius + 0.14) ** 2 * (radius / B) ** 6).sum(dim=1)
        return E_pol + E_sa

    def energy_terms(self, x: torch.Tensor) -> dict:
        r = self._dist(x, self.bond_idx)
        E_bond = (0.5 * self.bond_k * (r - self.bond_r0) ** 2).sum(-1)
        th = self._angle(x, self.ang_idx)
        E_ang = (0.5 * self.ang_k * (th - self.ang_t0) ** 2).sum(-1)
        phi = self._dihedral(x, self.tor_idx)
        E_tor = (self.tor_k * (1.0 + torch.cos(self.tor_n * phi - self.tor_phase))).sum(-1)
        E_nb = (self._pair_nb(x, self.npair, self.npair_qq, self.npair_sig, self.npair_eps)
                + self._pair_nb(x, self.epair, self.epair_qq, self.epair_sig, self.epair_eps))
        E_gb = self._gb_energy(x) if self.has_gb else E_bond.new_zeros(x.shape[0])
        return {
            "bond": E_bond, "angle": E_ang, "torsion": E_tor,
            "nonbonded": E_nb, "gb": E_gb,
            "total": E_bond + E_ang + E_tor + E_nb + E_gb,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, M, 3] Cartesian (nm) -> energy [N] (kJ/mol)."""
        return self.energy_terms(x)["total"]


