# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""BAT internal<->Cartesian transform: automatic z-matrix construction (from bond
connectivity) + NeRF placement, with the exact log|det| Jacobian. Pure torch+numpy.
"""
import torch
from torch import nn
# ──────────────────────────────────────────────────────────────────────
# z-matrix construction (automatic, from bond connectivity)
# ──────────────────────────────────────────────────────────────────────
def build_zmatrix(bonds, n_atoms: int, root: int | None = None):
    """Build a placement order + z-matrix references from the bond graph.

    Returns (order, refs) where:
      order[p] = atom index placed at position p (p = 0..n_atoms-1)
      refs[p]  = (r1, r2, r3) the atoms used as bond / angle / torsion
                 references for order[p]; entries are -1 when not applicable
                 (order[0]: (-1,-1,-1); order[1]: (r1,-1,-1);
                  order[2]: (r1,r2,-1); order[>=3]: (r1,r2,r3)).
    r1 is always an already-placed atom bonded to order[p]; r2 an already-placed
    neighbor of r1; r3 an already-placed neighbor of r2 (falling back to any
    already-placed atom to avoid degeneracy). BFS keeps references local so the
    reconstruction frames are well-conditioned.
    """
    adj = [[] for _ in range(n_atoms)]
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    if root is None:
        root = max(range(n_atoms), key=lambda i: len(adj[i]))  # most-connected atom

    order, placed, refs = [], set(), []
    # BFS placement order from root
    from collections import deque
    q = deque([root])
    seen = {root}
    while q:
        a = q.popleft()
        order.append(a)
        for nb in adj[a]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    assert len(order) == n_atoms, "molecule not connected — z-matrix needs one component"

    pos_of = {a: p for p, a in enumerate(order)}

    def placed_neighbors(atom, exclude):
        return [nb for nb in adj[atom]
                if nb in placed and nb not in exclude]

    for p, a in enumerate(order):
        if p == 0:
            refs.append((-1, -1, -1))
        elif p == 1:
            r1 = next(nb for nb in adj[a] if nb in placed)
            refs.append((r1, -1, -1))
        elif p == 2:
            r1 = next(nb for nb in adj[a] if nb in placed)
            # r2: a placed neighbor of r1 (else any placed atom != a, r1)
            cand = placed_neighbors(r1, {a})
            r2 = cand[0] if cand else next(x for x in placed if x not in (a, r1))
            refs.append((r1, r2, -1))
        else:
            r1 = next(nb for nb in adj[a] if nb in placed)
            cand2 = placed_neighbors(r1, {a})
            r2 = cand2[0] if cand2 else next(x for x in placed if x not in (a, r1))
            cand3 = placed_neighbors(r2, {a, r1})
            if cand3:
                r3 = cand3[0]
            else:
                r3 = next(x for x in placed if x not in (a, r1, r2))
            refs.append((r1, r2, r3))
        placed.add(a)
    return order, refs


# ──────────────────────────────────────────────────────────────────────
# Internal_Coordinates — differentiable BAT transform with analytic log-det
# ──────────────────────────────────────────────────────────────────────
class Internal_Coordinates(nn.Module):
    """Bond-angle-torsion (z-matrix) transform between Cartesian coordinates in
    a fixed root frame and internal coordinates, fully differentiable.

    Internal vector layout (dimension 3*M - 6):
        z = [ bonds (M-1) | angles (M-2) | torsions (M-3) ]
    where bonds[k] is the bond for order[k+1], angles[k] for order[k+2],
    torsions[k] for order[k+3] (placement order from build_zmatrix).

    The 6 global DOF are removed by the canonical frame:
        order[0] -> origin; order[1] -> (b, 0, 0); order[2] -> xy-plane (y>0).

    to_internal(x)  : x [N, M, 3] (nm) -> (z [N, 3M-6], logdet_ic_from_xyz [N])
    to_cartesian(z) : z [N, 3M-6]      -> (x [N, M, 3], logdet_xyz_from_ic [N])
    where logdet_xyz_from_ic = log|det d(xyz_free)/dz| = log b2 + sum_{i>=3}
    (2 log d_i + log sin a_i), and logdet_ic_from_xyz = -logdet_xyz_from_ic.
    """
    EPS = 1e-7

    def __init__(self, bonds, n_atoms: int, root: int | None = None):
        super().__init__()
        order, refs = build_zmatrix(bonds, n_atoms, root)
        self.M = n_atoms
        self.order = order
        self.refs = refs
        self.n_internal = 3 * n_atoms - 6
        # index tensors (in placement-position space) for vectorized ops
        self.register_buffer("order_t", torch.tensor(order, dtype=torch.long))
        # bond refs for positions 1..M-1 ; angle refs for 2..M-1 ; tors refs 3..M-1
        b_a, b_r1 = [], []
        for p in range(1, n_atoms):
            b_a.append(order[p]); b_r1.append(refs[p][0])
        an_a, an_r1, an_r2 = [], [], []
        for p in range(2, n_atoms):
            an_a.append(order[p]); an_r1.append(refs[p][0]); an_r2.append(refs[p][1])
        t_a, t_r1, t_r2, t_r3 = [], [], [], []
        for p in range(3, n_atoms):
            t_a.append(order[p]); t_r1.append(refs[p][0])
            t_r2.append(refs[p][1]); t_r3.append(refs[p][2])
        self.register_buffer("bond_a", torch.tensor(b_a, dtype=torch.long))
        self.register_buffer("bond_r1", torch.tensor(b_r1, dtype=torch.long))
        self.register_buffer("ang_a", torch.tensor(an_a, dtype=torch.long))
        self.register_buffer("ang_r1", torch.tensor(an_r1, dtype=torch.long))
        self.register_buffer("ang_r2", torch.tensor(an_r2, dtype=torch.long))
        self.register_buffer("tor_a", torch.tensor(t_a, dtype=torch.long))
        self.register_buffer("tor_r1", torch.tensor(t_r1, dtype=torch.long))
        self.register_buffer("tor_r2", torch.tensor(t_r2, dtype=torch.long))
        self.register_buffer("tor_r3", torch.tensor(t_r3, dtype=torch.long))

    # ---- forward: cartesian -> internal ----
    def to_internal(self, x: torch.Tensor):
        eps = self.EPS
        # bonds (M-1)
        bd = (x[:, self.bond_a] - x[:, self.bond_r1]).norm(dim=-1).clamp_min(eps)
        # angles (M-2): angle at r1 between (r2->? ) actually ∠(r2, r1, a)
        v1 = x[:, self.ang_r2] - x[:, self.ang_r1]
        v2 = x[:, self.ang_a] - x[:, self.ang_r1]
        cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1).clamp_min(eps) * v2.norm(dim=-1).clamp_min(eps))
        ang = torch.acos(cos.clamp(-1 + eps, 1 - eps))
        # torsions (M-3): dihedral(r3, r2, r1, a)
        tor = self._dihedral4(x[:, self.tor_r3], x[:, self.tor_r2],
                              x[:, self.tor_r1], x[:, self.tor_a])
        z = torch.cat([bd, ang, tor], dim=-1)
        logdet = self._logdet_xyz_from_ic(bd, ang)
        return z, -logdet  # ic-from-xyz logdet

    @staticmethod
    def _dihedral4(A, B, C, D):
        b1 = B - A; b2 = C - B; b3 = D - C
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        m1 = torch.cross(n1, b2n, dim=-1)
        return torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    def _split(self, z):
        M = self.M
        bd = z[:, : M - 1]
        ang = z[:, M - 1: (M - 1) + (M - 2)]
        tor = z[:, (M - 1) + (M - 2):]
        return bd, ang, tor

    def _logdet_xyz_from_ic(self, bd, ang):
        # b2 is the bond for order[2] => bd index 1 (order positions: bd[k]=order[k+1])
        eps = self.EPS
        logb2 = bd[:, 1].clamp_min(eps).log()
        # atoms order[3..]: bond index k>=2 in bd, angle index k>=1 in ang
        d_full = bd[:, 2:].clamp_min(eps)            # d_i for i>=3
        a_full = ang[:, 1:]                          # a_i for i>=3
        sin_full = a_full.sin().abs().clamp_min(eps)
        return logb2 + (2.0 * d_full.log() + sin_full.log()).sum(-1)

    # ---- inverse: internal -> cartesian (NeRF) ----
    def to_cartesian(self, z: torch.Tensor):
        eps = self.EPS
        N = z.shape[0]
        bd, ang, tor = self._split(z)
        dev, dt = z.device, z.dtype
        pos = [None] * self.M  # positions indexed by ATOM index
        o = self.order
        # order[0] at origin
        pos[o[0]] = z.new_zeros(N, 3)
        # order[1] at (b1, 0, 0)
        b1 = bd[:, 0]
        e = z.new_zeros(N, 3); e[:, 0] = b1
        pos[o[1]] = pos[o[0]] + e
        # order[2] in xy-plane: angle a2 = ∠(order[0]=r2? ) — refs[2]=(r1,r2)
        r1_2, r2_2 = self.refs[2][0], self.refs[2][1]
        b2 = bd[:, 1]; a2 = ang[:, 0]
        # direction from r1 toward r2 (in current frame), then rotate by a2 in plane
        u = pos[r2_2] - pos[r1_2]
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(eps)
        # perpendicular in xy-plane: rotate u by +90deg about z
        perp = torch.stack([-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])], dim=-1)
        d2 = pos[r1_2] + b2.unsqueeze(-1) * (torch.cos(a2).unsqueeze(-1) * u
                                             + torch.sin(a2).unsqueeze(-1) * perp)
        pos[o[2]] = d2
        # order[>=3] via NeRF
        for k in range(3, self.M):
            a = o[k]; r1, r2, r3 = self.refs[k]
            d = bd[:, k - 1]; an = ang[:, k - 2]; to = tor[:, k - 3]
            C = pos[r1]; B = pos[r2]; A = pos[r3]
            bc = C - B
            bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(eps)
            nrm = torch.cross(B - A, bc, dim=-1)
            nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(eps)
            m = torch.cross(nrm, bc, dim=-1)
            # sign of the torsion (nrm) term matches the atan2 dihedral
            # convention used in to_internal (_dihedral4); see tests/test_bat.py
            d2v = (-(d * torch.cos(an)).unsqueeze(-1) * bc
                   + (d * torch.sin(an) * torch.cos(to)).unsqueeze(-1) * m
                   - (d * torch.sin(an) * torch.sin(to)).unsqueeze(-1) * nrm)
            pos[a] = C + d2v
        x = torch.stack([pos[i] for i in range(self.M)], dim=1)  # [N, M, 3]
        logdet = self._logdet_xyz_from_ic(bd, ang)
        return x, logdet

    def align_to_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Rigid-body map x into the canonical root frame (order[0] origin,
        order[1] on +x, order[2] in xy-plane y>0). Energy is frame-invariant;
        this is only needed to compare a raw Cartesian frame against to_cartesian."""
        z, _ = self.to_internal(x)
        xr, _ = self.to_cartesian(z)
        return xr


