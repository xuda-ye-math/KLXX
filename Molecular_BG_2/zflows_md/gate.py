#!/usr/bin/env python
"""Mode-coverage gate for the staged Boltzmann generator.

A stage can clear the ESS gate yet have the FLOW silently drop a marginal-dihedral mode (e.g. the
omega peptide bond's dominant trans / theta=0 basin): ESS measures IS-weight variance AMONG the
surviving particles, so it is blind to a whole mode that is simply absent. This module builds an
INDEPENDENT mode template ONCE -- a uniform-torsion Langevin at the SOFT cap (torsion modes are
cap-independent, so the easy soft cap discovers every well) -- detects the major modes of the
heavy-atom multimodal proper dihedrals, and returns a checker that, given the flow's DIRECT
pushforward, fails the stage when any major template mode is under-covered. run_boltzmann then
shrinks t_k (the same path as an ESS reject). Default-off: drivers that never build a checker are
byte-identical to before.
"""
import numpy as np
import torch
from zflows_md.dihedral import proper_torsions, dihedral, multimodality
from zflows_md.utils import langevin


def _sym_hist(deg, n_bins):
    """+-phi symmetrized periodic marginal histogram (achiral target), normalized to sum 1."""
    a = np.asarray(deg, float).ravel(); a = a[np.isfinite(a)]
    a = np.concatenate([a, -a])                                   # achiral: p(phi)=p(-phi)
    a = (a + 180.0) % 360.0 - 180.0
    h, _ = np.histogram(a, bins=n_bins, range=(-180.0, 180.0))
    return h / max(h.sum(), 1)


def _regions(h, n_bins, floor_mult, p_min):
    """Major modes = contiguous (periodic) runs of bins above floor_mult x uniform, with region
    population >= p_min. Returns a list of boolean bin-masks (length n_bins)."""
    floor = floor_mult / n_bins
    occ = h > floor
    if occ.all():
        return [np.ones(n_bins, bool)]                            # one broad mode (no empty gap)
    z = int(np.where(~occ)[0][0])                                 # rotate to an empty bin -> no wrap split
    occ_r = np.roll(occ, -z)
    masks, i = [], 0
    while i < n_bins:
        if occ_r[i]:
            j = i
            while j < n_bins and occ_r[j]:
                j += 1
            m_r = np.zeros(n_bins, bool); m_r[i:j] = True
            masks.append(np.roll(m_r, z))
            i = j
        else:
            i += 1
    return [m for m in masks if h[m].sum() >= p_min]


def build_mode_checker(B, u, u0, wrap, *, set_soft_cap, mc_step, prm, device,
                       n_walk=50000, iters=4000, top=6, n_bins=36, floor_mult=0.5,
                       p_min=0.10, tau_cov=0.25, check_n=20000, status=print):
    """Build the INDEPENDENT mode template ONCE; return mode_check_fn(y_internal)->bool.

    set_soft_cap(): callback that puts u at the SOFT cap (e_min / r_max) for template sampling --
    run_boltzmann re-sets the regularization per stage afterwards, so this leaves no lasting state.
    The returned closure reconstructs dihedrals the figure's way (template-mean bonds/angles + the
    candidate's raw torsions) so flow and template are compared on torsions alone.
    """
    set_soft_cap()                                                # modes are cap-independent: sample at the EASY soft cap
    x = u0.samples(n_walk).to(device)
    x = wrap(langevin(x, u, step=mc_step, iters=iters, taming=0.0))
    ic, ts = B["ic"], B["tor_start"]; M = ic.M
    mu_b = x[:, :M - 1].mean(0, keepdim=True)                     # template mean bonds/angles (whitening-robust, like the figure)
    mu_a = x[:, M - 1:ts].mean(0, keepdim=True)
    proper = proper_torsions(prm); quads = [t[0] for t in proper]; names = ["-".join(t[2]) for t in proper]

    def _dih(y):                                                  # internal (n,d) -> named proper dihedrals (deg), (n, n_quads)
        tors = y[:, ts:]
        z = torch.cat([mu_b.expand(tors.shape[0], -1), mu_a.expand(tors.shape[0], -1), tors], dim=1)
        cart = ic.to_cartesian(z)[0].detach().cpu().numpy()
        return np.stack([dihedral(cart, q) for q in quads], axis=-1)

    tmpl = _dih(x)
    def canon(p): s = p.split("-"); return min(p, "-".join(s[::-1]))      # dedup equivalent torsions
    def heavy(p): s = p.split("-"); return s[0] != "H" and s[-1] != "H"   # heavy-atom backbone first
    seen = {}
    for i in range(len(names)):
        bal = multimodality(np.concatenate([tmpl[:, i], -tmpl[:, i]]))[0]
        if bal < 0.3:
            continue
        c = canon(names[i])
        if c not in seen or bal > seen[c][1]:
            seen[c] = (i, bal)
    keep = [i for i, _ in sorted(seen.values(), key=lambda v: (not heavy(names[v[0]]), -v[1]))][:top]
    tmpl_hist = {i: _sym_hist(tmpl[:, i], n_bins) for i in keep}
    modes = {i: _regions(tmpl_hist[i], n_bins, floor_mult, p_min) for i in keep}
    status(f"[mode-gate] template {n_walk}x{iters} Langevin @soft cap -> gating "
           f"{[(names[i], len(modes[i])) for i in keep]}  (tau_cov={tau_cov}, p_min={p_min})")

    def mode_check_fn(y):
        """y = samples to gate (internal coords) -- the candidate's POST-SHARPEN deliverable subsample.
        True iff every gated dihedral's major template modes are covered (pop >= tau_cov x template pop)."""
        if not keep:
            return True
        if y.shape[0] > check_n:
            y = y[torch.randint(0, y.shape[0], (check_n,), device=y.device)]
        fd = _dih(y)
        ok = True
        for i in keep:
            hf = _sym_hist(fd[:, i], n_bins)
            for mask in modes[i]:
                fpop, tpop = float(hf[mask].sum()), float(tmpl_hist[i][mask].sum())
                if fpop < tau_cov * tpop:
                    status(f"[mode-gate] {names[i]} mode UNCOVERED: flow {fpop:.3f} "
                           f"< {tau_cov} x template {tpop:.3f}")
                    ok = False
        return ok

    return mode_check_fn
