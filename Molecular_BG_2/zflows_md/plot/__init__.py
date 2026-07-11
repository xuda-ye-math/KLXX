"""zflows_md.plot — figure generation for the molecular Boltzmann generators,
kept in its own submodule so the core package and training code stay clean.

Modules:
    marginals  : torsion-marginal panels (BG vs annealed SMC reference); also
                 carries the SMC-replay helpers ``short_md`` / ``flow_proper_smc``
                 used to regenerate samples for the plot
    dihedrals  : per-dihedral marginal helper (``fab_marginals``)
    conformers : 3D conformer renders at the gauche/trans states
    summary    : cross-molecule summary tables / figures
    ablation   : ablation-ladder (per-stage ESS arc) plots
    table      : LaTeX/markdown result tables
    pymol      : PyMOL-based molecule renders

These modules import matplotlib (and some import torch/openmm); they are kept out
of ``zflows_md.__init__`` so importing the core package stays lightweight.
"""
