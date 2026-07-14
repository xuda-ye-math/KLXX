"""zflows_md — a self-contained BRANCH of zflows for molecular Boltzmann generators
on whitened internal coordinates (pure PyTorch). It VENDORS the zflows flow/loss/
potential/utils machinery directly (core/, flow.py, loss.py, potential.py, utils.py)
and adds the MD layer on top, so the package depends on nothing outside itself.

Public surface:
    zflows_md.potential  : Potential, linear_combination (vendored) +
                           PDB_Potential, Torsion_Target, Source (MD targets)
    zflows_md.flow       : NCSF, ... (vendored normalizing flows)
    zflows_md.loss       : loss_compile, forward_KL, ... (vendored)
    zflows_md.utils      : langevin, lbfgs, resample, ... (vendored) + pdb/prmtop helpers
    zflows_md.forcefield : Amber_Force_Field   (internal Cartesian FF helper)
    zflows_md.coords     : Internal_Coordinates (BAT internal<->Cartesian)
    zflows_md.boltzmann  : build, run_boltzmann (assemble + train the BG ladder)
    zflows_md.dihedral   : proper_torsions, dihedral, multimodality (torsion analysis)
    zflows_md.gate       : build_mode_checker (BG-training mode-coverage gate)
"""
from . import flow, loss, potential, utils, forcefield, coords, boltzmann, dihedral, gate
from .potential import PDB_Potential, Torsion_Target, Source, Potential, linear_combination
from .flow import NCSF
from .forcefield import Amber_Force_Field
from .coords import Internal_Coordinates
from .boltzmann import build, run_boltzmann

__all__ = ["flow", "loss", "potential", "utils", "forcefield", "coords", "boltzmann", "dihedral", "gate",
           "PDB_Potential", "Torsion_Target", "Source", "Potential", "linear_combination",
           "NCSF", "Amber_Force_Field", "Internal_Coordinates", "build", "run_boltzmann"]
