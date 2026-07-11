"""Generate prmtop + rst7 for diverse HETERO-ATOM molecules via GAFF, for the
non-alkane BG examples (10d-60d). Run in the isolated `molparam` env:

    python zflows_md/data/gen_molecules.py

Writes <name>.prmtop / <name>.rst7 into this `zflows_md/data/` folder; a per-molecule
train.py then trains the BG on them. PREPARATION ONLY — does not run any BG.
"""
import os

# name -> (SMILES, hetero-atoms, approx d=3*natoms-6)
MOLS = {
    "methanol":       ("CO",            "O",    12),   # smallest O
    "chloroethane":   ("CCCl",          "Cl",   18),   # halogen
    "ethanol":        ("CCO",           "O",    21),
    "pyridine":       ("c1ccncc1",      "N",    27),   # aromatic N ring
    "phenol":         ("c1ccccc1O",     "O",    33),   # aromatic OH
    "glycerol":       ("OCC(O)CO",      "O",    36),   # triol
    "morpholine":     ("C1COCCN1",      "N,O",  39),   # N+O ring
    "diethanolamine": ("OCCNCCO",       "N,O",  48),   # N + 2 OH, flexible
    "pentylamine":    ("CCCCCN",        "N",    51),   # amine chain
    "hexylamine":     ("CCCCCCN",       "N",    60),   # high-d amine
    "triethanolamine":("OCCN(CCO)CCO",  "N,O",  69),   # high-d, N + 3 OH
    # (nma — N+O amide, d=30 — already exists as nma.prmtop)
}
OUT = os.path.dirname(os.path.abspath(__file__))             # write into this zflows_md/data/ folder (where the BG data lives)


def main():
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openmm import app
    import parmed
    os.makedirs(OUT, exist_ok=True)
    for name, (smi, het, _) in MOLS.items():
        if os.path.exists(os.path.join(OUT, f"{name}.prmtop")):
            print(f"skip {name} (prmtop exists)"); continue
        try:
            mol = Molecule.from_smiles(smi)
            mol.generate_conformers(n_conformers=1)
            # AM1-BCC charges (the standard) — computed by SMIRNOFF via AmberTools sqm.
            smirnoff = SMIRNOFFTemplateGenerator(molecules=mol, forcefield="openff-2.1.0")
            ff = app.ForceField()
            ff.registerTemplateGenerator(smirnoff.generator)
            top = mol.to_topology().to_openmm()
            system = ff.createSystem(top, nonbondedMethod=app.NoCutoff)
            pos = mol.conformers[0].to_openmm()
            st = parmed.openmm.load_topology(top, system, xyz=pos)
            st.save(os.path.join(OUT, f"{name}.prmtop"), overwrite=True)
            st.save(os.path.join(OUT, f"{name}.rst7"), overwrite=True)
            print(f"OK {name:12s} {het:5s} atoms={len(st.atoms):3d} d={3*len(st.atoms)-6}")
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
