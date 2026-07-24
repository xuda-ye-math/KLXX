#!/usr/bin/env python
"""Build the 26-atom ACE--L-PRO--NME ff96/OBC1 runtime bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from jflows_md.bundle_build.builder import write_bundle


HERE = Path(__file__).resolve().parent
ENV_PREFIX = Path(sys.executable).parent.parent
TLEAP = ENV_PREFIX / "bin" / "tleap"
FF96_LEAPRC = ENV_PREFIX / "dat" / "leap" / "cmd" / "oldff" / "leaprc.ff96"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "bundle")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not TLEAP.is_file() or not FF96_LEAPRC.is_file():
        raise FileNotFoundError(
            f"AmberTools ff96 inputs not found: tleap={TLEAP}, leaprc={FF96_LEAPRC}"
        )

    with tempfile.TemporaryDirectory(prefix="ac_pro_nhme_ff96_") as directory:
        work = Path(directory)
        leap_input = work / "leap.in"
        leap_input.write_text(
            "\n".join(
                (
                    f"source {FF96_LEAPRC}",
                    "mol = sequence { ACE PRO NME }",
                    "check mol",
                    "saveamberparm mol ac_pro_nhme.prmtop ac_pro_nhme.rst7",
                    "savepdb mol ac_pro_nhme.pdb",
                    "quit",
                    "",
                )
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [str(TLEAP), "-f", str(leap_input)],
            cwd=work,
            check=True,
        )

        write_bundle(
            output,
            name="ac_pro_nhme_ff96_obc1",
            target="ac_pro_nhme",
            prmtop_path=work / "ac_pro_nhme.prmtop",
            coordinate_path=work / "ac_pro_nhme.rst7",
            model={
                "ambertools_version": "26.0.0",
                "force_field": "Amber ff96",
                "implicit_solvent": "OBC1 (igb=2)",
                "radii": "mbondi2",
                "sasa": "ACE",
                "solute_dielectric": 1.0,
                "solvent_dielectric": 78.5,
                "salt_molar": 0.0,
                "nonbonded_method": "NoCutoff",
                "constraints": None,
                "parameter_provenance": "ff96 ACE/PRO/NME residue templates",
                "structure_source": "AmberTools tleap sequence { ACE PRO NME }",
            },
            canonical_smiles="CC(=O)N1CCC[C@H]1C(=O)NC",
            expected_formula="C8H14N2O2",
            expected_charge=0,
            minimize=False,
            zmatrix={
                "root": 6,
                "prefix": (6, 16, 18, 13),
                "overrides": {
                    16: (6, -1, -1),
                    18: (16, 6, -1),
                    13: (16, 18, 6),
                },
            },
            fixed_stereocenters=(
                {
                    "label": "proline_ca_L",
                    "torsion_index": 0,
                    "atoms": (16, 6, 18, 13),
                    "configuration": "S",
                    "cip_priority_atoms": (6, 18, 13, 17),
                },
            ),
        )
    print(f"bundle ready: {output}")


if __name__ == "__main__":
    main()
