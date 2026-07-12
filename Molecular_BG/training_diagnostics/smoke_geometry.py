#!/usr/bin/env python
"""Cheap deterministic geometry/MOL2 smoke test; no AmberTools invocation."""

from __future__ import annotations

import hashlib

from geometry import build_alkane_geometry, render_mol2
from build_bundles import _graph_equivalence_groups
import parameters as P


def main() -> None:
    expected_groups = ((1, 4), (2, 6), (1, 2, 2, 6), (2, 2, 4, 6))
    for index, carbon_count in enumerate(P.CARBON_COUNTS):
        first = build_alkane_geometry(carbon_count)
        second = build_alkane_geometry(carbon_count)
        assert first.formula == P.FORMULAS[index]
        assert first.atom_count * 3 - 6 == P.DIMENSIONS[index]
        text = render_mol2(first)
        assert text == render_mol2(second)
        assert text.count("@<TRIPOS>ATOM") == 1
        assert text.count("@<TRIPOS>BOND") == 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        groups = tuple(sorted(map(len, _graph_equivalence_groups(first))))
        assert groups == expected_groups[index], (groups, expected_groups[index])
        print(
            f"PASS {P.MOLECULES[index]} atoms={first.atom_count} "
            f"d={P.DIMENSIONS[index]} mol2={digest[:12]}"
        )


if __name__ == "__main__":
    main()
