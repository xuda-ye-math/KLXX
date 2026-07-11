"""Molecular companions for the local :mod:`jflows` package.

The package keeps import-time dependencies light so bundle construction can run
in an OpenMM/AmberTools environment while training runs in a JAX environment.
Public JAX objects are imported lazily through ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "Mixed_Identity",
    "Molecular_Bundle",
    "Molecular_Potential",
    "Molecular_Source",
    "available_bundles",
    "mixed_mala",
    "potential_space_smc",
]


_EXPORTS = {
    "Mixed_Identity": (".flow", "Mixed_Identity"),
    "Molecular_Bundle": (".system", "Molecular_Bundle"),
    "Molecular_Potential": (".potential", "Molecular_Potential"),
    "Molecular_Source": (".source", "Molecular_Source"),
    "available_bundles": (".system", "available_bundles"),
    "mixed_mala": (".mcmc", "mixed_mala"),
    "potential_space_smc": (".smc", "potential_space_smc"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
