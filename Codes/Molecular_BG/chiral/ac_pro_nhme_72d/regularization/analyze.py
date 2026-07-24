#!/usr/bin/env python
"""Analyze Ac-Pro-NHMe with the shared chiral regularization protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SHARED_ROOT = HERE.parents[1] / "regularization"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    shared_run = load("run", SHARED_ROOT / "run.py")
    shared_run.ROOT = HERE
    shared_run.CONFIG_PATH = HERE / "config.json"
    shared_analyze = load(
        "_shared_chiral_regularization_analyze", SHARED_ROOT / "analyze.py"
    )
    shared_analyze.main()


if __name__ == "__main__":
    main()
