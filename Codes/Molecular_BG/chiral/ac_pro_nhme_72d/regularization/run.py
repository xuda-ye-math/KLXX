#!/usr/bin/env python
"""Run the shared chiral regularization protocol for Ac-Pro-NHMe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "regularization" / "run.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "_shared_chiral_regularization_run", SHARED
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared runner: {SHARED}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = HERE
    module.CONFIG_PATH = HERE / "config.json"
    module.main()


if __name__ == "__main__":
    main()
