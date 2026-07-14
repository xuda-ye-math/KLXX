"""Run the complete local zflows_md test suite through pytest.

Run:  python tests/run_all.py
"""

from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def run():
    return pytest.main(["-q", str(HERE)])


if __name__ == "__main__":
    raise SystemExit(run())
