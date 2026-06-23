"""Run every zflows_md smoke test without requiring pytest (style: ../zflows_md/tests/run_all.py).

Discovers test_*.py here, runs each `test_*` function, reports pass/fail, exits non-zero on
any failure. Puts the LOCAL in-development zflows_md first on sys.path.

Run:  python zflows_md_tests/run_all.py
"""
import os, sys, glob, importlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root -> the local zflows_md/
sys.path.insert(0, HERE)                     # so test_*.py are importable by name


def run():
    total = fails = 0
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        mod = importlib.import_module(os.path.basename(path)[:-3])
        for name in sorted(dir(mod)):
            fn = getattr(mod, name)
            if name.startswith("test_") and callable(fn):
                total += 1
                try:
                    fn()
                    print(f"  PASS  {mod.__name__}.{name}")
                except Exception as e:
                    fails += 1
                    print(f"  FAIL  {mod.__name__}.{name}: {e}")
                    traceback.print_exc()
    print(f"\n{total - fails}/{total} smoke tests passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
