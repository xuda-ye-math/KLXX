"""Smoke test: the zflows_md package structure is intact after the bg/ dissolution
refactor, and the package is free of hardcoded absolute paths.

Layout invariant: exactly four subfolders (core, plot, data, template); everything
else is a plain top-level .py module. No bg/ submodule.

Run:  PYTHONPATH=<repo> python -m pytest tests/test_structure.py -q
(or simply `pytest tests/` from the repo root; conftest.py sets the path).
"""
import importlib, os, glob
import zflows_md

# top-level plain .py modules (core BG library + analysis helpers)
TOP_MODS = ["boltzmann", "potential", "forcefield", "coords", "flow", "loss", "utils",
            "dihedral", "gate"]
# figure modules under plot/ (bare-noun names)
PLOT_MODS = ["dihedrals", "conformers", "ablation", "summary", "table", "pymol", "marginals"]
# the only four subfolders allowed
FOLDERS = ["core", "plot", "data", "template"]
PKG_DIR = os.path.dirname(os.path.abspath(zflows_md.__file__))


def test_local_package_in_use():
    # the in-development source copy (sibling of this tests/ dir), not an installed site-packages one
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root = parent of tests/
    assert os.path.dirname(os.path.dirname(os.path.abspath(zflows_md.__file__))) == repo, zflows_md.__file__


def test_only_four_subfolders():
    dirs = sorted(d for d in os.listdir(PKG_DIR)
                  if os.path.isdir(os.path.join(PKG_DIR, d)) and d != "__pycache__")
    assert dirs == sorted(FOLDERS), f"expected {sorted(FOLDERS)}, got {dirs}"
    assert not os.path.isdir(os.path.join(PKG_DIR, "bg")), "bg/ should be dissolved"


def test_core_library_surface():
    for name in TOP_MODS:
        assert hasattr(zflows_md, name), f"missing top-level module {name}"
    from zflows_md.boltzmann import build, run_boltzmann, run_asmc          # noqa: F401  (training API)
    from zflows_md.flow import NCSF                                          # noqa: F401


def test_top_level_modules_import():
    for m in TOP_MODS:
        importlib.import_module(f"zflows_md.{m}")


def test_plot_submodule_exists():
    files = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(PKG_DIR, "plot", "*.py"))}
    assert set(PLOT_MODS) <= files, f"plot/ missing modules: {set(PLOT_MODS) - files}"
    assert os.path.exists(os.path.join(PKG_DIR, "plot", "__init__.py"))


def test_plot_modules_import():
    # every plot module imports cleanly, including the guarded pymol (no import-time side effects)
    for m in PLOT_MODS:
        importlib.import_module(f"zflows_md.plot.{m}")


def test_data_assets_present():
    data = os.path.join(PKG_DIR, "data")
    # prmtop + rst7 for every molecule tested (adp, diethanolamine, glycerol)
    for mol in ("alanine_dipeptide", "diethanolamine", "glycerol"):
        assert os.path.exists(os.path.join(data, f"{mol}.prmtop")), f"missing {mol}.prmtop"
        assert os.path.exists(os.path.join(data, f"{mol}.rst7")), f"missing {mol}.rst7"
    assert os.path.exists(os.path.join(data, "gen_molecules.py"))           # data-prep tool lives here


def test_no_hardcoded_absolute_paths():
    bad = []
    for root, dirs, fs in os.walk(PKG_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in fs:
            if f.endswith((".py", ".sh")):
                txt = open(os.path.join(root, f)).read()
                for needle in ("/mnt/projects/", "/home/"):
                    if needle in txt:
                        bad.append(os.path.join(os.path.relpath(root, PKG_DIR), f))
    assert not bad, f"hardcoded absolute paths found in: {sorted(set(bad))}"
