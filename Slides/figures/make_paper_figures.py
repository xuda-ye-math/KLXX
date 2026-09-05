"""Re-render the paper's figures at slide font sizes, into Slides/figures/.

The paper's renderers under ``Codes/`` are executed unchanged, so the colours,
contours, panel order and labels stay exactly as the project draws them. Two
things are patched for the duration of the call:

* the figure canvas is shrunk by ``SCALE``, which enlarges every font relative
  to the canvas, so the text survives projection;
* ``Figure.savefig`` is redirected into ``Slides/figures/``, so nothing under
  ``Codes/`` or ``Paper/`` is written.

Run:
    /home/xuda/.envs/jflows/bin/python \
        /data/projects/KLXX/Slides/figures/make_paper_figures.py
"""

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
CODES = Path("/data/projects/KLXX/Codes")

# The canvas keeps the size the project's renderer chose, so its layout is
# unchanged; only the fonts are raised, which is what projection needs.
SCALE = 1.0
SLIDE_FONTS = {
    "font.size": 17, "axes.labelsize": 18, "axes.titlesize": 18,
    "legend.fontsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
}

# source renderer, the basename it writes, and the slide copy to produce
# The two HD figures are ported directly from Paper/figures at the author's
# instruction, so they are not regenerated here.
JOBS = [
    (CODES / "Achiral/plot_dihedrals.py", "dihedrals.png", "achiral_dihedrals.png"),
]


def run(script: Path, produced: str, out_name: str) -> None:
    """Execute one renderer with a shrunken canvas and a redirected savefig."""
    target = HERE / out_name
    real_savefig = Figure.savefig
    real_subplots = plt.subplots
    real_figure = plt.figure
    written: list[Path] = []

    def savefig(self, fname, *args, **kwargs):
        name = Path(str(fname)).name
        dest = target if name == produced else HERE / name
        if name.startswith("."):          # atomic-write temp file
            dest = HERE / name
        kwargs.setdefault("bbox_inches", "tight")
        written.append(dest)
        return real_savefig(self, dest, *args, **kwargs)

    def scaled(figsize):
        if figsize is None:
            return None
        return (figsize[0] * SCALE, figsize[1] * SCALE)

    def subplots(*args, **kwargs):
        if "figsize" in kwargs:
            kwargs["figsize"] = scaled(kwargs["figsize"])
        return real_subplots(*args, **kwargs)

    def figure(*args, **kwargs):
        if "figsize" in kwargs:
            kwargs["figsize"] = scaled(kwargs["figsize"])
        return real_figure(*args, **kwargs)

    # some renderers write atomically: savefig to a temp name, then Path.replace.
    # redirect that move as well, so nothing lands under Codes/
    real_replace = Path.replace

    def replace(self, target):
        # savefig has already redirected the temp file into HERE; follow it
        source = self if self.exists() else HERE / self.name
        name = Path(str(target)).name
        dest = HERE / (out_name if name == produced else name)
        return real_replace(source, dest)

    # the slide's frame title already says what the figure is, and the panel
    # headers carry dimensions that the molecules table has just given
    from matplotlib.axes import Axes
    real_set_title = Axes.set_title
    real_suptitle = Figure.suptitle

    def set_title(self, label, *a, **k):
        label = re.sub(r"\s*\(\$?d\s*=[^)]*\)", "", str(label))
        label = label.replace("neutral ", "")
        for dash in ("\u2014", "--", "\u2013"):
            label = label.replace(f" {dash} ", ", ")
        return real_set_title(self, label, *a, **k)

    def suptitle(self, *a, **k):
        return None

    real_rc_update = plt.rcParams.update

    def rc_update(*args, **kwargs):
        real_rc_update(*args, **kwargs)
        real_rc_update(SLIDE_FONTS)

    Figure.savefig, plt.subplots, plt.figure = savefig, subplots, figure
    Axes.set_title, Figure.suptitle = set_title, suptitle
    Path.replace = replace
    plt.rcParams.update = rc_update
    real_rc_update(SLIDE_FONTS)
    try:
        spec = importlib.util.spec_from_file_location(
            f"paperfig_{script.parent.name}_{script.stem}", script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            module.main()
    finally:
        Figure.savefig, plt.subplots, plt.figure = (
            real_savefig, real_subplots, real_figure)
        Path.replace = real_replace
        Axes.set_title, Figure.suptitle = real_set_title, real_suptitle
        plt.rcParams.update = real_rc_update
        plt.close("all")
    print(f"{script.parent.name}/{script.name} -> "
          f"{', '.join(p.name for p in written)}", flush=True)


def main() -> None:
    for script, produced, out_name in JOBS:
        if not script.exists():
            print(f"missing {script}, skipped", flush=True)
            continue
        run(script, produced, out_name)


if __name__ == "__main__":
    main()
