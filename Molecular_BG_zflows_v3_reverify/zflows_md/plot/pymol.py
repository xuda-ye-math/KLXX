"""Ray-trace the three glycerol rotamer PDBs (written by conformers.py) into transparent
ball-and-stick PNGs with a single shared camera, so the panels are directly comparable.
Run in the molviz env:  conda run -n molviz pymol -cq zflows_md/plot/pymol.py
"""
import os

try:
    from pymol import cmd
except ModuleNotFoundError:  # optional renderer; ordinary package imports stay usable
    cmd = None

# molecule data root resolved from this file (no hardcoded absolute path)
F = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "Molecular_BG", "glycerol_36d")
STATES = ["gauche_minus", "trans", "gauche_plus"]


def main():
    if cmd is None:
        raise RuntimeError(
            "PyMOL is required only for rendering; run this script in a PyMOL environment"
        )
    cmd.set("ray_opaque_background", 0)        # transparent -> blends onto the montage panel
    cmd.set("orthoscopic", 1)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.42)
    cmd.set("specular", 0.30)
    cmd.set("shininess", 55)
    cmd.set("ray_shadows", 1)
    cmd.set("depth_cue", 0)
    cmd.set("sphere_scale", 0.28)              # ball-and-stick: balls subordinate-but-clear
    cmd.set("stick_radius", 0.13)
    cmd.bg_color("white")

    for st in STATES:                          # all loaded -> orient fits the common frame
        cmd.load(os.path.join(F, f"conformer_{st}.pdb"), st)
    cmd.hide("everything")
    cmd.show("sticks"); cmd.show("spheres")
    cmd.color("grey65", "elem C")
    cmd.color("firebrick", "elem O")
    cmd.color("grey80", "elem H")
    cmd.orient()                               # fit all three (they are pre-aligned on the C-C bond)
    cmd.turn("x", -18); cmd.turn("y", 18)      # oblique tilt so the gauche+/- depth difference reads
    cmd.zoom("all", 0.7)
    view = cmd.get_view()

    for st in STATES:
        cmd.disable("all"); cmd.enable(st)
        cmd.set_view(view)
        cmd.ray(1500, 1500)
        out = os.path.join(F, f"pymol_{st}.png")
        cmd.png(out, dpi=400)
        print("rendered", out)


if __name__ == "__main__":                      # pymol -cq runs this as __main__; importing is now side-effect-free
    main()
