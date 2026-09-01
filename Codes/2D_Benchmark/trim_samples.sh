#!/usr/bin/env bash
# Trim the uniform white margin around every 2D benchmark samples.png.
#
# result.py writes each figure with the matplotlib default padding, which
# leaves a border of solid white on all four sides.  ImageMagick's -trim
# removes exactly that border: it grows the bounding box until it meets a
# pixel differing from the corner colour, so nothing inside the axes is
# touched.  +repage clears the offset -trim would otherwise record.
#
# The files are rewritten in place; they are tracked by git, so a trim can be
# undone with `git checkout -- <path>`.  Running the script twice is harmless:
# the second pass finds no margin left to remove.

set -euo pipefail

BASE=/data/projects/KLXX/Codes/2D_Benchmark
FUZZ=1%          # tolerance for "white", to absorb PNG antialiasing
BORDER=4         # thin margin kept so the axes do not touch the image edge

for target in Himmelblau Two-Moon Three-Well Sparse; do
    img="$BASE/$target/results/samples.png"
    if [[ ! -f "$img" ]]; then
        echo "$target: no samples.png, skipped"
        continue
    fi
    before=$(magick identify -format '%wx%h' "$img")
    magick "$img" -fuzz "$FUZZ" -trim +repage \
        -bordercolor white -border "$BORDER" "$img"
    after=$(magick identify -format '%wx%h' "$img")
    echo "$target: $before -> $after"
done
