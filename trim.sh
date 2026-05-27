#!/usr/bin/env bash
# Trim white margins from every .png in this folder and its subfolders (in place) using ImageMagick.
set -euo pipefail

cd "$(dirname "$0")"

shopt -s nullglob globstar
for f in ./**/*.png; do
    magick mogrify -bordercolor white -border 1x1 -trim +repage "$f"
    echo "trimmed $f"
done
