#!/usr/bin/env bash
# Trim the whitespace margins from every figure that Paper/main.tex includes
# and write the trimmed copy into this folder (Paper/figures/), mirroring the
# source layout in subfolders, so main.tex can \includegraphics them directly.
#
# Source figures stay untouched (read-only); only Paper/figures/ is written.
# Requires ImageMagick ('magick' on v7, 'convert' on v6).
#
#   bash Paper/figures/trim.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # Paper/figures
ROOT="$(cd "$HERE/../.." && pwd)"                       # repo root
FUZZ="1%"                                               # tolerance for anti-aliased borders

if command -v magick >/dev/null 2>&1; then IM=(magick)
elif command -v convert >/dev/null 2>&1; then IM=(convert)
else echo "ERROR: ImageMagick not found (need 'magick' or 'convert' on PATH)"; exit 1; fi

ts() { date '+%F %T'; }

# Every figure included by main.tex, as a path relative to the repo root.
# The destination mirrors this path under Paper/figures/, with the redundant
# intermediate 'figures/' component dropped (e.g. Phi4_Lattice_8/figures/x.png
# -> Phi4_Lattice_8/x.png).
SRCS=(
  # 5.1 -- 2D benchmark (samples + importance-resampled, six targets)
  "2D_Benchmark/2D_Threewell/samples.png"
  "2D_Benchmark/2D_Threewell/resample.png"
  "2D_Benchmark/2D_Himmelblau/samples.png"
  "2D_Benchmark/2D_Himmelblau/resample.png"
  "2D_Benchmark/2D_Annulus/samples.png"
  "2D_Benchmark/2D_Annulus/resample.png"
  "2D_Benchmark/2D_Python/samples.png"
  "2D_Benchmark/2D_Python/resample.png"
  "2D_Benchmark/2D_Chessboard/samples.png"
  "2D_Benchmark/2D_Chessboard/resample.png"
  "2D_Benchmark/2D_Sparse/samples.png"
  "2D_Benchmark/2D_Sparse/resample.png"
  # 5.2 -- sensor array
  "Sensor_Array/samples.png"
  "Sensor_Array/resample.png"
  # 5.3 -- high-dimensional product ladder
  "HD_Product_Ladder/ESS_ladder.png"
  # 5.4 -- phi^4 lattice (8x8)
  "Phi4_Lattice_8/figures/fig_methods.png"
  # 5.5 -- clock lattice
  "Clock_Lattice/figures/fig_clock_target.png"
  "Clock_Lattice/figures/fig_clock_esscurves.png"
  "Clock_Lattice/figures/occupancy_bias_B10k.png"
  # 5.6 -- screened-Poisson source inversion
  "Poisson_Inverse/figures/fig_setup.png"
  "Poisson_Inverse/figures/poisson_ladders.png"
  # 6 -- molecular Boltzmann generators
  "Molecular_BG/glycerol_36d/ladder.png"
  "Molecular_BG/glycerol_36d/conformers.png"
  "Molecular_BG/glycerol_36d/dihedrals.png"
  "Molecular_BG/diethanolamine_48d/ladder.png"
  "Molecular_BG/diethanolamine_48d/conformers.png"
  "Molecular_BG/diethanolamine_48d/dihedrals.png"
  "Molecular_BG/adp_60d/ess_history.png"
  "Molecular_BG/adp_60d/dihedrals.png"
  "Molecular_BG/adp_60d/conformers.png"
  "Molecular_BG/adp_60d/ramachandran.png"
)

echo "[$(ts)] START trim.sh | ${IM[0]} | fuzz=$FUZZ | ${#SRCS[@]} figures -> $HERE"
i=0; missing=0
for srcrel in "${SRCS[@]}"; do
  i=$((i + 1))
  dstrel="${srcrel//\/figures\//\/}"          # drop intermediate figures/ in dest
  src="$ROOT/$srcrel"
  dst="$HERE/$dstrel"
  if [[ ! -f "$src" ]]; then
    echo "[$(ts)] [$i/${#SRCS[@]}] MISSING  $srcrel"; missing=$((missing + 1)); continue
  fi
  mkdir -p "$(dirname "$dst")"
  "${IM[@]}" "$src" -fuzz "$FUZZ" -trim +repage "$dst"
  echo "[$(ts)] [$i/${#SRCS[@]}] trimmed  $dstrel"
done
echo "[$(ts)] DONE | $((i - missing))/$i written under $HERE${missing:+ ($missing missing)}"
