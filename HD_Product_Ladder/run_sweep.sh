#!/usr/bin/env bash
# AIS-ladder sweep: one flow per M, d=256, 1000 steps, inverse compile skipped.
# Run:  bash run_sweep.sh        (foreground)
#   or: nohup bash run_sweep.sh > full_run.stdout 2>&1 &   (overnight)
set -euo pipefail

cd "$(dirname "$0")"
PY=~/.envs/torch/bin/python

for M in 1 2 4 8 16 32; do
    echo "===== M=$M ====="
    "$PY" train.py --M "$M"
done

echo "===== sweep done, plotting ====="
"$PY" plot_ess.py
