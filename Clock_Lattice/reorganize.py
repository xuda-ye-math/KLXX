# pyright: reportArgumentType=false
"""One-time reorganization (no retraining): rename method-less balance runs
data_L4.pth / data_L8.pth -> data_L{L}_balance.pth, updating the internal
tag and config['method'], then regenerate figures and tables."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

for old, new in [('L4', 'L4_balance'), ('L8', 'L8_balance')]:
    src = HERE / f'data_{old}.pth'
    dst = HERE / f'data_{new}.pth'
    if not src.exists():
        print(f'skip {src.name} (not found)')
        continue
    if dst.exists():
        print(f'skip {src.name} -> {dst.name} (target exists)')
        continue
    d = torch.load(src, weights_only=False, map_location='cpu')
    d['tag'] = new
    d['config']['method'] = 'balance'
    torch.save(d, dst)
    src.unlink()
    print(f'renamed {src.name} -> {dst.name} (tag/method updated)')

# regenerate figures + tables under the new tags
sys.path.insert(0, str(HERE))
import plot_results
import train as T
for tag in ['L4_balance', 'L8_balance']:
    if (HERE / f'data_{tag}.pth').exists():
        plot_results.main(tag)
T.write_results()
print('results_table rebuilt')
