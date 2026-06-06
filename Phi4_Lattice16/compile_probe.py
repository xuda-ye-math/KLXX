"""Time the NSF inverse compile at a given lattice size (no QT/pilot needed)."""
import argparse, time, os, sys
import torch
from zflows.flow import NSF

ap = argparse.ArgumentParser()
ap.add_argument('--L', type=int, required=True)
args = ap.parse_args()
D = args.L * args.L
device = 'cuda'
flow = NSF(a=[-3.0]*D, b=[3.0]*D, bins=16, transforms=6, hidden_features=(256, 256)).to(device)
flow.zeros()
G = flow.t()
x = torch.randn(500, D, device=device)
t0 = time.perf_counter()
G.enable_inv_ladj()
y, _ = G.inv_ladj(x)
torch.cuda.synchronize()
t_compile = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(10):
    y, _ = G.inv_ladj(x)
torch.cuda.synchronize()
t_step = (time.perf_counter() - t0) / 10
print(f"L={args.L} D={D}: compile+first-call {t_compile:.1f}s, steady inverse {t_step*1000:.0f} ms/call", flush=True)
