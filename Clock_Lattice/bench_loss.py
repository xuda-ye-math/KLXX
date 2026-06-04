# pyright: reportArgumentType=false, reportCallIssue=false
"""loss_compile affordability test at production size (L=8: D=64, NCSF 16/6/(256,256)).

Fused stage loss (KL + lam*X_mu + X_mix on a packed [2B, d] batch, roll-pairing)
eager vs zflows.loss.loss_compile. Stdout only."""
import math
import sys
import time

import torch

sys.path.insert(0, '.')
from zflows.flow import NCSF
from zflows.potential import Uniform
from zflows.loss import loss_compile
from zflows.utils import suppress_warnings
from core import bridge
from potential import Clock

suppress_warnings()
device = 'cuda'
L, D, B = 8, 64, 8000
lim = math.pi
torch.manual_seed(0)


def sync():
    torch.cuda.synchronize()


def timeit(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return 1000.0 * (time.perf_counter() - t0) / n


u0 = Uniform([-lim] * D, [lim] * D, device=device)
u = Clock(L, 6, 1.0, 0.5).to(device)
for p in (u0, u):
    p.enable_grad(mode="default")
    p.enable_eval(mode="default")
u_prev, u_next = bridge(u0, u, 0.25), bridge(u0, u, 0.6)

flow = NCSF(a=[-lim] * D, b=[lim] * D, bins=16, transforms=6,
            hidden_features=(256, 256)).to(device)
flow.zeros()
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)


def fused(yp, source, target, G, lam, half):
    x, ladj = G.call_and_ladj(yp)
    z = source(x) - target(yp) - ladj
    z_mu, z_mix = z[:half], z[half:]
    return (z_mu.mean() + lam * (z_mu - z_mu.roll(1)).abs().mean()
            + (z_mix - z_mix.roll(1)).abs().mean())


yp = (torch.rand(2 * B, D, device=device) * 2 - 1) * lim

F = flow.t()


def eager_step():
    loss = fused(yp, u_prev, u_next, F, 1.0, B)
    opt.zero_grad(); loss.backward(); opt.step()


ms = timeit(eager_step, n=3)
print(f"eager fused [2x{B},{D}]: {ms:8.1f} ms/step", flush=True)

t0 = time.perf_counter()
closs = loss_compile(fused, u_prev, u_next, flow.t(), 1.0, B)
v = closs(yp); v.backward(); opt.zero_grad()
sync()
print(f"loss_compile build+first call: {time.perf_counter()-t0:.1f} s", flush=True)


def comp_step():
    loss = closs(yp)
    opt.zero_grad(); loss.backward(); opt.step()


ms = timeit(comp_step, n=3)
print(f"compiled fused [2x{B},{D}]: {ms:8.1f} ms/step", flush=True)

with torch.no_grad():
    e = fused(yp, u_prev, u_next, flow.t(), 1.0, B)
    c = closs(yp)
    print(f"value check after optimizer steps: |eager-compiled| = "
          f"{(e-c).abs().item():.2e} ({'OK' if (e-c).abs().item() < 1e-3 else 'MISMATCH'})",
          flush=True)
print("DONE", flush=True)
