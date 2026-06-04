# pyright: reportArgumentType=false, reportCallIssue=false
"""Compile-feature sanity/timing test at the PRODUCTION network size.

D = 64 (L=8 clock), NCSF(bins=16, transforms=6, hidden=(256,256)).
Times, with cuda synchronization:
  1. langevin(50 iters) on [8000,64]: Potential.enable_grad 'default' vs
     'reduce-overhead'
  2. NCSF forward call_and_ladj + backward: eager vs t().enable_for_ladj()
  3. NCSF inverse on [4000,64]: eager (inv compile is NOT attempted -- known
     pathologically slow Dynamo trace for deep autoregressive inverses)
  4. full mimic of one train_stage step at batch 2000 and 8000
Writes nothing (stdout only).
"""
import math
import sys
import time

import torch

sys.path.insert(0, '.')
from zflows.flow import NCSF
from zflows.potential import Uniform
from zflows.utils import langevin, resample, compute_ESS_log, suppress_warnings
from core import bridge, wrap_torus
from potential import Clock

suppress_warnings()
device = 'cuda'
L, D = 8, 64
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


def make_pots(mode):
    u0 = Uniform([-lim] * D, [lim] * D, device=device)
    u = Clock(L, 6, 1.0, 0.5).to(device)
    for p in (u0, u):
        p.enable_grad(mode=mode)
        p.enable_eval(mode=mode)
    return u0, u


x8k = (torch.rand(8000, D, device=device) * 2 - 1) * lim
x2k = x8k[:2000]

print("== 1. langevin(50 iters) on [8000,64], bridge potential ==", flush=True)
for mode in ("default", "reduce-overhead"):
    u0, u = make_pots(mode)
    ut = bridge(u0, u, 0.6)
    ms = timeit(lambda: langevin(x8k, ut, step=1e-3, iters=50), n=3)
    print(f"  mode={mode:<16} {ms:8.1f} ms/call", flush=True)

print("== 2. NCSF forward call_and_ladj + backward, [8000,64] ==", flush=True)
flow = NCSF(a=[-lim] * D, b=[lim] * D, bins=16, transforms=6,
            hidden_features=(256, 256)).to(device)
flow.zeros()
u0, u = make_pots("reduce-overhead")
u_prev, u_next = bridge(u0, u, 0.25), bridge(u0, u, 0.6)


def fwd_bwd(F, xb):
    xx, ladj = F.call_and_ladj(xb)
    z = u_prev(xx) - u_next(xb) - ladj
    loss = z.mean() + (z - z.roll(1)).abs().mean()
    loss.backward()


F_eager = flow.t()
ms = timeit(lambda: fwd_bwd(F_eager, x8k), n=3)
print(f"  eager        [8000]: {ms:8.1f} ms", flush=True)
ms = timeit(lambda: fwd_bwd(F_eager, x2k), n=3)
print(f"  eager        [2000]: {ms:8.1f} ms", flush=True)
try:
    F_c = flow.t()
    t0 = time.perf_counter()
    F_c.enable_for_ladj()
    ms = timeit(lambda: fwd_bwd(F_c, x8k), n=3)
    print(f"  for_ladj-compiled [8000]: {ms:8.1f} ms "
          f"(compile {time.perf_counter()-t0:.0f}s)", flush=True)
except Exception as e:
    print(f"  enable_for_ladj failed: {type(e).__name__}: {e}", flush=True)

print("== 3. NCSF inverse: eager vs enable_inv_ladj ==", flush=True)
with torch.no_grad():
    ms = timeit(lambda: F_eager.inv(x8k[:4000]), n=3)
    print(f"  eager inv [4000,64]: {ms:8.1f} ms", flush=True)
    ms = timeit(lambda: F_eager.inv(x8k[:1000]), n=3)
    print(f"  eager inv [1000,64]: {ms:8.1f} ms", flush=True)
try:
    F_i = flow.t()
    t0 = time.perf_counter()
    F_i.enable_inv_ladj()
    with torch.no_grad():
        _ = F_i.inv_ladj(x8k[:4000])              # trigger compile/capture
    sync()
    print(f"  inv_ladj compile+first call: {time.perf_counter()-t0:.1f} s",
          flush=True)
    with torch.no_grad():
        ms = timeit(lambda: F_i.inv_ladj(x8k[:4000]), n=3)
        print(f"  inv_ladj [4000,64]: {ms:8.1f} ms", flush=True)
        ms = timeit(lambda: F_i.inv_ladj(x8k[:1000]), n=3)
        print(f"  inv_ladj [1000,64]: {ms:8.1f} ms", flush=True)
    # correctness: does the compiled path track a parameter update?
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.01 * torch.randn_like(p))
        y0, l0 = flow.t().inv.call_and_ladj(x8k[:256])   # fresh eager truth
        y1, l1 = F_i.inv_ladj(x8k[:256])
        err = (y0 - y1).abs().max().item() + (l0 - l1).abs().max().item()
        print(f"  tracks param update: max err = {err:.2e} "
              f"({'OK' if err < 1e-4 else 'STALE — do not use in training'})",
              flush=True)
except Exception as e:
    print(f"  enable_inv_ladj failed: {type(e).__name__}: {e}", flush=True)

print("== 4. full mimic train step (reduce-overhead pots, eager flow) ==", flush=True)


def step(batch):
    xb = x8k[:batch]
    half = batch // 2
    with torch.no_grad():
        logw = u_prev.eval(xb) - u_next.eval(xb)
        xm = resample(xb, (logw - logw.max()).exp())
        xm = wrap_torus(langevin(xm, u_next, step=1e-3, iters=50), lim)
        xb_bar = F_eager.inv(xb[:half])
        xh = wrap_torus(langevin(xb[:half], u_next, step=1e-3, iters=50), lim)
        xmix = torch.cat([xh, xb_bar])
    fwd_bwd(flow.t(), xm)
    fwd_bwd(flow.t(), xmix)


for b in (2000, 8000):
    ms = timeit(lambda: step(b), n=3)
    print(f"  batch {b}: {ms:8.1f} ms/step", flush=True)
print("DONE", flush=True)
