# pyright: reportArgumentType=false, reportCallIssue=false
"""Sanity: does flow.zeros() (and training) keep the CAPTURED compiled paths
correct -- enable_inv_ladj (CUDA graphs) and loss_compile -- across repeated
zeros -> train -> zeros cycles, with outputs cloned out of the static buffers?

Production size: D=64, NCSF(16, 6, (256,256)). Stdout only."""
import math
import sys

import torch

sys.path.insert(0, '.')
from zflows.flow import NCSF
from zflows.potential import Uniform
from zflows.loss import loss_compile
from zflows.utils import suppress_warnings
from core import fused_stage_loss
from potential import Clock

suppress_warnings()
device = 'cuda'
L, D, B = 8, 64, 2000
lim = math.pi
torch.manual_seed(0)

u0 = Uniform([-lim] * D, [lim] * D, device=device)
u = Clock(L, 6, 1.0, 0.5).to(device)
for p in (u0, u):
    p.enable_grad(mode="reduce-overhead")
    p.enable_eval(mode="default")

flow = NCSF(a=[-lim] * D, b=[lim] * D, bins=16, transforms=6,
            hidden_features=(256, 256)).to(device)
flow.zeros()
F_inv = flow.t().enable_inv_ladj()
tp = torch.zeros((), device=device)
tn = torch.full((), 0.6, device=device)
closs = loss_compile(fused_stage_loss, u0, u, tp, tn, flow.t(), 1.0, B)

x = (torch.rand(B, D, device=device) * 2 - 1) * lim
yp = (torch.rand(2 * B, D, device=device) * 2 - 1) * lim

ok = True
for cycle in (1, 2):
    flow.zeros()                                   # identity reset, in place
    with torch.no_grad():
        y, l = F_inv.inv_ladj(x)
        y, l = y.clone(), l.clone()                # clone out of static buffers
        e_id = (y - x).abs().max().item() + l.abs().max().item()
    print(f"[cycle {cycle}] after zeros(): compiled inverse is identity? "
          f"max err = {e_id:.2e} ({'OK' if e_id < 1e-4 else 'FAIL'})", flush=True)
    ok &= e_id < 1e-4

    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    for _ in range(5):                             # simulate training
        loss = closs(yp)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        y1 = F_inv.inv_ladj(x)[0].clone()
        y0 = flow.t().inv(x)                       # fresh eager truth
        e_tr = (y1 - y0).abs().max().item()
        c1 = closs(yp).item()
        c0 = fused_stage_loss(yp, u0, u, tp, tn, flow.t(), 1.0, B).item()
    print(f"[cycle {cycle}] after 5 train steps: compiled inverse vs fresh "
          f"eager max err = {e_tr:.2e} ({'OK' if e_tr < 1e-4 else 'FAIL'}); "
          f"compiled loss vs eager: |{c1:.6f} - {c0:.6f}| = {abs(c1-c0):.2e} "
          f"({'OK' if abs(c1-c0) < 1e-3 else 'FAIL'})", flush=True)
    ok &= e_tr < 1e-4 and abs(c1 - c0) < 1e-3

    # temperature-stage change: .fill_() new (t_{k-1}, t_k) must be seen by
    # the ONE compiled loss with no recompile/staleness
    tp.fill_(0.25 * cycle)
    tn.fill_(min(0.25 * cycle + 0.35, 1.0))
    with torch.no_grad():
        c1 = closs(yp).item()
        c0 = fused_stage_loss(yp, u0, u, tp, tn, flow.t(), 1.0, B).item()
    print(f"[cycle {cycle}] after temperature fill_ (tp={tp.item():.2f}, "
          f"tn={tn.item():.2f}): compiled vs eager |{c1:.6f} - {c0:.6f}| = "
          f"{abs(c1-c0):.2e} ({'OK' if abs(c1-c0) < 1e-3 else 'FAIL'})",
          flush=True)
    ok &= abs(c1 - c0) < 1e-3
    tp.fill_(0.0); tn.fill_(0.6)                   # restore for next cycle

print("VERDICT:", "ALL OK -- zeros()/training tracked by captured compiled "
      "paths (with output clones)" if ok else "FAIL -- do not reuse captures "
      "across zeros()", flush=True)
print("DONE", flush=True)
