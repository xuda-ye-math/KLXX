# One-shot VRAM probe for the full-size L=8 run: replicates the three peak
# memory ops of train.py at production sizes WITHOUT running the ladder.
#   (a) compiled fused stage loss: packed batch 2B through G.call_and_ladj
#       with autograd forward + backward (the training-step peak),
#   (b) compiled fused inverse inv_ladj at the training batch B,
#   (c) inv_ladj at the validation/composition chunk size,
# each reported as torch.cuda max allocated / reserved.  Resident tensors
# (1M validation set, pools) are allocated too so the headroom number is
# honest.  Usage: /opt/torch/bin/python vram_probe.py [--chunk 200000]
import sys
import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zflows.flow import NCSF
from zflows.loss import loss_compile
from zflows.utils import suppress_warnings
from core.boltzmann import fused_stage_loss
from potential import Clock
from zflows.potential import Uniform
import parameters as PRM

suppress_warnings()
device = 'cuda'

ap = argparse.ArgumentParser()
ap.add_argument('--chunk', type=int, default=200000)
args = ap.parse_args()

D, B, NV, NP = PRM.D, PRM.N_BATCH, PRM.N_VALID, PRM.N_POOL
print(f"D={D} B={B} N_VALID={NV} N_POOL={NP} chunk={args.chunk}")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU total {total:.1f} GiB")

a = [-PRM.NSF_LIM] * D
b = [PRM.NSF_LIM] * D
flow = NCSF(a, b, bins=PRM.BINS, transforms=PRM.TRANSFORMS,
            hidden_features=PRM.HIDDEN).to(device)
flow.zeros()
u0 = Uniform(a, b, device=device)
u = Clock(PRM.L, PRM.P, PRM.J, PRM.H).to(device)
for pot in (u0, u):                      # same modes as train.py
    pot.enable_grad(mode="reduce-overhead")
    pot.enable_eval(mode="default")

# resident tensors of the real run (validation set + QT pool + hat pool)
resident = [torch.rand(NV, D, device=device) * 2 * PRM.NSF_LIM - PRM.NSF_LIM,
            torch.rand(NP, D, device=device), torch.rand(NP, D, device=device)]

tp_buf = torch.zeros((), device=device)
tn_buf = torch.full((), 0.25, device=device)
closs = loss_compile(fused_stage_loss, u0, u, tp_buf, tn_buf, flow.t(),
                     float(PRM.LAMBDA), B)
F_inv = flow.t().enable_inv_ladj()
opt = torch.optim.Adam(flow.parameters(), lr=PRM.LR)


def report(label):
    alloc = torch.cuda.max_memory_allocated() / 2**30
    resv = torch.cuda.max_memory_reserved() / 2**30
    print(f"[{label}] peak allocated {alloc:.2f} GiB  reserved {resv:.2f} GiB "
          f"({100 * resv / total:.0f}% of card)")
    torch.cuda.reset_peak_memory_stats()


# (a) training step: packed 2B batch, forward + backward + opt.step
yp = torch.rand(2 * B, D, device=device) * 2 * PRM.NSF_LIM - PRM.NSF_LIM
for i in range(3):                       # first iter compiles, later iters steady
    loss = closs(yp)
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()
report(f"train step 2B={2*B}")

# (b) compiled inverse at the training batch size
with torch.no_grad():
    xb = torch.rand(B, D, device=device) * 2 * PRM.NSF_LIM - PRM.NSF_LIM
    for i in range(2):
        yt, ladj = F_inv.inv_ladj(xb)
        yt, ladj = yt.clone(), ladj.clone()
torch.cuda.synchronize()
report(f"inv_ladj B={B}")

# (c) compiled inverse at the validation/composition chunk size
with torch.no_grad():
    xc = torch.rand(args.chunk, D, device=device) * 2 * PRM.NSF_LIM - PRM.NSF_LIM
    for i in range(2):
        yt, ladj = F_inv.inv_ladj(xc)
        yt, ladj = yt.clone(), ladj.clone()
torch.cuda.synchronize()
report(f"inv_ladj chunk={args.chunk}")

print("OK")
