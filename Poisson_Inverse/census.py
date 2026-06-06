# pyright: reportArgumentType=false
"""Fast inference on a (partial or final) ladder checkpoint: compose the stage
inverses, extend by identity, reweight at the ladder-top temper, print the
well census. Collapse check without waiting for the run.

    ~/.envs/torch/bin/python census.py data_m6_kl_partial.pth
"""
import sys
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import compute_ESS_log, suppress_warnings
from core import compose_pushforward
import parameters as p
import potential as pot
from train import well_census

suppress_warnings()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

D = torch.load(HERE / sys.argv[1], weights_only=False, map_location='cpu')
cfg = D['config']
p.M_LOW, p.M_FULL, p.N_SENSORS = cfg['m_low'], cfg['m_full'], cfg['n_sensors']
t_top = D.get('t_now', 1.0)
torch.manual_seed(123)
B = pot.build(p, device)
d, d_full = B['d_low'], B['d_full']
std_all = B['std_all'].to(device)
u0 = Gaussian([0.0] * d, [1.0] * d, device=device).enable_eval(mode="default")
u_low_t = B['make'](d, temper=t_top).enable_eval(mode="default")
u_full_t = B['make'](d_full, temper=t_top)

flow = NSF(a=[-p.NSF_LIM] * d, b=[p.NSF_LIM] * d, bins=cfg.get('bins', p.BINS),
           transforms=cfg.get('transforms', p.TRANSFORMS),
           hidden_features=cfg.get('hidden', p.HIDDEN)).to(device)
F_inv = flow.t().enable_inv_ladj()

n = 50000
y_low, logw_low = compose_pushforward(
    flow, F_inv, [s['state_dict'] for s in D['stages']], u0, u_low_t, n, device)
low_ess = compute_ESS_log(logw_low).item()
with torch.no_grad():
    xh = torch.randn(n, d_full - d, device=device)
    y_ext = torch.cat([y_low, xh], dim=1)
    # misfit() is UNTEMPERED Phi: scale the extension correction by t_top (R6)
    logw_full = logw_low + t_top * (u_low_t.misfit(y_low) - u_full_t.misfit(y_ext))
fine_ess = compute_ESS_log(logw_full).item()

tr_nc = B['xi_truth'][:d].to(device).clone(); tr_nc[0] = 0.0
occ_push = well_census(y_low, std_all, tr_nc, p.ALPHA)
w = (logw_full - logw_full.max()).exp()
occ_rew = well_census(y_low, std_all, tr_nc, p.ALPHA, weights=w)

print(f"{D['tag']}: K={len(D['stages'])} t_top={t_top:.4f} "
      f"ladder={[round(t, 3) for t in D['ladder']]}")
print(f"low ESS={low_ess:.4f}  FINE ESS={fine_ess:.4f}  (at temper {t_top:.3f})")
print(f"wells pushforward: {occ_push}")
print(f"wells reweighted:  {occ_rew}")
n_push = sum(1 for v in occ_push.values() if v > 0.01)
print(f"=> {n_push} wells hold >1% of pushforward mass "
      f"(referee lattice: 10 wells at full temper)")

# per-well decomposition: restricted fine ESS and restricted low ESS inside
# each well separate in-well distortion (low restricted ESS small) from
# well-weight error (restricted fine high, global fine small)
th0 = y_low[:, 0] * std_all[0]
n_lab = torch.round(th0 * p.ALPHA / (2.0 * math.pi)).long()
s_lab = ((y_low * tr_nc.unsqueeze(0)).sum(-1) > 0).long()
pair = (n_lab * 2 + s_lab)
print("per-well restricted ESS (wells holding >2% pushforward):")
for u_ in pair.unique().tolist():
    m_ = pair == u_
    if m_.float().mean() < 0.02:
        continue
    lab = (u_ // 2 if u_ >= 0 else -((-u_ + 1) // 2), u_ % 2)
    el = compute_ESS_log(logw_low[m_]).item()
    ef = compute_ESS_log(logw_full[m_]).item()
    print(f"  well {lab}: frac {m_.float().mean():.3f}  "
          f"low ESS {el:.3f}  fine ESS {ef:.3f}")
