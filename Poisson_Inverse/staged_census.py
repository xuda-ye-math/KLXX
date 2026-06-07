# pyright: reportArgumentType=false
"""Staged-sampler census: replay the saved per-stage state_dicts EXACTLY as the
procedure runs (per stage: inverse map -> per-stage reweight -> resample ->
Langevin rejuvenation), then the per-step fine correction at t=1, and census
the well weights. No composed map anywhere.
    ~/.envs/torch/bin/python staged_census.py data_balance_o0.01.pth
"""
import sys, math
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import compute_ESS_log, resample, langevin
from core import bridge, identity_wrap
import parameters as p
import potential as pot
from train import well_census

device = 'cuda' if torch.cuda.is_available() else 'cpu'
D = torch.load(HERE / sys.argv[1], weights_only=False, map_location='cpu')
cfg = D['config']
p.M_LOW, p.M_FULL, p.N_SENSORS = cfg['m_low'], cfg['m_full'], cfg['n_sensors']
p.SIGMA_OBS = cfg['sigma_obs']
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 123
torch.manual_seed(seed)
B = pot.build(p, device)
d, d_full, std_all = B['d_low'], B['d_full'], B['std_all'].to(device)
u0 = Gaussian([0.0] * d, [1.0] * d, device=device)
u = B['u_low']

flow = NSF(a=[-p.NSF_LIM] * d, b=[p.NSF_LIM] * d, bins=p.BINS,
           transforms=p.TRANSFORMS, hidden_features=p.HIDDEN).to(device)
F_inv = flow.t().enable_inv_ladj()

N = 100000
Y = u0.samples(N)
t_prev = 0.0
with torch.no_grad():
    for s in D['stages']:
        t_k = s['t']
        u_prev = bridge(u0, u, t_prev).enable_grad().enable_eval(mode="default")
        u_next = bridge(u0, u, t_k).enable_grad().enable_eval(mode="default")
        flow.load_state_dict({k: v.to(device) for k, v in s['state_dict'].items()})
        outs, lws = [], []
        for xb in Y.split(50000):
            yt, ladj = F_inv.inv_ladj(xb)
            yt, ladj = yt.clone(), ladj.clone()
            lws.append(u_prev.eval(xb).clone() - u_next.eval(yt).clone() + ladj)
            outs.append(yt)
        yt, logw = torch.cat(outs), torch.cat(lws)
        ess = compute_ESS_log(logw).item()
        Y = resample(yt, (logw - logw.max()).exp())
        Y = langevin(Y, u_next, step=p.MC_STEP, iters=p.MC_ITERS,
                     chunk=max(1, N // 50000))
        print(f"  stage t={t_k:.3f}: per-stage ESS {ess:.3f}")
        t_prev = t_k
    # per-step fine correction at t = 1 (extension by prior high modes)
    xh = torch.randn(N, d_full - d, device=device)
    ye = torch.cat([Y, xh], dim=1)
    logw_f = u.misfit(Y) - B['u_full'].misfit(ye)
    fine = compute_ESS_log(logw_f).item()

tr_nc = B['xi_truth'][:d].to(device).clone(); tr_nc[0] = 0.0
occ = well_census(Y, std_all, tr_nc, p.ALPHA,
                  weights=(logw_f - logw_f.max()).exp())
R = torch.load(HERE / f"referee_o{cfg['sigma_obs']:g}.pth",
               weights_only=False, map_location='cpu')
ref = R['census']
tv = 0.5 * sum(abs(ref.get(k, 0) - occ.get(k, 0)) for k in set(ref) | set(occ))
print(f"staged-sampler census ({sys.argv[1]}): fine step ESS at t=1: {fine:.3f}")
print(f"TV(staged census, referee) = {tv:.4f}")
print({k: round(v, 3) for k, v in occ.items() if v > 0.005})
