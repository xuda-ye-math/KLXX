"""Recompute the Naeem k-NN coverage at k = 3, 4, 5 for each trained method,
WITHOUT retraining. The 20k pushforward samples y[:20000] are already saved in
data.pth; only the QT hat_pool (the coverage reference x) is missing, and it is
built deterministically (torch.manual_seed(1) -> u0.samples -> quench_and_temper
with fixed params), so we rebuild it here and validate against the saved QT
occupancy + the saved k=5 coverage.

Usage: ~/.envs/torch/bin/python recompute_coverage.py
"""
import torch
torch.set_num_threads(32)
from zflows.potential import Gaussian
from zflows.utils import suppress_warnings

import core
from core import SensorArrayPosterior, quench_and_temper, coverage, mode_coverage_nearest
import parameters as P

suppress_warnings()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
KS = (3, 4, 5)


def rebuild_qt_pool(u0, u1):
    """Reproduce the exact QT pool built in train.main()."""
    torch.manual_seed(1)
    x_qt = u0.samples(P.P_QT)
    hat_pool = quench_and_temper(x_qt, u1, sigma=P.QT_SIGMA,
                                 opt_step=P.QT_OPT_STEP, opt_iters=P.QT_OPT_ITERS,
                                 mc_step=P.QT_MC_STEP, mc_iters=P.QT_MC_ITERS)
    return hat_pool


def main():
    d = torch.load('data.pth', map_location=device, weights_only=False)
    centers = d['centers'].to(device)

    # rebuild targets exactly as train.build_targets()
    sensors = core.sensor_positions(P.N_SENSORS, P.SENSOR_LIM)
    data = core.make_data(P.THETA_STAR, sensors, P.ELL, P.SIGMA_OBS, seed=P.DATA_SEED)
    u0 = Gaussian(mean=[0.0] * P.N_SRC, variance=[P.SIGMA_PRIOR ** 2] * P.N_SRC).to(device)
    u1 = SensorArrayPosterior(sensors, data, P.ELL, P.SIGMA_OBS, P.SIGMA_PRIOR).to(device)
    u1.enable_grad(); u1.enable_eval()

    # ---- validate QT-pool reconstruction ----
    hat_pool = rebuild_qt_pool(u0, u1)
    qt_cov, qt_occ, _ = mode_coverage_nearest(hat_pool, centers, frac=P.MODE_FRAC)
    saved_occ = d['qt_occupancy']
    print(f"QT pool rebuilt: {hat_pool.shape[0]} samples  modes_found={qt_cov*d['nmodes']:.0f}/{d['nmodes']}")
    print(f"  occ (rebuilt): {[round(v,3) for v in qt_occ.tolist()]}")
    print(f"  occ (saved)  : {[round(v,3) for v in saved_occ.tolist()]}")
    print(f"  saved qt_mode_coverage={d['qt_mode_coverage']}\n")

    ref = hat_pool[:min(2000, hat_pool.shape[0])]

    # ---- recompute coverage at each k; reproduce saved k=5 as a check ----
    hdr = f"{'method':<24}" + "".join(f"  k={k}" for k in KS) + "   (saved k=5)   modes"
    print(hdr)
    print("-" * len(hdr))
    for m, r in d['runs'].items():
        y = r['samples'].to(device)
        covs = {k: coverage(y, ref, k=k) for k in KS}
        row = f"{m:<24}" + "".join(f"  {covs[k]:.3f}" for k in KS)
        row += f"      {r['knn_coverage']:.3f}      {r['modes_found']}/{d['nmodes']}"
        print(row)


if __name__ == '__main__':
    main()
