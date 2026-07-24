"""Standalone parallel-tempering reference for the tilted phi^4 minority weight
p_+ = P(m > 0), independent of the MALA metastable-vacuum construction.

WHY: the mirror+reweight reference (reference.py) samples ONE vacuum of the
h = 0 action and reweights; that is only valid while the chain stays confined in
that well. Parallel tempering removes the metastability problem entirely: hot
replicas melt the ~9 k_BT domain-wall barrier, replica swaps carry barrier
crossings down to the target replica (beta = 1), which then visits BOTH vacua in
their correct tilted ratio. p_+ is then a DIRECT sample fraction -- no
reweighting, no single-vacuum assumption.

Method (no MALA, no gradients):
  - Replicas at inverse temperatures beta_1 = 1 (target) > ... > beta_R (hot).
  - Local moves: checkerboard random-walk Metropolis. Even then odd sublattice;
    per site propose phi' = phi + sigma_r * xi, accept by the exact local energy
    change at inverse temperature beta_r. sigma_r adapts PER REPLICA during burn.
  - Replica swaps: every sweep, adjacent-pair exchange accepted by
    min(1, exp((beta_i - beta_j)(U_i - U_j))) with U the full tilted action.
  - Estimate: over the kept sweeps, p_+ = fraction of the beta = 1 walkers with
    m = mean(phi) > 0; std error from the spread across independent walkers.

Robustness (per MC review): walkers are started SPLIT between the +1 and -1
wells so the estimate cannot inherit a single-well bias and burn-in becomes
self-checking. The run logs the guards that must hold to trust the number:
running p_+ flat, mean/min barrier crossings per walker, and the WORST adjacent
swap acceptance (one adjacent pair with negligible acceptance would break
communication across the temperature grid).

The phi^4 action is written out here directly (not imported) so this is a true
independent cross-check of reference.py's energy and of the trained flow.

Run from the repo root (GPU):
    python Codes/Lattice_Phi4/pt_reference.py
Writes the temporary run log to artifacts/pt_reference.log and prints p_+
for L = 6, 8.  The artifacts directory can be removed after recording the
reference values used by the final figure and table scripts.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "pt_reference.log"

# --- physics (identical action to reference.py, plus the h tilt) -------------
KAPPA = 0.40
LAMBDA = 0.50
SIZES = ((6, 0.0257), (8, 0.0144))       # (L, h) per lattice size

# --- parallel-tempering controls ---------------------------------------------
R = 20                     # replicas
BETA_MIN = 0.05            # hottest inverse temperature (melts the ~9 kT barrier)
WALKERS = 512              # independent replica stacks (parallel chains)
BURN_SWEEPS = 4000         # sweeps before collecting
KEEP_SWEEPS = 16000        # collected sweeps
TUNE_EVERY = 100           # sigma-adaptation cadence during burn
SEED = 7


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def make_action(L: int, h: float):
    """full_energy(phi) and nn_sum(phi) for the tilted phi^4 action.

    U(phi) = sum_x [ -2 kappa phi_x (phi_{x+e1} + phi_{x+e2})
                     + phi_x^2 + lambda (phi_x^2 - 1)^2 + h phi_x ].
    Lattice axes are the last two of phi.
    """

    def full_energy(phi):
        hop = phi * (jnp.roll(phi, 1, axis=-2) + jnp.roll(phi, 1, axis=-1))
        s = -2.0 * KAPPA * hop + phi ** 2 + LAMBDA * (phi ** 2 - 1.0) ** 2 + h * phi
        return s.sum(axis=(-2, -1))

    def nn_sum(phi):
        return (jnp.roll(phi, 1, axis=-2) + jnp.roll(phi, -1, axis=-2)
                + jnp.roll(phi, 1, axis=-1) + jnp.roll(phi, -1, axis=-1))

    return full_energy, nn_sum


def run_size(L: int, h: float):
    assert L % 2 == 0, "checkerboard sublattice update requires even L"
    full_energy, nn_sum = make_action(L, h)
    betas = jnp.asarray(np.geomspace(1.0, BETA_MIN, R), dtype=jnp.float32)   # (R,)
    b = betas[:, None, None, None]                                           # (R,1,1,1)

    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    even = jnp.asarray(((ii + jj) % 2 == 0), dtype=bool)                     # (L,L)
    odd = ~even
    nsub = int(np.asarray(even).sum())                                      # sites per sublattice

    def half_sweep(phi, mask, sigma, key):
        # Metropolis update of the `mask` sublattice at every replica/walker
        k1, k2 = jax.random.split(key)
        prop = phi + sigma * jax.random.normal(k1, phi.shape)
        nn = nn_sum(phi)
        dphi = prop - phi
        dE = (dphi * (-2.0 * KAPPA * nn + h)
              + (prop ** 2 - phi ** 2)
              + LAMBDA * ((prop ** 2 - 1.0) ** 2 - (phi ** 2 - 1.0) ** 2))
        acc = (jax.random.uniform(k2, phi.shape) < jnp.exp(-b * dE)) & mask[None, None, :, :]
        phi = jnp.where(acc, prop, phi)
        acc_r = acc.sum(axis=(1, 2, 3)) / (nsub * phi.shape[1])              # (R,) per-replica accept
        return phi, acc_r

    def swap(phi, key, parity):
        # adjacent-pair replica exchange; parity selects (0,1),(2,3),.. or (1,2),(3,4),..
        U = full_energy(phi)                                                 # (R,W)
        i = jnp.arange(parity, R - 1, 2)
        dbeta = betas[i] - betas[i + 1]                                      # (P,)
        dU = U[i] - U[i + 1]                                                 # (P,W)
        do = jax.random.uniform(key, dU.shape) < jnp.exp(jnp.minimum(dbeta[:, None] * dU, 0.0))
        do4 = do[:, :, None, None]
        lo, hi = phi[i], phi[i + 1]
        phi = phi.at[i].set(jnp.where(do4, hi, lo))
        phi = phi.at[i + 1].set(jnp.where(do4, lo, hi))
        return phi, do.mean(axis=1)                                          # (P,) per-pair accept

    @jax.jit
    def sweep(phi, sigma, key):
        ke, ko, ks0, ks1 = jax.random.split(key, 4)
        phi, ae = half_sweep(phi, even, sigma, ke)
        phi, ao = half_sweep(phi, odd, sigma, ko)
        phi, s0 = swap(phi, ks0, 0)
        phi, s1 = swap(phi, ks1, 1)
        loc = 0.5 * (ae + ao)                                               # (R,)
        swap_min = jnp.minimum(s0.min(), s1.min())                          # worst adjacent level
        swap_mean = 0.5 * (s0.mean() + s1.mean())
        return phi, loc, swap_min, swap_mean

    key = jax.random.key(SEED + L)
    key, k0 = jax.random.split(key)
    # split start: half the walkers in the +well, half in the -well (unbiased)
    signs = jnp.where(jnp.arange(WALKERS) < WALKERS // 2, 1.0, -1.0)[None, :, None, None]
    phi = signs + 0.3 * jax.random.normal(k0, (R, WALKERS, L, L))
    sigma = jnp.asarray(0.12 / np.sqrt(np.asarray(betas)))[:, None, None, None]  # (R,1,1,1)

    log(f"L={L} h={h}: START PT  R={R} beta 1..{BETA_MIN} walkers={WALKERS} "
        f"burn={BURN_SWEEPS} keep={KEEP_SWEEPS}  (split +/- start)")
    t0 = time.time()
    for s in range(1, BURN_SWEEPS + 1):
        key, ks = jax.random.split(key)
        phi, loc, sw_min, sw_mean = sweep(phi, sigma, ks)
        if s % TUNE_EVERY == 0:
            loc_np = np.asarray(loc)                                        # (R,)
            # grow sigma when acceptance is above target (bigger moves -> lower acc)
            fac = np.clip((np.clip(loc_np, 1e-3, 1.0) / 0.5) ** 0.5, 0.5, 2.0)
            sigma = jnp.clip(sigma * jnp.asarray(fac)[:, None, None, None], 1e-3, 5.0)
            if s % 1000 == 0:
                log(f"  burn {s:>5}/{BURN_SWEEPS}  loc_acc[min,max]=[{loc_np.min():.2f},{loc_np.max():.2f}]"
                    f"  swap[min,mean]=[{float(sw_min):.2f},{float(sw_mean):.2f}]"
                    f"  p_+@b1={float((phi[0].mean(axis=(-2,-1))>0).mean()):.3f}"
                    f"  {1000*(time.time()-t0)/s:.0f} ms/sweep")

    half = KEEP_SWEEPS // 2
    hits = np.zeros(WALKERS, dtype=np.int64)                                # all kept sweeps
    hits2 = np.zeros(WALKERS, dtype=np.int64)                               # 2nd-half only
    tot = 0
    tot2 = 0
    sw_min_run = []
    crossings = np.zeros(WALKERS, dtype=np.int64)
    prev_sign = None
    for s in range(1, KEEP_SWEEPS + 1):
        key, ks = jax.random.split(key)
        phi, loc, sw_min, sw_mean = sweep(phi, sigma, ks)
        m1 = np.asarray(phi[0].mean(axis=(-2, -1)))                         # (W,) magnetization at beta=1
        pos = (m1 > 0.0)
        hits += pos
        tot += 1
        if s > half:
            hits2 += pos
            tot2 += 1
        sw_min_run.append(float(sw_min))
        if prev_sign is not None:
            crossings += (pos != prev_sign)
        prev_sign = pos
        if s % 2000 == 0:
            log(f"  keep {s:>5}/{KEEP_SWEEPS}  p_+(run)={hits.sum()/(tot*WALKERS):.4f}"
                f"  swap_min(2k)={np.min(sw_min_run[-2000:]):.2f}"
                f"  crossings[mean,min]=[{crossings.mean():.1f},{crossings.min()}]")

    p_walker = hits / tot
    p_plus = float(p_walker.mean())
    sem = float(p_walker.std(ddof=1) / np.sqrt(WALKERS))
    p_2nd = float((hits2 / tot2).mean())                                    # flatness cross-check
    log(f"L={L}: DONE  p_+ = {p_plus:.4f} +- {sem:.4f}  (2nd-half {p_2nd:.4f})  "
        f"crossings[mean,min]=[{crossings.mean():.1f},{crossings.min()}]  "
        f"swap_min(all)={np.min(sw_min_run):.2f}   {time.time()-t0:.0f}s")
    return dict(L=L, p_plus=p_plus, sem=sem, p_2nd=p_2nd,
                cross_mean=float(crossings.mean()), cross_min=int(crossings.min()),
                swap_min=float(np.min(sw_min_run)))


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()
    log("parallel-tempering phi^4 reference  |  jax " + jax.__version__
        + "  backend " + jax.default_backend())
    res = [run_size(L, h) for L, h in SIZES]
    log("=== SUMMARY ===")
    for r in res:
        ok = (r["cross_min"] > 0 and r["swap_min"] > 0.15
              and abs(r["p_plus"] - r["p_2nd"]) < 3 * r["sem"])
        log(f"  L={r['L']}: p_+ = {r['p_plus']:.4f} +- {r['sem']:.4f}  "
            f"(2nd-half {r['p_2nd']:.4f})  crossings[mean,min]="
            f"[{r['cross_mean']:.1f},{r['cross_min']}]  swap_min={r['swap_min']:.2f}  "
            f"trust_guards={'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
