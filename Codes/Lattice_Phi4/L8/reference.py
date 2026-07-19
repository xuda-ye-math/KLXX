"""Reference ensemble for the phi^4 lattice benchmark — writes phi4_reference.npz.

The h = 0 action is exactly Z2 symmetric, so the two vacua are related by the
mirror phi -> -phi. Long MALA chains sample the m > 0 vacuum of the symmetric
action (the barrier is never crossed and never needs to be); the mirror of the
trace IS the m < 0 vacuum ensemble, exactly. Reweighting the pair by
exp(-h sum phi) / exp(+h sum phi) then yields the broken-phase occupancies

    p_+ / p_- = E_+[exp(-h sum phi)] / E_+[exp(+h sum phi)],

up to the O(exp(-barrier)) inter-well overlap. The saved npz holds an
UNWEIGHTED magnetization trace (multinomially resampled from the weighted
mirror pair), a resampled set of configurations for the vacuum heatmaps, and
the barrier height read off the weighted magnetization histogram.

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/data/projects/jflows python \
        Codes/Lattice_Phi4/L8/reference.py
Writes temporary reference data and its log below ``artifacts/``.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import potential_from
from jflows.utils import langevin, resample

langevin_jit = eqx.filter_jit(langevin)   # one compiled executable per shape

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "reference.log"

# lattice action (h enters only through the reweighting)
L = 8                   # lattice side; D = L*L sites
D = L * L
KAPPA = 0.40            # hopping
LAMBDA = 0.50           # quartic self-coupling
H = 0.0144              # explicit Z2 breaking (frozen by the pilot scan)

# MALA sampling of the h = 0 action inside the m > 0 vacuum
WALKERS: int = 4096     # parallel chains, all started at phi = +1
BURN_STEPS: int = 300000  # burn-in steps (safe to run long: Z2-fold blocks inter-well leak)
KEEPS: int = 1000        # kept states per walker
KEEP_EVERY: int = 200    # steps between kept states
MC_DT: float = 1e-3   # MALA step size

# stored ensemble
TRACE_SZIE: int = 1000000  # unweighted magnetization trace length (resampled)
SNAP_SZIE: int = 256       # stored configurations (vacuum heatmaps)
HIST_BINS: int = 81     # magnetization histogram bins on (-PLT_LIM, PLT_LIM)
PLT_LIM = 1.6           # half-width of the magnetization histogram window


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


# h = 0 action: exactly Z2 symmetric
def phi4_h0_energy(x):
    phi = x.reshape(-1, L, L)
    hop = phi * (jnp.roll(phi, 1, axis=1) + jnp.roll(phi, 1, axis=2))
    s = -2.0 * KAPPA * hop + phi**2 + LAMBDA * (phi**2 - 1.0) ** 2
    return s.sum(axis=(1, 2))


u_h0 = potential_from(phi4_h0_energy)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START phi4 reference L={L} (D={D}) | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | kappa={KAPPA} lambda={LAMBDA} h={H} | "
        f"walkers={WALKERS} burn={BURN_STEPS} keeps={KEEPS}x{KEEP_EVERY} "
        f"MALA step={MC_DT}")
    key = jax.random.key(11)

    # burn-in inside the m > 0 vacuum of the symmetric action
    x = jnp.ones((WALKERS, D))
    x = langevin_jit(
        jax.random.fold_in(key, 0), x, u_h0,
        dt=MC_DT, steps=BURN_STEPS, adjust=True,
    )
    x = jnp.where(jnp.mean(x, axis=1, keepdims=True) < 0.0, -x, x)   # Z2-fold into the m>0 well
    x = jax.block_until_ready(x)
    log(f"burn-in done ({BURN_STEPS} MALA steps); mean m = {float(x.mean()):.4f}")

    trace = np.empty((KEEPS, WALKERS, D), dtype=np.float32)
    t0 = time.time()
    for t in range(1, KEEPS + 1):
        x = langevin_jit(
            jax.random.fold_in(key, t), x, u_h0,
            dt=MC_DT, steps=KEEP_EVERY, adjust=True,
        )
        x = jnp.where(jnp.mean(x, axis=1, keepdims=True) < 0.0, -x, x)   # keep every walker in the m>0 well
        trace[t - 1] = np.asarray(x, dtype=np.float32)
        if t % 200 == 0 or t == 1:
            log(f"keep {t:>5}/{KEEPS}   mean m = {float(x.mean()):.4f}   "
                f"{1000.0 * (time.time() - t0) / t:.0f} ms/keep")
    flat = trace.reshape(-1, D)                       # [KEEPS*WALKERS, D], m > 0 well

    # Z2 mirror pair, reweighted by the field term exp(-h sum phi)
    s_sum = flat.sum(axis=1, dtype=np.float64)
    lw = np.concatenate([-H * s_sum, +H * s_sum])     # [+well, mirrored -well]
    w = np.exp(lw - lw.max())
    n = flat.shape[0]
    p_plus = float(w[:n].sum() / w.sum())
    dF = float(np.log(max(p_plus, 1e-9) / max(1.0 - p_plus, 1e-9)))
    log(f"reweighted occupancy: p(m>0) = {p_plus:.4f}   Delta F = {dF:.3f} kT")

    m_all = np.concatenate([flat.mean(axis=1), -flat.mean(axis=1)])
    key_m, key_s = jax.random.split(jax.random.fold_in(key, 10000))
    m_trace = np.asarray(resample(key_m, jnp.asarray(m_all)[:, None],
                                  jnp.asarray(w), N=TRACE_SZIE)).ravel()

    # configurations for the vacuum heatmaps (resampled from the mirror pair)
    cfg_all = np.concatenate([flat, -flat], axis=0)
    samples_t1 = np.asarray(resample(key_s, jnp.asarray(cfg_all),
                                     jnp.asarray(w), N=SNAP_SZIE))

    # barrier height from the weighted magnetization histogram
    h_w, e = np.histogram(m_all, bins=HIST_BINS, range=(-PLT_LIM, PLT_LIM),
                          density=True, weights=w)
    c = 0.5 * (e[:-1] + e[1:])
    gap = h_w[np.abs(c) < 0.2]
    barrier = float(np.log(h_w.max() / max(gap.min(), 1e-300)))
    log(f"barrier estimate = {barrier:.2f} kT (weighted histogram)")

    np.savez(ARTIFACTS / "phi4_reference.npz",
             m_trace=m_trace.astype(np.float32),
             samples_t1=samples_t1.astype(np.float32),
             barrier=barrier, walkers=WALKERS, keeps=KEEPS)
    log(f"DONE — phi4_reference.npz written "
        f"(m_trace {m_trace.shape[0]}, samples {samples_t1.shape})")


if __name__ == "__main__":
    main()
