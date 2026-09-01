"""2D benchmark — FAB (flow annealed importance sampling bootstrap).

A baseline for the four KL-based objectives of ``train.py`` in this folder.
Every shared setting — source, target, flow architecture, validation and batch
size, step count, learning rate, MALA parameters and the quench and temper
constants — is imported from that ``train.py`` rather than restated here, so
the two runs differ only in the objective and cannot drift apart.

FAB minimizes the alpha-divergence at alpha = 2, whose gradient is

    grad D_{alpha=2}(pi || nu)  ~  - E_{pi^2/nu} [ grad log nu(y) ] ,

so the loss is a self-normalized weighted log-likelihood over samples drawn
from ``pi^2/nu`` (Midgley et al., 2022). Those samples come from annealed
importance sampling along the geometric path from the flow to that target,

    pi_beta ~ nu^{1-beta} (pi^2/nu)^beta ,

equivalently  log pi_beta = log nu + 2 beta z  with  z = log(pi/nu), so

    U_beta(y) = (2 beta - 1) log nu(y) + 2 beta U(y) .

At beta = 0 this is the flow, at beta = 1/2 it is exactly the target pi, and
at beta = 1 it is pi^2/nu. The ladder is the two levels of `BETAS`: one leg
nu -> pi, then one leg pi -> pi^2/nu.

The AIS is written out here rather than taken from ``jflows``: the packed
``annealed_importance_sampling`` rejuvenates under the target rather than
under the exact intermediate, which is adequate for a target surrogate but
not for the alpha-2 estimator, whose weights must match the path they were
accumulated along. Every level here rejuvenates under its own ``U_beta`` with
MALA, so the incremental weights and the kernels agree.

The replay buffer follows the reference implementation: entries hold
``(y, log_w)``, the store is a circular array overwritten in place, and a
minibatch is drawn with probability proportional to ``rank^{-temperature}``
where ``rank`` counts how many additions ago the entry arrived, so recent
entries are favored and ``temperature = 0`` gives uniform sampling. Stored
log-weights are used as written and are not recomputed on retrieval.

Run from the repo root, e.g.
    python Codes/2D_Benchmark/Himmelblau/train_fab.py
Writes ``artifacts/data_fab.npz`` and ``artifacts/train_fab.log`` and no
figures: ``result.py`` renders every figure, reading that file alongside
``data.npz``. Quench and temper plays no part in FAB training; it is rebuilt
from the same keys as ``train.py`` solely to supply the coverage reference, so
the coverage numbers of the two drivers are measured against the same set.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows.flow import NSF
from jflows.potential import potential_from
from jflows.utils import (
    compute_ESS_log,
    coverage,
    importance_weights_log,
    langevin,
    quench_and_temper,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "train_fab.log"
DATA = ARTIFACTS / "data_fab.npz"


def _load_benchmark():
    """Import this folder's train.py for its constants, source and target.

    train.py guards training behind ``if __name__ == "__main__"``, so the
    import defines the potentials and the parameter block and runs nothing.
    Importing rather than copying is deliberate: it makes it impossible for
    the FAB baseline and the KL runs to disagree on a shared parameter.
    """
    spec = importlib.util.spec_from_file_location("_bench_train", HERE / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_train"] = module
    spec.loader.exec_module(module)
    return module


B = _load_benchmark()

u0, u1 = B.u0, B.u1
NSF_LIM, BINS, TRANSFORMS, HIDDEN_FEATURES = (
    B.NSF_LIM, B.BINS, B.TRANSFORMS, B.HIDDEN_FEATURES)
VALID_SZIE, BATCH_SZIE, TRAIN_STEPS, LR = (
    B.VALID_SZIE, B.BATCH_SZIE, B.TRAIN_STEPS, B.LR)
MC_DT, MC_STEPS = B.MC_DT, B.MC_STEPS
REFER_SZIE, MELT, OPT_DT, OPT_STEPS, QT_MC_STEPS, COVERAGE_K = (
    B.REFER_SZIE, B.MELT, B.OPT_DT, B.OPT_STEPS, B.QT_MC_STEPS, B.COVERAGE_K)
PLT_LIM, GRID_SIZE, PRIOR_SIZE = B.PLT_LIM, B.GRID_SIZE, B.PRIOR_SIZE

# ── FAB-specific ────────────────────────────────────────────────────────
BETAS = (0.5, 1.0)          # the two AIS levels: nu -> pi, then pi -> pi^2/nu
BUFFER_LENGTH: int = 20000  # replay buffer capacity, in samples
BUFFER_TEMP: float = 1.0    # recency exponent; 0 gives uniform sampling
BUFFER_MIN: int = 4000      # buffer is filled to this size before it is drawn from

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8   # same Adam constants as jflows.train


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


# ── flow densities ──────────────────────────────────────────────────────
def log_nu(flow, y):
    """log nu(y) for nu = (flow.inv)_# mu_0, up to mu_0's normalizer.

    The flow maps target -> source (type 'G' in jflows), so with x = G(y)
    and ladj = log|det J_G(y)|,  nu(y) = mu_0(G(y)) |det J_G(y)| .
    """
    x, ladj = flow.call_and_ladj(y)
    return -u0(x) + ladj


def log_ratio(flow, y):
    """z(y) = log pi(y) - log nu(y), up to an additive constant."""
    return -u1(y) - log_nu(flow, y)


def beta_potential(flow, beta: float):
    """U_beta(y) = (2 beta - 1) log nu(y) + 2 beta U(y)."""
    flow = jax.lax.stop_gradient(flow)

    def energy(y):
        return (2.0 * beta - 1.0) * log_nu(flow, y) + 2.0 * beta * u1(y)

    return potential_from(energy)


# ── annealed importance sampling along nu -> pi -> pi^2/nu ──────────────
@eqx.filter_jit
def ais(key, flow, x_source):
    """Return (y, log_w): samples targeting pi^2/nu and their AIS log weights."""
    flow = jax.lax.stop_gradient(flow)
    y = flow.inv(x_source)
    log_w = jnp.zeros(y.shape[0])

    beta_prev = 0.0
    for m, beta in enumerate(BETAS, start=1):
        # incremental AIS weight for the move from pi_{beta_prev} to pi_{beta}
        log_w = log_w + 2.0 * (beta - beta_prev) * log_ratio(flow, y)
        y = langevin(jax.random.fold_in(key, m), y, beta_potential(flow, beta),
                     dt=MC_DT, steps=MC_STEPS, adjust=True)
        beta_prev = beta

    return jax.lax.stop_gradient(y), jax.lax.stop_gradient(log_w)


# ── replay buffer (circular store, recency-prioritized draws) ───────────
class Buffer:
    def __init__(self, max_length: int, dim: int):
        self.max_length = max_length
        self.y = np.zeros((max_length, dim), dtype=np.float64)
        self.log_w = np.zeros(max_length, dtype=np.float64)
        self.add_count = np.zeros(max_length, dtype=np.int64)
        self.index = 0
        self.filled = 0
        self.current_add = 0

    def add(self, y, log_w):
        n = y.shape[0]
        idx = (np.arange(n) + self.index) % self.max_length
        self.y[idx] = np.asarray(y, dtype=np.float64)
        self.log_w[idx] = np.asarray(log_w, dtype=np.float64)
        self.current_add += 1
        self.add_count[idx] = self.current_add
        self.index = (self.index + n) % self.max_length
        self.filled = min(self.filled + n, self.max_length)

    def sample(self, rng, batch_size: int):
        m = self.filled
        rank = self.current_add - self.add_count[:m] + 1
        probs = rank.astype(np.float64) ** (-BUFFER_TEMP)
        probs /= probs.sum()
        take = min(batch_size, m)
        idx = rng.choice(m, size=take, replace=False, p=probs)
        return jnp.asarray(self.y[idx]), jnp.asarray(self.log_w[idx])


# ── FAB loss: - sum_i softmax(log_w)_i log nu(y_i) ──────────────────────
@eqx.filter_jit
def loss_and_grad(flow, y, log_w):
    def loss_fn(f):
        return -jnp.sum(jax.nn.softmax(log_w) * log_nu(f, y))

    return eqx.filter_value_and_grad(loss_fn)(flow)


# ── Adam, matching the constants of jflows.train (optax is not installed
# and the packed optimizer is private) ─────────────────────────────────
@eqx.filter_jit
def adam_step(params, first, second, grads, t):
    first = jax.tree.map(lambda m, g: BETA1 * m + (1 - BETA1) * g, first, grads)
    second = jax.tree.map(lambda v, g: BETA2 * v + (1 - BETA2) * g * g, second, grads)
    m_hat = jax.tree.map(lambda m: m / (1 - BETA1 ** t), first)
    v_hat = jax.tree.map(lambda v: v / (1 - BETA2 ** t), second)
    params = jax.tree.map(lambda p, m, v: p - LR * m / (jnp.sqrt(v) + EPS),
                          params, m_hat, v_hat)
    return params, first, second


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()
    log(f"START {HERE.name} FAB | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"VALID_SZIE={VALID_SZIE} BATCH_SZIE={BATCH_SZIE} TRAIN_STEPS={TRAIN_STEPS} LR={LR} "
        f"MC={MC_DT}x{MC_STEPS} (MALA) | AIS betas={BETAS} | "
        f"buffer={BUFFER_LENGTH} temp={BUFFER_TEMP} min={BUFFER_MIN} | "
        f"shared settings imported from {HERE / 'train.py'}")

    x_valid = u0.samples(jax.random.key(2), VALID_SZIE)

    log("building the quench and temper coverage reference (evaluation only) ...")
    y_hat_ref = quench_and_temper(
        jax.random.key(1), u0.samples(jax.random.key(4), REFER_SZIE), u1,
        melt=MELT, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
        mc_dt=MC_DT, mc_steps=QT_MC_STEPS, mc_adjust=True,
    )
    y_hat_ref = jax.block_until_ready(y_hat_ref)
    log(f"QT reference set ready ({REFER_SZIE} samples)")

    flow = NSF(jax.random.key(0), a=[-NSF_LIM, -NSF_LIM], b=[NSF_LIM, NSF_LIM],
               bins=BINS, transforms=TRANSFORMS,
               hidden_features=HIDDEN_FEATURES).zeros()

    xs = np.linspace(-PLT_LIM, PLT_LIM, GRID_SIZE)
    X1, X2 = np.meshgrid(xs, xs, indexing="xy")
    grid = jnp.stack([jnp.asarray(X1.ravel()), jnp.asarray(X2.ravel())], axis=-1)
    store = {"X1": X1, "X2": X2,
             "U_grid": np.asarray(u1(grid)).reshape(X1.shape),
             "prior": np.asarray(u0.samples(jax.random.key(42), PRIOR_SIZE))}
    np.savez(DATA, **store)
    log(f"wrote the energy grid and source samples to {DATA}")

    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    buffer = Buffer(BUFFER_LENGTH, x_valid.shape[1])
    rng = np.random.default_rng(0)

    ess_hist = []
    t0 = time.time()
    for step in range(TRAIN_STEPS):
        k_batch, k_ais = jax.random.split(jax.random.fold_in(jax.random.key(7), step))
        idx = jax.random.choice(k_batch, VALID_SZIE, (BATCH_SZIE,), replace=False)
        x_batch = x_valid[idx]

        if step == 0:
            log("first AIS call: tracing and compiling the two-level sweep ...")
            t_c = time.time()
        y_ais, log_w_ais = ais(k_ais, flow, x_batch)
        y_ais = jax.block_until_ready(y_ais)
        if step == 0:
            log(f"first AIS call returned in {time.time() - t_c:.1f}s")
        buffer.add(y_ais, log_w_ais)

        if buffer.filled >= BUFFER_MIN:
            y_b, log_w_b = buffer.sample(rng, BATCH_SZIE)
        else:
            y_b, log_w_b = y_ais, log_w_ais

        _, grads = loss_and_grad(flow, y_b, log_w_b)
        grads = eqx.filter(grads, eqx.is_inexact_array)
        params, first, second = adam_step(params, first, second, grads, step + 1)
        flow = eqx.combine(params, static)

        batch_ess = float(compute_ESS_log(
            importance_weights_log(x_batch, u0, u1, flow, type="G")))
        ess_hist.append(batch_ess)
        if step < 5 or (step + 1) % 10 == 0:
            log(f"[FAB] step {step + 1:>5}/{TRAIN_STEPS}  batch ESS = {batch_ess:.4f}  "
                f"buffer = {buffer.filled}  {time.time() - t0:.0f}s")

    flow = jax.block_until_ready(flow)
    jax.effects_barrier()

    y_push = flow.inv(x_valid)
    log_weights = importance_weights_log(x_valid, u0, u1, flow, type="G")
    ess = float(compute_ESS_log(log_weights))
    cov = float(coverage(y_push, y_hat_ref, k=COVERAGE_K, chunks=10))
    store["samples_FAB"] = np.asarray(y_push)
    store["coverage_FAB"] = np.asarray(cov)
    store["ess_hist_FAB"] = np.asarray(ess_hist)
    store["final_ess_FAB"] = np.asarray(ess)
    np.savez(DATA, **store)
    log(f"[FAB] done in {time.time() - t0:.1f}s   final ESS = {ess:.4f}   "
        f"coverage_k={COVERAGE_K} = {cov:.4f}   (VALID_SZIE = {VALID_SZIE})")
    log(f"DONE — data at {DATA}")


if __name__ == "__main__":
    main()
