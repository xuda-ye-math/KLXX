"""2D Threewell benchmark — X-regularized forward KL on a three-well target.

Four objectives, all trained by the packed jflows drivers from the same
identity-initialized NSF on the same source pool:

    KL                :  train_forward_KLX_G,  coeff_lambda = 0
    KL+X_mu           :  train_forward_KLX_G,  coeff_lambda = 1
    KL+X_mu+X_hat_mu  :  train_forward_KLXX_G, (alpha, beta) = (1, 0)
    KL+X_mu+X_mix     :  train_forward_KLXX_G, (alpha, beta) = (1/2, 1/2)

The source is a narrow Gaussian (sigma = 0.25) covering none of the three
wells, so the bare forward KL trains on whatever its AIS chain reaches; the
X_hat_mu / X_mix objectives draw mode discovery from the quench-and-temper
measure. All Langevin kernels run MALA (mc_adjust = True). Final ESS is the
flow importance-sampling ESS on the full fixed source set, and coverage
(k = 5) is measured against a quench-and-temper reference pool.

Run from the repo root:
    conda activate jflows && PYTHONPATH=/mnt/projects/jflows python \
        Codes/2D_Benchmark/Three-Well/train.py
Writes samples.png, ess.png, and train_status.log next to this file.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, potential_from
from jflows.train import Monitor, train_forward_KLX_G, train_forward_KLXX_G
from jflows.utils import compute_ESS, coverage, importance_weights, quench_and_temper

HERE = Path(__file__).resolve().parent
LOG = HERE / "train_status.log"

# boundary of the domain
SIGMA = 0.25           # standard deviation of the isotropic Gaussian source mu_0
PLT_LIM = 2.0          # half-width of the plot window
NSF_LIM = 2.0          # half-width of the NSF spline domain

# NSF flow architecture
BINS: int = 32
TRANSFORMS: int = 6
HIDDEN_FEATURES = (128, 128)

# training parameters
N_VALID: int = 80000  # the fixed source set (training pool + final ESS / coverage)
N_BATCH: int = 1000    # source samples per training step
STEPS: int = 500      # Adam optimization steps
LR: float = 1e-3       # Adam learning rate

# data pipeline (single-hop AIS + MALA rejuvenation)
LADDER: int = 1        # AIS rungs per manufactured target batch
MC_STEP: float = 2e-3  # Langevin step size
MC_ITERS: int = 50     # Langevin steps per rung / per hat_mu freshening

# quench and temper (the wide-coverage measure hat_mu)
N_POOL: int = 4000     # quench-and-temper pool size
MELT: float = 2.0      # melt scale (std of the Gaussian scatter)
OPT_STEP: float = 0.5  # L-BFGS trial alpha (armijo)
OPT_ITERS: int = 200   # L-BFGS iterations
QT_MC_ITERS: int = 1000  # temper length of the coverage-reference pool

COVERAGE_K: int = 5    # k-NN ball of the coverage metric

METHODS = (
    "KL",
    "KL+X_mu",
    "KL+X_mu+X_hat_mu",
    "KL+X_mu+X_mix",
)
METHOD_LABEL = {
    "KL":               "forward KL",
    "KL+X_mu":          r"forward KL+$\mathrm{X}_\mu$",
    "KL+X_mu+X_hat_mu": r"forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$",
    "KL+X_mu+X_mix":    r"forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$",
}
METHOD_COLOR = {
    "KL":               "#1F77B4A0",   # tab:blue
    "KL+X_mu":          "#2CA02CA0",   # tab:green
    "KL+X_mu+X_hat_mu": "#D62728A0",   # tab:red
    "KL+X_mu+X_mix":    "#9467BDA0",   # tab:purple
}

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})
CMAP = LinearSegmentedColormap.from_list("light_yellow_red", ["#fffefa", "#ffe5e5"])


# source: narrow Gaussian mu_0
u0 = Nlog_Gaussian(mean=[0.0, 0.0], variance=[SIGMA**2, SIGMA**2])


# target: Threewell potential U(x) = 6 ((x1^2-1)^2 + (x2^2-1)^2 + sin(x1 + 2 x2))
def threewell_energy(x):
    x1, x2 = x[..., 0], x[..., 1]
    return 6.0 * ((x1**2 - 1.0) ** 2 + (x2**2 - 1.0) ** 2 + jnp.sin(x1 + 2.0 * x2))


u1 = potential_from(threewell_energy)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START Threewell | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"N_VALID={N_VALID} N_BATCH={N_BATCH} STEPS={STEPS} LR={LR} "
        f"MC={MC_STEP}x{MC_ITERS} (MALA) ladder={LADDER} "
        f"QT: N_POOL={N_POOL} melt={MELT} opt={OPT_STEP}x{OPT_ITERS}")
    x_valid = u0.samples(jax.random.key(2), N_VALID)
    flow0 = NSF(jax.random.key(0), a=[-NSF_LIM, -NSF_LIM], b=[NSF_LIM, NSF_LIM],
                bins=BINS, transforms=TRANSFORMS,
                hidden_features=HIDDEN_FEATURES).zeros()

    # coverage-reference pool: quench and temper on the target
    log("building the quench-and-temper coverage reference pool ...")
    y_hat_ref = quench_and_temper(jax.random.key(1), u0.samples(jax.random.key(4), N_POOL),
                                  u1, melt=MELT, opt_step=OPT_STEP, opt_iters=OPT_ITERS,
                                  mc_step=MC_STEP, mc_iters=QT_MC_ITERS)
    y_hat_ref = jax.block_until_ready(y_hat_ref)
    log(f"QT pool ready ({N_POOL} particles)")

    runs = {}
    for name in METHODS:
        t0 = time.time()
        mon = Monitor(100, f"[{name}] ", log)
        if name == "KL":
            flow, ess_hist = train_forward_KLX_G(
                x_valid, u0, u1, flow0, n_batch=N_BATCH, steps=STEPS, lr=LR,
                ladder=LADDER, mc_step=MC_STEP, mc_iters=MC_ITERS,
                coeff_lambda=0.0, monitor=mon)
        elif name == "KL+X_mu":
            flow, ess_hist = train_forward_KLX_G(
                x_valid, u0, u1, flow0, n_batch=N_BATCH, steps=STEPS, lr=LR,
                ladder=LADDER, mc_step=MC_STEP, mc_iters=MC_ITERS,
                coeff_lambda=1.0, monitor=mon)
        elif name == "KL+X_mu+X_hat_mu":
            flow, ess_hist = train_forward_KLXX_G(
                x_valid, u0, u1, flow0, n_pool=N_POOL, n_batch=N_BATCH,
                steps=STEPS, lr=LR, ladder=LADDER, melt=MELT,
                opt_step=OPT_STEP, opt_iters=OPT_ITERS,
                mc_step=MC_STEP, mc_iters=MC_ITERS,
                coeff_lambda=1.0, coeff_alpha=1.0, coeff_beta=0.0, monitor=mon)
        else:  # KL+X_mu+X_mix
            flow, ess_hist = train_forward_KLXX_G(
                x_valid, u0, u1, flow0, n_pool=N_POOL, n_batch=N_BATCH,
                steps=STEPS, lr=LR, ladder=LADDER, melt=MELT,
                opt_step=OPT_STEP, opt_iters=OPT_ITERS,
                mc_step=MC_STEP, mc_iters=MC_ITERS,
                coeff_lambda=1.0, coeff_alpha=0.5, coeff_beta=0.5, monitor=mon)
        flow = jax.block_until_ready(flow)
        jax.effects_barrier()

        y_push = flow.inv(x_valid)
        ess = float(compute_ESS(importance_weights(x_valid, u0, u1, flow, type="G")))
        cov = float(coverage(y_push, y_hat_ref, k=COVERAGE_K, chunk=10))
        runs[name] = {"samples": np.asarray(y_push),
                      "ess_history": np.asarray(ess_hist),
                      "final_ess": ess, "coverage": cov}
        log(f"[{name}] done in {time.time() - t0:.1f}s   final ESS = {ess:.4f}   "
            f"coverage_k={COVERAGE_K} = {cov:.4f}   (N_VALID = {N_VALID})")

    # ── samples.png: pushforward panels over the target energy ──
    log("rendering samples.png + ess.png ...")
    n = 300
    xs = np.linspace(-PLT_LIM, PLT_LIM, n)
    X1, X2 = np.meshgrid(xs, xs, indexing="xy")
    grid = jnp.stack([jnp.asarray(X1.ravel()), jnp.asarray(X2.ravel())], axis=-1)
    U_grid = np.asarray(u1(grid)).reshape(X1.shape)
    levels = np.linspace(-1.0, 8.0, 20).tolist()
    prior_np = np.asarray(u0.samples(jax.random.key(42), 5000))

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for col, name in enumerate(METHODS):
        ax = axes[col]
        ax.contourf(X1, X2, U_grid, levels=levels, cmap=CMAP.reversed(), extend="max")
        ax.contour(X1, X2, U_grid, levels=levels, colors="gray", linewidths=0.2, alpha=0.2)
        ax.scatter(prior_np[:, 0], prior_np[:, 1], s=0.04, alpha=0.3, color="gray", zorder=5)
        s = runs[name]["samples"]
        ax.scatter(s[:, 0], s[:, 1], s=0.04, alpha=0.35, color=METHOD_COLOR[name], zorder=10)
        ax.set_xlim(-PLT_LIM, PLT_LIM); ax.set_ylim(-PLT_LIM, PLT_LIM)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_title(f"{METHOD_LABEL[name]}\n"
                     f"ESS = $\\mathbf{{{runs[name]['final_ess']:.2f}}}$, "
                     f"cvrg = $\\mathbf{{{runs[name]['coverage']:.2f}}}$")
    plt.tight_layout()
    samples_out = HERE / "samples.png"
    fig.savefig(samples_out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {samples_out}")

    # ── ess.png: per-step training ESS histories ──
    fig, ax_ess = plt.subplots(1, 1, figsize=(5, 4))
    for name in METHODS:
        ax_ess.plot(runs[name]["ess_history"], color=METHOD_COLOR[name],
                    label=METHOD_LABEL[name], linewidth=0.6)
    ax_ess.set_xlabel("step"); ax_ess.set_ylabel("ESS")
    ax_ess.set_xlim(0, STEPS); ax_ess.set_ylim(0, 1)
    ax_ess.legend(loc="lower right")
    plt.tight_layout()
    ess_out = HERE / "ess.png"
    fig.savefig(ess_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"DONE — figures at {samples_out} and {ess_out}")


if __name__ == "__main__":
    main()
