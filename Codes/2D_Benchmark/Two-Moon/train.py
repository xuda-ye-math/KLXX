"""2D Two-Moon benchmark — X-regularized forward KL on the two-moons target.

Four objectives, all trained by the packed jflows drivers from the same
identity-initialized NSF on the same source pool:

    KL                :  train_forward_KLX_G,  coeff_lambda = 0
    KL+X_mu           :  train_forward_KLX_G,  coeff_lambda = 1
    KL+X_mu+X_hat_mu  :  train_forward_KLXX_G, (alpha, beta) = (1, 0)
    KL+X_mu+X_mix     :  train_forward_KLXX_G, (alpha, beta) = (1/2, 1/2)

The target approximates the two-moons density by a Gaussian mixture whose
centers sit evenly along the two interleaved crescent arcs (MOON_K centers
per moon, stroke width MOON_SIGMA) — two disjoint curved components, so the
benchmark stresses the flow's expressivity on distorted ridges and mode
discovery across the gap at once. All Langevin kernels run MALA
(mc_adjust = True). Final ESS is the flow importance-sampling ESS on the
full fixed source set, and coverage (k = 5) is measured against a
quench-and-temper reference pool.

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/data/projects/jflows python \
        Codes/2D_Benchmark/Two-Moon/train.py
Writes final figures below ``results/`` and the temporary log below
``artifacts/``.
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
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture
from jflows.train import Monitor, train_forward_KLX_G, train_forward_KLXX_G
from jflows.utils import (
    compute_ESS_log,
    coverage,
    importance_weights_log,
    quench_and_temper,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
LOG = ARTIFACTS / "train.log"

# boundary of the domain
SIGMA = 1.0            # standard deviation of the isotropic Gaussian source mu_0
PLT_LIM = 4.5          # half-width of the plot window
NSF_LIM = 5.5          # half-width of the NSF spline domain

# NSF flow architecture
BINS: int = 32
TRANSFORMS: int = 6
HIDDEN_FEATURES = (128, 128)

# training parameters
VALID_SZIE: int = 50000   # the fixed source set (training pool + final ESS / coverage)
BATCH_SZIE: int = 200     # source samples per training step
POOL_SIZE: int = 0        # 0: quench the complete validation population
TRAIN_STEPS: int = 500      # Adam optimization steps
LR: float = 2e-3       # Adam learning rate

# data pipeline (single-hop AIS + MALA rejuvenation)
LADDER: int = 1        # AIS levels per manufactured target batch
MC_DT: float = 2e-3  # Langevin step size
MC_STEPS: int = 40     # Langevin steps per level / per hat_mu freshening

# quench and temper (the wide-coverage measure hat_mu)
REFER_SZIE: int = 1000    # coverage-reference population size
MELT: float = 2.0      # melt scale (std of the Gaussian scatter)
OPT_DT: float = 0.5  # L-BFGS initial trial step size (Armijo)
OPT_STEPS: int = 200   # L-BFGS iterations
QT_MC_STEPS: int = 1000  # temper length of the coverage-reference pool

COVERAGE_K: int = 5    # k-NN ball of the coverage metric

# two-moon mixture target: MOON_K Gaussian centers evenly spaced along each
# crescent arc (the two-moons parametrization, vertically stretched and with
# a vertical gap pushing the two components apart)
MOON_K: int = 48            # Gaussian centers per moon
MOON_SIGMA: float = 0.15    # per-mode standard deviation (stroke width)
MOON_SCALE: float = 2.0     # horizontal scale of the unit two-moons pair
MOON_Y_STRETCH: float = 1.4  # vertical stretch of the arcs (distortion)
MOON_Y_SEP: float = 0.25    # vertical half-gap between the components (unit arcs)

_t = np.linspace(0.0, np.pi, MOON_K)
_upper = np.stack([np.cos(_t) - 0.5,
                   MOON_Y_STRETCH * (np.sin(_t) + MOON_Y_SEP)], axis=-1)
_lower = np.stack([0.5 - np.cos(_t),
                   -MOON_Y_STRETCH * (np.sin(_t) + MOON_Y_SEP)], axis=-1)
MODE_MEANS = (MOON_SCALE * np.concatenate([_upper, _lower], axis=0)).tolist()

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


# source: Gaussian mu_0
u0 = Nlog_Gaussian(mean=[0.0, 0.0], variance=[SIGMA**2, SIGMA**2])

# target: two-moon Gaussian mixture
u1 = Nlog_Gaussian_Mixture(
    weights=[1.0] * len(MODE_MEANS),
    mean=MODE_MEANS,
    variance=[[MOON_SIGMA**2, MOON_SIGMA**2]] * len(MODE_MEANS),
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START Two-Moon | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"VALID_SZIE={VALID_SZIE} BATCH_SZIE={BATCH_SZIE} TRAIN_STEPS={TRAIN_STEPS} LR={LR} "
        f"MC={MC_DT}x{MC_STEPS} (MALA) ladder={LADDER} "
        f"QT: REFER_SZIE={REFER_SZIE} melt={MELT} opt={OPT_DT}x{OPT_STEPS} | "
        f"moons: K={MOON_K}/moon sigma={MOON_SIGMA} scale={MOON_SCALE}")
    x_valid = u0.samples(jax.random.key(2), VALID_SZIE)
    flow0 = NSF(jax.random.key(0), a=[-NSF_LIM, -NSF_LIM], b=[NSF_LIM, NSF_LIM],
                bins=BINS, transforms=TRANSFORMS,
                hidden_features=HIDDEN_FEATURES).zeros()

    # coverage-reference pool: quench and temper on the target
    log("building the quench-and-temper coverage reference pool ...")
    y_hat_ref = quench_and_temper(
        jax.random.key(1), u0.samples(jax.random.key(4), REFER_SZIE), u1,
        melt=MELT, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
        mc_dt=MC_DT, mc_steps=QT_MC_STEPS, mc_adjust=True,
    )
    y_hat_ref = jax.block_until_ready(y_hat_ref)
    log(f"QT pool ready ({REFER_SZIE} particles)")

    runs = {}
    for name in METHODS:
        t0 = time.time()
        mon = Monitor(200, f"[{name}] ", log)
        if name == "KL":
            flow, batch_ess_hist = train_forward_KLX_G(
                x_valid, u0, u1, flow0,
                batch_size=BATCH_SZIE, train_steps=TRAIN_STEPS, lr=LR,
                ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS,
                coeff_lambda=0.0, mc_adjust=True, monitor=mon)
        elif name == "KL+X_mu":
            flow, batch_ess_hist = train_forward_KLX_G(
                x_valid, u0, u1, flow0,
                batch_size=BATCH_SZIE, train_steps=TRAIN_STEPS, lr=LR,
                ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS,
                coeff_lambda=1.0, mc_adjust=True, monitor=mon)
        elif name == "KL+X_mu+X_hat_mu":
            flow, batch_ess_hist = train_forward_KLXX_G(
                x_valid, u0, u1, flow0,
                pool_size=POOL_SIZE,
                batch_size=BATCH_SZIE, train_steps=TRAIN_STEPS, lr=LR,
                ladder=LADDER, melt=MELT, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
                mc_dt=MC_DT, mc_steps=MC_STEPS,
                coeff_lambda=1.0, coeff_alpha=1.0, coeff_beta=0.0,
                mc_adjust=True, monitor=mon)
        else:  # KL+X_mu+X_mix
            flow, batch_ess_hist = train_forward_KLXX_G(
                x_valid, u0, u1, flow0,
                pool_size=POOL_SIZE,
                batch_size=BATCH_SZIE, train_steps=TRAIN_STEPS, lr=LR,
                ladder=LADDER, melt=MELT, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
                mc_dt=MC_DT, mc_steps=MC_STEPS,
                coeff_lambda=1.0, coeff_alpha=0.5, coeff_beta=0.5,
                mc_adjust=True, monitor=mon)
        flow = jax.block_until_ready(flow)
        jax.effects_barrier()

        y_push = flow.inv(x_valid)
        log_weights = importance_weights_log(x_valid, u0, u1, flow, type="G")
        ess = float(compute_ESS_log(log_weights))
        cov = float(coverage(y_push, y_hat_ref, k=COVERAGE_K, chunks=10))
        runs[name] = {"samples": np.asarray(y_push),
                      "batch_ess_hist": np.asarray(batch_ess_hist),
                      "final_ess": ess, "coverage": cov}
        log(f"[{name}] done in {time.time() - t0:.1f}s   final ESS = {ess:.4f}   "
            f"coverage_k={COVERAGE_K} = {cov:.4f}   (VALID_SZIE = {VALID_SZIE})")

    # ── samples.png: pushforward panels over the target energy ──
    log("rendering samples.png + ess.png ...")
    n = 300
    xs = np.linspace(-PLT_LIM, PLT_LIM, n)
    X1, X2 = np.meshgrid(xs, xs, indexing="xy")
    grid = jnp.stack([jnp.asarray(X1.ravel()), jnp.asarray(X2.ravel())], axis=-1)
    U_grid = np.asarray(u1(grid)).reshape(X1.shape)
    levels = np.linspace(0.0, 15.0, 50).tolist()
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
    samples_out = RESULTS / "samples.png"
    fig.savefig(samples_out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {samples_out}")

    # ── ess.png: per-step training ESS histories ──
    fig, ax_ess = plt.subplots(1, 1, figsize=(5, 4))
    for name in METHODS:
        ax_ess.plot(runs[name]["batch_ess_hist"], color=METHOD_COLOR[name],
                    label=METHOD_LABEL[name], linewidth=0.6)
    ax_ess.set_xlabel("step"); ax_ess.set_ylabel("ESS")
    ax_ess.set_xlim(0, TRAIN_STEPS); ax_ess.set_ylim(0, 1)
    ax_ess.legend(loc="lower right")
    plt.tight_layout()
    ess_out = RESULTS / "ess.png"
    fig.savefig(ess_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"DONE — figures at {samples_out} and {ess_out}")


if __name__ == "__main__":
    main()
