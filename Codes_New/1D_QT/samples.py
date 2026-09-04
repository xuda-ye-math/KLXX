"""1D quench and temper illustration — target density on a grid.

The target is the 1D Rastrigin potential

    U(x) = 1/2 x^2 + 4 cos(2 pi x),

whose quadratic term confines the mass while the cosine splits it into wells
spaced one unit apart, centred near x = k + 1/2. The wells deepen toward the
origin, so the target distribution pi(x) proportional to exp(-U(x)) is
multimodal with well populations that fall off away from x = 0. This is the
setting in which quench and temper is illustrated.

This driver evaluates U on a grid, normalizes exp(-U) by the trapezoidal rule,
and stores both. It performs no training and draws no samples.

Run from the repo root:
    python Codes_New/1D_QT/samples.py
Writes ``artifacts/data.npz`` (grid, potential, normalized target density) and
``artifacts/samples.log``. Render the figure from that file with ``result.py``.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import Nlog_Gaussian, potential_from
from jflows.utils import quench_and_temper

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "samples.log"
DATA = ARTIFACTS / "data.npz"

PLT_LIM: float = 5.0       # half-width of the grid stored for the figures
GRID_SIZE: int = 2001      # grid points; the wells have unit period
NORM_LIM: float = 12.0     # half-width of the wide grid the normalizers use
NORM_SIZE: int = 24001     # grid points of that wide grid
SIGMA: float = 1.0         # standard deviation of the Gaussian source pi_0

# quench and temper (the wide-coverage measure hat_pi)
QT_SIZE: int = 100000      # source samples fed to quench and temper
MELT: float = 2.0          # melt scale (std of the Gaussian scatter)
OPT_ALPHA: float = 0.5     # L-BFGS initial trial step size (Armijo)
OPT_STEPS: int = 100       # L-BFGS iterations of the quench
MC_DT: float = 1e-3        # MALA step size of the temper
MC_STEPS_2: int = 100      # MALA steps of the temper
COEFF_QT: float = 0.0      # energy-weighted resampling and second temper; 0 disables it
QT_BINS: int = 300         # histogram bins used to draw hat_pi over [-PLT_LIM, PLT_LIM]


# target: 1D Rastrigin potential U(x) = 1/2 x^2 + 4 cos(2 pi x)
def rastrigin_energy(x):
    x1 = x[..., 0]
    return 0.5 * x1**2 + 4.0 * jnp.cos(2.0 * jnp.pi * x1)


u1 = potential_from(rastrigin_energy)

# source: Gaussian pi_0 centred at the origin
u0 = Nlog_Gaussian(mean=[0.0], variance=[SIGMA**2])


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def unnormalized_on(u, lim: float, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid, potential, and shifted density exp(-(U - min U)) of `u` on [-lim, lim]."""
    xs = np.linspace(-lim, lim, size)
    U = np.asarray(u(jnp.asarray(xs)[:, None]))
    return xs, U, np.exp(-(U - U.min()))


def normalized(u, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Density of `u` on the stored grid, normalized on the wide grid.

    Both potentials are defined up to an additive constant, so each density is
    normalized by quadrature rather than by an analytic constant. The stored
    grid is truncated and discrete, so its own normalizer is compared against
    the wide one; a bad limit or too coarse a spacing shows up as disagreement.
    """
    xs, U, p_raw = unnormalized_on(u, PLT_LIM, GRID_SIZE)
    xs_w, _, p_raw_w = unnormalized_on(u, NORM_LIM, NORM_SIZE)
    Z = float(np.trapezoid(p_raw_w, xs_w))
    Z_plt = float(np.trapezoid(p_raw, xs))
    log(f"{name}: U in [{U.min():.4f}, {U.max():.4f}] | normalizer {Z:.6f} on "
        f"[{-NORM_LIM}, {NORM_LIM}] against {Z_plt:.6f} on [{-PLT_LIM}, {PLT_LIM}], "
        f"relative difference {abs(Z_plt - Z) / Z:.3e} | stored-grid mass {Z_plt / Z:.6f}")
    return xs, U, p_raw / Z


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START 1D_QT samples | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"U(x) = 1/2 x^2 + 4 cos(2 pi x) | sigma={SIGMA} | "
        f"PLT_LIM={PLT_LIM} GRID_SIZE={GRID_SIZE}")

    xs, U, pi = normalized(u1, "target pi ")
    _, U0, pi0 = normalized(u0, "source pi_0")

    # quench and temper: melt the source cloud, quench it into the wells of U,
    # then temper it with MALA around those wells
    log(f"quench and temper on {QT_SIZE} source samples | melt={MELT} "
        f"opt={OPT_ALPHA}x{OPT_STEPS} mc={MC_DT}x{MC_STEPS_2} (MALA) coeff_qt={COEFF_QT}")
    x0 = u0.samples(jax.random.key(0), QT_SIZE)
    y_qt = quench_and_temper(
        jax.random.key(1), x0, u1,
        melt=MELT, opt_dt=OPT_ALPHA, opt_steps=OPT_STEPS,
        mc_dt=MC_DT, mc_steps=MC_STEPS_2, mc_adjust=True,
        coeff_qt=COEFF_QT,
    )
    y_qt = np.asarray(jax.block_until_ready(y_qt))
    x0 = np.asarray(x0)
    log(f"QT samples ready | range [{y_qt.min():.4f}, {y_qt.max():.4f}] | "
        f"{np.mean(np.abs(y_qt) > PLT_LIM) * 100:.3f}% outside the stored grid")

    edges = np.linspace(-PLT_LIM, PLT_LIM, QT_BINS + 1)
    qt_density, _ = np.histogram(y_qt[:, 0], bins=edges, density=True)
    qt_x = 0.5 * (edges[:-1] + edges[1:])

    # well occupancy: QT weights a well by basin volume, the target by depth,
    # so the two columns below should differ, most visibly in the outer wells
    log("well occupancy (well centre: QT fraction, target fraction)")
    for centre in np.arange(-4.5, 5.0, 1.0):
        in_well = np.abs(y_qt[:, 0] - centre) < 0.5
        grid_well = np.abs(xs - centre) < 0.5
        log(f"  x = {centre:+.1f}: QT {in_well.mean():.4f}, "
            f"target {float(np.trapezoid(pi[grid_well], xs[grid_well])):.4f}")

    np.savez(DATA, x=xs, U=U, pi=pi, U0=U0, pi0=pi0,
             x0=x0, y_qt=y_qt, qt_x=qt_x, qt_density=qt_density,
             plt_lim=np.asarray(PLT_LIM), sigma=np.asarray(SIGMA))
    log(f"wrote the grid, both potentials, both densities, and the QT "
        f"samples and histogram to {DATA}")


if __name__ == "__main__":
    main()
