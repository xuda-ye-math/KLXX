"""Occupancy-bias scaling figure for the B=2000 staged samplers — rendered
purely from ``artifacts/occupancy_bias_B2000/data.npz``; no GPU and no
recomputation. Style follows the X-regularization paper's
occupancy-bias figure: log-log, +-2 standard-error bars, dashed N^{-1/2}
Monte Carlo reference anchored at the first level.

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    python Codes/Lattice_Clock/plot_occupancy_bias_B2000.py
Writes ``results/occupancy_bias_B2000.png``.
"""

import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
ANALYSIS = ARTIFACTS / "occupancy_bias_B2000"
DATA = ANALYSIS / "data.npz"

METHODS = ("kl", "klxx")
MAX_REPORT_K = 6
LABEL = {"kl": "forward KL",
         "klxx": r"KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$"}
COLOR = {"kl": "tab:blue", "klxx": "tab:red"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    with np.load(DATA, allow_pickle=False) as data:
        d = dict(data)
    ks = [int(k) for k in d["ks"] if int(k) <= MAX_REPORT_K]
    n_base = int(d["BASE_SZIE"])
    Ns = np.array([n_base * 2 ** k for k in ks], dtype=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "mathtext.fontset": "cm", "font.family": "serif",
    })

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
    for m in METHODS:
        bias = np.array([d[f"bias_{m}_k{k}"].mean() for k in ks])
        sem = np.array([d[f"bias_{m}_k{k}"].std(ddof=1)
                        / np.sqrt(len(d[f"bias_{m}_k{k}"]))
                        if len(d[f"bias_{m}_k{k}"]) > 1 else 0.0 for k in ks])
        ax.errorbar(Ns, bias, yerr=2 * sem, marker="o", ms=4, lw=1.2,
                    capsize=3, color=COLOR[m], label=LABEL[m])
        slope = np.polyfit(np.log(Ns), np.log(bias), 1)[0]
        log(f"{m}: biases={['%.5f' % b for b in bias]} slope={slope:.3f}")
    b0 = d["bias_klxx_k0"].mean()
    ax.plot(Ns, b0 * (Ns / Ns[0]) ** -0.5, ls="--", lw=1.0, color="gray",
            label=r"$N^{-1/2}$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(Ns))
    ax.xaxis.set_major_formatter(FixedFormatter([
        r"$10^4$", r"$2\!\times\!10^4$", r"$4\!\times\!10^4$",
        r"$8\!\times\!10^4$", r"$1.6\!\times\!10^5$",
        r"$3.2\!\times\!10^5$", r"$6.4\!\times\!10^5$",
    ]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([2e-3, 5e-3, 1e-2, 2e-2, 4e-2]))
    ax.yaxis.set_major_formatter(FixedFormatter([
        r"$2\!\times\!10^{-3}$", r"$5\!\times\!10^{-3}$", r"$10^{-2}$",
        r"$2\!\times\!10^{-2}$", r"$4\!\times\!10^{-2}$",
    ]))
    ax.yaxis.set_minor_locator(NullLocator())
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor", fontsize=7.5)
    ax.set_xlabel(r"particle count $N$")
    ax.set_ylabel("occupancy bias")
    ax.legend(fontsize=8)
    plt.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "occupancy_bias_B2000.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"DONE — saved {out}")


if __name__ == "__main__":
    main()
