"""KL + lambda X_pi against KLXX at d = 100: sample ESS over lambda, and batch ESS curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
CURVE_LAMBDAS = (1.0, 2.0, 4.0, 8.0)
CURVE_COLORS = ("#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E")
KLXX_COLOR = "#D62728"
KLXX_LABEL = r"KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    }
)


def load(tag, slug):
    return np.load(ARTIFACTS / tag / f"{slug}.npz")


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


klxx = load("klxx", "klxx")
klxx_ess = float(klxx["final_ess"])


def plot_lambda(ax):
    ess = [float(load(f"lambda_{value:.1f}", "klx")["final_ess"]) for value in LAMBDAS]
    ax.plot(
        LAMBDAS, ess, color="#1F77B4", marker="o", markersize=3.5, linewidth=1.5,
        label=r"KL+$\lambda\,\mathrm{X}_\pi$",
    )
    ax.axhline(klxx_ess, color=KLXX_COLOR, linewidth=1.5, label=KLXX_LABEL)
    ax.set_title(r"(a) Sample ESS at $d=100$", loc="left")
    ax.set_xlabel(r"coefficient $\lambda$")
    ax.set_ylabel("sample ESS")
    ax.set_xticks(range(0, 11))
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="lower left", framealpha=0.9)


def plot_training(ax):
    series = [(f"lambda_{value:.1f}", "klx", r"KL+$%d\,\mathrm{X}_\pi$" % value, color, "-")
              for value, color in zip(CURVE_LAMBDAS, CURVE_COLORS)]
    series.append(("klxx", "klxx", KLXX_LABEL, KLXX_COLOR, "-"))
    for tag, slug, label, color, style in series:
        ess = np.asarray(load(tag, slug)["batch_ess_history"], dtype=float)
        steps = np.arange(len(ess))
        window = len(ess) // 50
        ax.plot(steps, ess, color=color, linewidth=0.3, alpha=0.15)
        average = moving_average(ess, window)
        ax.plot(
            steps[window - 1 : window - 1 + len(average)], average,
            color=color, linewidth=1.1, linestyle=style, label=label,
        )
    ax.set_title(r"(b) Batch ESS at $d=100$", loc="left")
    ax.set_xlabel("step")
    ax.set_ylabel("batch ESS")
    ax.set_xlim(0, len(ess))
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")


RESULTS.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": (5.0, 4.6)})
plot_lambda(axes[0])
plot_training(axes[1])
fig.tight_layout()
fig.savefig(RESULTS / "ess.png", dpi=400, bbox_inches="tight")
plt.close(fig)

rows = []
for value in sorted(float(p.name.split("_")[1]) for p in ARTIFACTS.glob("lambda_*")):
    data = load(f"lambda_{value:.1f}", "klx")
    history = np.asarray(data["batch_ess_history"], dtype=float)
    rows.append((f"{value:.1f}", float(data["final_ess"]), float(history[-100:].mean())))
history = np.asarray(klxx["batch_ess_history"], dtype=float)
rows.append(("KLXX", klxx_ess, float(history[-100:].mean())))
lines = [
    "# KL + lambda X_pi at d = 100",
    "",
    "| lambda | sample ESS (EVAL_SIZE) | batch ESS, mean of the last 100 steps |",
    "|---:|---:|---:|",
]
lines += [f"| {c} | {s:.4f} | {b:.4f} |" for c, s, b in rows]
(RESULTS / "tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
