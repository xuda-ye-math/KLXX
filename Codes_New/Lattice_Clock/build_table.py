"""Build the paired per-stage ESS table from ``train.py`` artifacts.

The clock comparison is defined by selected-proposal validation ESS over the
complete validation set at every accepted forward KL stage. KLXX must use the
same accepted stage schedule stored in ``t_hist``. The aggregate
reported by the experiment is the propagation factor

    F_hat_Sigma = sum_{k=0}^{K} product_{j=k+1}^{K} ESS_j**(-1/2),

with the empty product equal to one, so that the summand at k = K is one and
F_hat_Sigma >= K + 1. This is not the direct ESS of a composed map. This
script implements that reporting contract and writes a centered Markdown table
in the paper layout, plus machine-readable CSV files.

Run from the repository root after all paired runs finish::

    python Codes/Lattice_Clock/build_table.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
SCHEDULE_CONTRACT = "paired_kl_t_hist_v1"
MIN_BATCH_SIZE = 250
METHOD_LABELS = {
    "kl": "forward KL",
    "klxx": (
        r"KL+$\mathrm{X}_\mu$+"
        r"$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$"
    ),
}


@dataclass(frozen=True)
class Run:
    batch_size: int
    method: str
    t_hist: np.ndarray
    ess: np.ndarray

    @property
    def factor(self) -> float:
        """Propagation factor ``sum_k prod_{j>k} ESS_j**(-1/2)``.

        Accumulated over the accepted stages from the last one backwards, so
        that the running product carries the suffix ``prod_{j=k+1}^{K}`` and
        the ``k = K`` summand is the empty product.
        """
        total = 1.0
        suffix = 1.0
        for ess in self.ess[::-1]:
            suffix /= float(np.sqrt(ess))
            total += suffix
        return total

    @property
    def geometric_mean(self) -> float:
        return float(np.exp(np.mean(np.log(self.ess))))


def _load(path: Path) -> Run:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "batch_size",
            "method",
            "complete",
            "schedule_contract",
            "t_hist",
            "valid_selected_ess",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        if str(data["schedule_contract"]) != SCHEDULE_CONTRACT:
            raise ValueError(f"{path}: incompatible schedule contract")
        if not bool(data["complete"]):
            raise ValueError(f"{path}: incomplete stage schedule")
        method = str(data["method"])
        if method not in METHOD_LABELS:
            raise ValueError(f"{path}: unknown method {method!r}")
        t_hist = np.asarray(data["t_hist"], dtype=np.float64)
        ess = np.asarray(data["valid_selected_ess"], dtype=np.float64)
        batch_size = int(data["batch_size"])

    if t_hist.ndim != 1 or ess.ndim != 1 or len(t_hist) != len(ess):
        raise ValueError(f"{path}: t_hist/ESS shape mismatch")
    if not len(t_hist) or t_hist[-1] != 1.0:
        raise ValueError(f"{path}: accepted stage schedule does not end at t=1")
    if not np.all(np.isfinite(ess)) or np.any(ess <= 0.0) or np.any(ess > 1.0):
        raise ValueError(f"{path}: invalid validation ESS")
    return Run(batch_size, method, t_hist, ess)


def load_pairs() -> list[tuple[Run, Run]]:
    runs: dict[tuple[int, str], Run] = {}
    for method in METHOD_LABELS:
        for path in sorted(ARTIFACTS.glob(f"{method}_B*/data.npz")):
            run = _load(path)
            key = (run.batch_size, run.method)
            if key in runs:
                raise ValueError(
                    f"duplicate artifact for B={run.batch_size}, {run.method}"
                )
            runs[key] = run

    if not runs:
        raise FileNotFoundError(f"no completed run artifacts below {ARTIFACTS}")

    batch_sizes = sorted({batch for batch, _ in runs}, reverse=True)
    pairs: list[tuple[Run, Run]] = []
    for batch_size in batch_sizes:
        missing = [m for m in METHOD_LABELS if (batch_size, m) not in runs]
        if missing:
            raise ValueError(
                f"B={batch_size}: paired artifacts incomplete; missing {missing}"
            )
        kl = runs[(batch_size, "kl")]
        klxx = runs[(batch_size, "klxx")]
        if not np.array_equal(kl.t_hist, klxx.t_hist):
            raise ValueError(f"B={batch_size}: forward KL and KLXX t_hist differ")
        pairs.append((kl, klxx))
    return pairs


def _fmt_ess(value: float, better: bool) -> str:
    text = f"{value:.3f}"
    return f"**{text}**" if better else text


def _fmt_factor(value: float, better: bool) -> str:
    text = f"{value:.1f}"
    return f"**{text}**" if better else text


def _markdown_table(pairs: list[tuple[Run, Run]]) -> str:
    max_stages = max(len(kl.t_hist) for kl, _ in pairs)
    stage_headers = [str(k) for k in range(1, max_stages + 1)]
    lines = [
        '<div align="center">',
        "",
        "| stage $k$ | " + " | ".join(stage_headers) + r" | $\hat F_\Sigma$ |",
        "| :--- | " + " | ".join([":-:"] * (max_stages + 1)) + " |",
    ]
    for kl, klxx in pairs:
        pad = ["—"] * (max_stages - len(kl.t_hist))
        t_cells = [f"{t:.3f}" for t in kl.t_hist] + pad
        lines.append(
            f"| $t_k$ ($B = {kl.batch_size}$) | "
            + " | ".join(t_cells)
            + " | — |"
        )

        kl_cells: list[str] = []
        klxx_cells: list[str] = []
        for ess_kl, ess_klxx in zip(kl.ess, klxx.ess):
            kl_cells.append(_fmt_ess(ess_kl, ess_kl > ess_klxx))
            klxx_cells.append(_fmt_ess(ess_klxx, ess_klxx > ess_kl))
        kl_cells.extend(pad)
        klxx_cells.extend(pad)
        kl_f_better = kl.factor < klxx.factor
        klxx_f_better = klxx.factor < kl.factor
        lines.append(
            f"| {METHOD_LABELS['kl']} | "
            + " | ".join(kl_cells)
            + f" | {_fmt_factor(kl.factor, kl_f_better)} |"
        )
        lines.append(
            f"| {METHOD_LABELS['klxx']} | "
            + " | ".join(klxx_cells)
            + f" | {_fmt_factor(klxx.factor, klxx_f_better)} |"
        )
    lines.extend(["", "</div>", ""])
    return "\n".join(lines)


def _write_csv(pairs: list[tuple[Run, Run]]) -> None:
    with (RESULTS / "per_level_ess.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["batch_size", "stage", "t", "kl_ess", "klxx_ess"])
        for kl, klxx in pairs:
            for stage, (t, ess_kl, ess_klxx) in enumerate(
                zip(kl.t_hist, kl.ess, klxx.ess), start=1
            ):
                writer.writerow(
                    [kl.batch_size, stage, f"{t:.9g}", f"{ess_kl:.9g}",
                     f"{ess_klxx:.9g}"]
                )

    with (RESULTS / "factors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["batch_size", "method", "stages", "geometric_mean_ess", "F_hat_sigma"]
        )
        for kl, klxx in pairs:
            for run in (kl, klxx):
                writer.writerow(
                    [run.batch_size, run.method, len(run.ess),
                     f"{run.geometric_mean:.9g}", f"{run.factor:.9g}"]
                )


def main() -> None:
    pairs = [
        pair for pair in load_pairs()
        if pair[0].batch_size >= MIN_BATCH_SIZE
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = _markdown_table(pairs)
    (RESULTS / "tables.md").write_text(
        "# p-state clock — per-stage ESS on a shared stage schedule\n\n"
        "The final column is the propagation factor "
        "$\\hat F_\\Sigma = \\sum_{k=0}^{K} \\prod_{j=k+1}^{K} "
        "\\mathrm{ESS}_j^{-1/2}$; smaller is better. It "
        "summarizes stagewise weight degeneracy and is not a full-chain "
        "ESS or endpoint error estimate. Every ESS is the selected-proposal "
        "validation ESS over the complete validation set.\n\n"
        + table
    )
    _write_csv(pairs)

    print(f"wrote {RESULTS / 'tables.md'} and CSV data")
    for kl, klxx in pairs:
        print(
            f"B={kl.batch_size}: stages={len(kl.ess)} "
            f"forward KL F_hat_sigma={kl.factor:.1f} "
            f"(GM={kl.geometric_mean:.3f}) | "
            f"KLXX F_hat_sigma={klxx.factor:.1f} "
            f"(GM={klxx.geometric_mean:.3f})"
        )


if __name__ == "__main__":
    main()
