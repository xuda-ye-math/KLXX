"""Build the paired per-level ESS table from ``train.py`` artifacts.

The Clock comparison is defined by the full-validation ESS at every accepted
KL level.  KLXX must use the exact same accepted ``t_hist``.  The aggregate
reported by the original experiment is the error-propagation factor

    F = product_k ESS_k**(-1),

not the direct ESS of a composed map.  This script implements that reporting
contract and writes a centered Markdown table in the paper layout, plus
machine-readable CSV files.

Run from the repository root after all paired runs finish::

    source ~/.envs/jflows/bin/activate
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
        """Error-propagation factor ``prod_k 1 / ESS_k``."""
        return float(np.prod(1.0 / self.ess))

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
            raise ValueError(f"{path}: incomplete ladder")
        method = str(data["method"])
        if method not in METHOD_LABELS:
            raise ValueError(f"{path}: unknown method {method!r}")
        t_hist = np.asarray(data["t_hist"], dtype=np.float64)
        ess = np.asarray(data["valid_selected_ess"], dtype=np.float64)
        batch_size = int(data["batch_size"])

    if t_hist.ndim != 1 or ess.ndim != 1 or len(t_hist) != len(ess):
        raise ValueError(f"{path}: t_hist/ESS shape mismatch")
    if not len(t_hist) or t_hist[-1] != 1.0:
        raise ValueError(f"{path}: accepted history does not end at t=1")
    if not np.all(np.isfinite(ess)) or np.any(ess <= 0.0) or np.any(ess > 1.0):
        raise ValueError(f"{path}: invalid validation ESS")
    return Run(batch_size, method, t_hist, ess)


def load_pairs() -> list[tuple[Run, Run]]:
    runs: dict[tuple[int, str], Run] = {}
    for path in sorted(ARTIFACTS.glob("*_B*/data.npz")):
        run = _load(path)
        key = (run.batch_size, run.method)
        if key in runs:
            raise ValueError(f"duplicate artifact for B={run.batch_size}, {run.method}")
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
            raise ValueError(f"B={batch_size}: KL and KLXX t_hist differ")
        pairs.append((kl, klxx))
    return pairs


def _fmt_ess(value: float, better: bool) -> str:
    text = f"{value:.3f}"
    return f"**{text}**" if better else text


def _fmt_factor(value: float, better: bool) -> str:
    text = f"{value:.1f}"
    return f"**{text}**" if better else text


def _markdown_table(pairs: list[tuple[Run, Run]]) -> str:
    max_levels = max(len(kl.t_hist) for kl, _ in pairs)
    stage_headers = [str(k) for k in range(1, max_levels + 1)]
    lines = [
        '<div align="center">',
        "",
        "| stage $k$ | " + " | ".join(stage_headers) + " | $F$ |",
        "| :--- | " + " | ".join([":-:"] * (max_levels + 1)) + " |",
    ]
    for kl, klxx in pairs:
        pad = ["—"] * (max_levels - len(kl.t_hist))
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
        writer = csv.writer(handle)
        writer.writerow(["batch_size", "level", "t", "kl_ess", "klxx_ess"])
        for kl, klxx in pairs:
            for level, (t, ess_kl, ess_klxx) in enumerate(
                zip(kl.t_hist, kl.ess, klxx.ess), start=1
            ):
                writer.writerow(
                    [kl.batch_size, level, f"{t:.9g}", f"{ess_kl:.9g}",
                     f"{ess_klxx:.9g}"]
                )

    with (RESULTS / "factors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_size", "method", "levels", "geometric_mean_ess", "F"])
        for kl, klxx in pairs:
            for run in (kl, klxx):
                writer.writerow(
                    [run.batch_size, run.method, len(run.ess),
                     f"{run.geometric_mean:.9g}", f"{run.factor:.9g}"]
                )


def main() -> None:
    pairs = load_pairs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = _markdown_table(pairs)
    (RESULTS / "tables.md").write_text(
        "# p-state clock — per-level ESS on a shared history\n\n"
        "The final column is the error-propagation factor "
        "$F = \\prod_k \\mathrm{ESS}_k^{-1}$; smaller is better.  Every "
        "ESS is the full-validation stage-gate value.\n\n"
        + table
    )
    _write_csv(pairs)

    print(f"wrote {RESULTS / 'tables.md'} and CSV data")
    for kl, klxx in pairs:
        print(
            f"B={kl.batch_size}: levels={len(kl.ess)} "
            f"KL F={kl.factor:.1f} (GM={kl.geometric_mean:.3f}) | "
            f"KLXX F={klxx.factor:.1f} (GM={klxx.geometric_mean:.3f})"
        )


if __name__ == "__main__":
    main()
