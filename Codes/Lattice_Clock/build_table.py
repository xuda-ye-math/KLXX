"""Build the results table from the run artifacts written by train.py.

Reads every ``artifacts/<tag>/data.npz`` and writes ``results/tables.csv``
and ``results/tables.md`` with one row per run: tag, dimensions, method, batch size,
schedule (stage count, completeness, the accepted t_k sequence), the composed-
generator final ESS, sector coverage / TV / mean |m| of the pushforward, kNN
coverage against the QT referee set, and the wall time.

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    python Codes/Lattice_Clock/build_table.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"

HDR = ["tag", "D", "P", "method", "batch_size", "train_steps",
       "stage_count", "complete", "t_list",
       "final_ess", "sectors", "abs_m", "knn", "wall_min"]


def load_rows():
    rows = []
    paths = [*ARTIFACTS.glob("kl_B*/data.npz"),
             *ARTIFACTS.glob("klxx_B*/data.npz")]
    for path in sorted(paths):
        d = np.load(path, allow_pickle=False)
        ts = np.asarray(d["t_list"], dtype=float)
        rows.append({
            "tag": str(d["tag"]), "D": int(d["D"]), "P": int(d["P"]),
            "method": str(d["method"]),
            "batch_size": int(d["batch_size"]),
            "train_steps": int(d["train_steps"]),
            "stage_count": len(ts),
            "complete": bool(d["complete"]),
            "t_list": " ".join(f"{t:.3f}" for t in ts),
            "final_ess": float(d["final_ess"]),
            "sectors": f"{int(d['sectors_push'])}/{int(d['P'])}",
            "abs_m": float(d["abs_m_push"]),
            "knn": float(d["knn_coverage"]),
            "wall_min": float(d["wall_s"]) / 60.0,
        })
    return rows


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main() -> None:
    rows = load_rows()
    assert rows, f"no run data found below {ARTIFACTS} — run train.py first"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "tables.csv", "w") as f:
        f.write(",".join(HDR) + "\n")
        for r in rows:
            f.write(",".join(fmt(r[h]) for h in HDR) + "\n")
    with open(RESULTS / "tables.md", "w") as f:
        f.write("# p-state clock — Boltzmann generator results\n\n")
        f.write("| " + " | ".join(HDR) + " |\n")
        f.write("|" + "---|" * len(HDR) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[h]) for h in HDR) + " |\n")
    print(f"wrote {RESULTS / 'tables.md'} (+ csv), {len(rows)} rows")
    for r in rows:
        print(f"  {r['tag']}: stages={r['stage_count']} complete={r['complete']} "
              f"final_ess={r['final_ess']:.4f} sectors={r['sectors']} "
              f"knn={r['knn']:.3f}")


if __name__ == "__main__":
    main()
