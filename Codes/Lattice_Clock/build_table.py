"""Build the results table from the data_*.npz files written by train.py.

Reads every data_<tag>.npz in this folder and writes tables.csv and
tables.md with one row per run: tag, dimensions, method, batch size,
ladder (stage count, completeness, the accepted t_k sequence), the composed-
generator final ESS, sector coverage / TV / mean |m| of the pushforward, kNN
coverage against the QT referee set, and the wall time.

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    python Codes/Lattice_Clock/build_table.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

HDR = ["tag", "D", "P", "method", "B", "steps", "K", "complete", "ladder",
       "final_ess", "sectors", "abs_m", "knn", "wall_min"]


def load_rows():
    rows = []
    for path in sorted(HERE.glob("data_*.npz")):
        d = np.load(path, allow_pickle=False)
        ts = np.asarray(d["ladder"], dtype=float)
        rows.append({
            "tag": str(d["tag"]), "D": int(d["D"]), "P": int(d["P"]),
            "method": str(d["method"]), "B": int(d["n_batch"]),
            "steps": int(d["steps"]), "K": len(ts),
            "complete": bool(d["complete"]),
            "ladder": " ".join(f"{t:.3f}" for t in ts),
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
    assert rows, "no data_*.npz found — run train.py first"
    with open(HERE / "tables.csv", "w") as f:
        f.write(",".join(HDR) + "\n")
        for r in rows:
            f.write(",".join(fmt(r[h]) for h in HDR) + "\n")
    with open(HERE / "tables.md", "w") as f:
        f.write("# p-state clock — Boltzmann generator results\n\n")
        f.write("| " + " | ".join(HDR) + " |\n")
        f.write("|" + "---|" * len(HDR) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[h]) for h in HDR) + " |\n")
    print(f"wrote {HERE / 'tables.md'} (+ csv), {len(rows)} rows")
    for r in rows:
        print(f"  {r['tag']}: K={r['K']} complete={r['complete']} "
              f"final_ess={r['final_ess']:.4f} sectors={r['sectors']} "
              f"knn={r['knn']:.3f}")


if __name__ == "__main__":
    main()
