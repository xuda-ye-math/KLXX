"""Build the loss x dimension tables from the saved sweep artifacts.

Reads every ``artifacts/k{k}/data.npz`` written by train.py and writes
``results/ess_table.csv``,
mode_coverage_table.csv, and tables.md:

    Table 1 — final ESS (the headline);
    Table 2 — strict mode coverage: modes_found / 2**k, a mode counting as
              covered only if it holds >= STRICT_FRAC of the uniform
              expected share of the evaluation pool;

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    python Codes/HD_Product/build_table.py
"""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"

STRICT_FRAC = 0.5   # threshold share of N / 2**k for a mode to count as covered

METHODS = (
    "KL",
    "KL+X_mu",
    "KL+X_mu+X_hat_mu",
    "KL+X_mu+X_mix",
)
LABEL = {
    "KL": "forward KL",
    "KL+X_mu": "KL+X_mu",
    "KL+X_mu+X_hat_mu": "KL+X_mu+X_hat_mu",
    "KL+X_mu+X_mix": "KL+X_mu+X_mix",
}


def load():
    found = {}
    for path in sorted(ARTIFACTS.glob("k*/data.npz")):
        d = np.load(path)
        found[int(d["k"])] = d
    return dict(sorted(found.items()))


def cell(data, k, m, field):
    run = data[k]
    if field == "final_ess":
        return float(run[f"final_ess_{m}"])
    counts = run[f"bucket_counts_{m}"].astype(np.float64)
    n, nmodes = counts.sum(), 2 ** k
    if field == "strict_cov":
        return float((counts >= STRICT_FRAC * n / nmodes).sum() / nmodes)
    raise ValueError(field)


def write_table(data, field, fname, fmt="{:.4f}"):
    ks = list(data.keys())
    header = ["loss \\ d"] + [str(int(data[k]["d"])) for k in ks]
    rows = [[LABEL[m]] + [fmt.format(cell(data, k, m, field)) for k in ks]
            for m in METHODS]
    with open(RESULTS / fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return header, rows


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    data = load()
    assert data, f"no run data found below {ARTIFACTS} — run train.py first"
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = "  ".join(
        f"k{k}(d{int(data[k]['d'])}):train_steps={int(data[k]['train_steps'])}"
        for k in data
    )
    batch_size = int(next(iter(data.values()))["batch_size"])
    h1, r1 = write_table(data, "final_ess", "ess_table.csv")
    h2, r2 = write_table(data, "strict_cov", "mode_coverage_table.csv")
    md = [
        "# HD product multi-well — summary tables",
        "",
        f"Run config: batch_size={batch_size}, exp(-x^2) coeff=12.  {cfg}",
        "",
        "## Table 1 — final ESS (loss x d)   [headline]",
        "",
        md_table(h1, r1),
        "",
        f"## Table 2 — strict mode coverage  modes_found / 2^k, "
        f"threshold {STRICT_FRAC}x uniform share (loss x d)",
        "",
        md_table(h2, r2),
        "",
    ]
    (RESULTS / "tables.md").write_text("\n".join(md))
    print(f"wrote {RESULTS / 'tables.md'} (+ 2 csv files)")
    print(md_table(h1, r1))


if __name__ == "__main__":
    main()
