"""Build the two headline 4 x k tables (final ESS and exact mode coverage) from
the data_k*.pth files. Writes ess_table.csv, mode_coverage_table.csv, tables.md."""
from pathlib import Path
import csv
import torch
import parameters as P
from core import mode_coverage

HERE = Path(__file__).resolve().parent
METHODS = list(P.METHODS)
LABEL = {
    'KL': 'forward KL',
    'KL+X_mu': 'KL+X_mu',
    'KL+X_mu+X_hat_mu': 'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix': 'KL+X_mu+X_mix',
}


def load():
    found = {}
    for path in sorted(HERE.glob('data_k*.pth')):
        d = torch.load(path, weights_only=False)
        found[d['k']] = d
    return dict(sorted(found.items()))


STRICT_FRAC = 0.5  # a mode counts as covered only if it holds >= 50% of the
                   # uniform expected share (N / 2^k); exposes under-filled modes


def cell(data, k, m, field):
    run = data[k]['runs'][m]
    if field == 'strict_cov':
        y = run['samples'].float()
        cov, _ = mode_coverage(y, k, frac=STRICT_FRAC)
        return cov
    return run[field]


def write_table(field, fname, fmt='{:.4f}'):
    data = load()
    ks = list(data.keys())
    dims = [data[k]['d'] for k in ks]
    rows = []
    header = ['loss \\ d'] + [str(dd) for dd in dims]
    for m in METHODS:
        row = [LABEL[m]] + [fmt.format(cell(data, k, m, field)) for k in ks]
        rows.append(row)
    with open(HERE / fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return header, rows, ks, data


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def main():
    ess_h, ess_r, ks, data = write_table('final_ess', 'ess_table.csv')
    cov_h, cov_r, _, _ = write_table('strict_cov', 'mode_coverage_table.csv')
    bal_h, bal_r, _, _ = write_table('mode_balance', 'mode_balance_table.csv')

    steps_line = '  '.join(f"k{k}(d{data[k]['d']}):steps={data[k]['steps']}" for k in ks)
    md = []
    md.append('# VD product multi-well -- summary tables\n')
    md.append(f'Run config: batch={P.BATCH}, exp(-x^2) coeff=12.  {steps_line}\n')
    md.append('## Table 1 -- final ESS (loss x d)   [headline]\n')
    md.append(md_table(ess_h, ess_r))
    md.append(f'\n\n## Table 2 -- strict mode coverage  modes_found / 2^k, '
              f'threshold {STRICT_FRAC:g}x uniform share (loss x d)\n')
    md.append(md_table(cov_h, cov_r))
    md.append('\n\n## Table 3 -- mode imbalance  TV(occupancy, uniform), lower=better (loss x d)\n')
    md.append(md_table(bal_h, bal_r))
    md.append('\n')
    (HERE / 'tables.md').write_text('\n'.join(md))

    print('\n'.join(md))
    print(f"\nWrote ess_table.csv, mode_coverage_table.csv, mode_balance_table.csv, "
          f"tables.md to {HERE}")


if __name__ == '__main__':
    main()
