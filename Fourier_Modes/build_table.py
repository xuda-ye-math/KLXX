"""Build the results table (4 methods x {fine-grid ESS, modes_found/4, Naeem k=5
coverage}) from data.pth. Writes results_table.csv and results_table.md."""
from pathlib import Path
import csv
import argparse

import torch

import parameters as P

HERE = Path(__file__).resolve().parent

LABEL = {
    'KL': 'forward KL',
    'KL+X_mu': 'KL+X_mu',
    'KL+X_mu+X_hat_mu': 'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix': 'KL+X_mu+X_mix',
}


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    data_path = HERE / (f"data_{args.tag}.pth" if args.tag else "data.pth")
    d = torch.load(data_path, weights_only=False)
    runs = d['runs']
    nmodes = d['nmodes']

    header = ['loss', 'train-grid ESS', 'fine-grid ESS',
              f'modes_found/{nmodes}', 'Naeem k=5 cov', 'occupancy']
    rows = []
    csv_rows = []
    for m in P.METHODS:
        r = runs[m]
        occ = '[' + ', '.join(f"{v:.2f}" for v in r['occupancy'].tolist()) + ']'
        ess_train = r.get('ess_train', float('nan'))
        rows.append([
            LABEL[m],
            f"{ess_train:.4f}" + (' (ABORTED)' if r['aborted'] else ''),
            f"{r['final_ess']:.4f}",
            f"{r['modes_found']}/{nmodes}",
            f"{r['knn_coverage']:.4f}",
            occ,
        ])
        csv_rows.append([m, ess_train, r['final_ess'], r['modes_found'], nmodes,
                         r['knn_coverage'], occ])

    csv_path = HERE / 'results_table.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['loss', 'train_grid_ess', 'fine_grid_ess', 'modes_found',
                    'nmodes', 'naeem_k5_coverage', 'occupancy'])
        w.writerows(csv_rows)

    qt_occ = '[' + ', '.join(f"{v:.2f}" for v in d['qt_occupancy'].tolist()) + ']'
    md = []
    md.append('# Fourier-field Bayesian inverse problem -- results\n')
    md.append(f"Run config: d={d['d']}, G={d['groups']}, {nmodes} sign-flip modes, "
              f"AMP={d['config']['AMP']}, sigma_obs={d['config']['SIGMA_OBS']}, "
              f"coarse {d['config']['COARSE_N']}x{d['config']['COARSE_N']} (train), "
              f"fine {d['config']['FINE_N']}x{d['config']['FINE_N']} (eval), "
              f"steps={d['steps']}, batch={d['config']['BATCH']}.\n")
    md.append(f"QT pool: modes_found={int(round(d['qt_mode_coverage']*nmodes))}/{nmodes}, "
              f"occupancy={qt_occ}.\n")
    md.append('## Results (4 methods, honest fine-grid eval)\n')
    md.append(md_table(header, rows))
    md.append('\n')
    md_text = '\n'.join(md)
    (HERE / 'results_table.md').write_text(md_text)

    print(md_text)
    print(f"\nWrote results_table.csv and results_table.md to {HERE}")


if __name__ == '__main__':
    main()
