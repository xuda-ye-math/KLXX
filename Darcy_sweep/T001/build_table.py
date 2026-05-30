"""Build results table from data.pth.
Headline row: 4 methods x {coarse-grid ESS, fine-grid ESS, modes_found/6, kNN coverage, mode occupancy}.
Writes results_table.csv and results_table.md.
"""
from pathlib import Path
import csv
import torch
import parameters as P

HERE = Path(__file__).resolve().parent

LABEL = {
    'KL':                'forward KL',
    'KL+X_mu':           'KL + X_mu',
    'KL+X_mu+X_hat_mu':  'KL + X_mu + X_hat_mu',
    'KL+X_mu+X_mix':     'KL + X_mu + X_mix',
}


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def main():
    d = torch.load(HERE / 'data.pth', weights_only=False)
    runs = d['runs']
    cfg = d['config']
    NMODES = 6

    header = ['loss', 'coarse ESS', 'fine ESS', f'modes/{NMODES}', 'kNN cov', 'occupancy']
    rows, csv_rows = [], []
    for m in P.METHODS:
        r = runs[m]
        occ = '[' + ', '.join(f"{v:.2f}" for v in r['occupancy']) + ']'
        modes_found = int(round(r['modes_found'] * NMODES))
        rows.append([
            LABEL[m],
            f"{r['final_ess_coarse']:.4f}",
            f"{r['final_ess_fine']:.4f}",
            f"{modes_found}/{NMODES}",
            f"{r['knn_coverage']:.3f}",
            occ,
        ])
        csv_rows.append([m, r['final_ess_coarse'], r['final_ess_fine'],
                         modes_found, NMODES, r['knn_coverage'], occ])

    with open(HERE / 'results_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['loss', 'ess_coarse', 'ess_fine', 'modes_found', 'nmodes',
                    'knn_coverage', 'occupancy'])
        w.writerows(csv_rows)

    md = []
    md.append('# Darcy_2D — Bayesian elliptic source-inversion')
    md.append('')
    md.append(f"Forward: -Lap u = g(x;theta) on [0,1]^2 periodic, "
              f"g = G0 (1 + delta cos(alpha v(x;theta))), spectral exact solve.")
    md.append(f"Coarse grid {cfg['N_GRID']}^2; fine grid {cfg['FINE_N_GRID']}^2.")
    md.append(f"G0={cfg['G0']}, alpha={cfg['ALPHA']}, delta={cfg['DELTA']}, "
              f"sigma_obs={cfg['SIGMA_OBS']}, {cfg['N_SENSORS']} sensors.")
    md.append(f"6 posterior modes (sign-flip x constant-mode shift); steps={cfg['STEPS']}.")
    md.append('')
    md.append('## Results')
    md.append('')
    md.append(md_table(header, rows))
    md.append('')
    text = '\n'.join(md)
    (HERE / 'results_table.md').write_text(text)
    print(text)
    print(f"\nWrote results_table.csv and results_table.md")


if __name__ == '__main__':
    main()
