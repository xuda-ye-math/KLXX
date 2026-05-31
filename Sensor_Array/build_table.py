"""Build the results table (markdown + csv) for the Sensor_Array experiment from
data.pth. One row per loss: final ESS, modes discovered (out of 3!=6), k=5 Naeem
coverage vs the QT pool, and the per-mode occupancy fractions.
"""
from pathlib import Path
import csv
import torch

HERE = Path(__file__).resolve().parent

METHODS = ('KL', 'KL+X_mu', 'KL+X_mu+X_hat_mu', 'KL+X_mu+X_mix')
PRETTY = {
    'KL':                'forward KL',
    'KL+X_mu':           'forward KL + X_mu',
    'KL+X_mu+X_hat_mu':  'forward KL + X_mu + X_hat_mu',
    'KL+X_mu+X_mix':     'forward KL + X_mu + X_(hat_mu+bar_nu)/2',
}


def main():
    data = torch.load(HERE / 'data.pth', weights_only=False)
    nmodes = data['nmodes']
    rows = []
    for m in METHODS:
        r = data['runs'][m]
        occ = [round(float(v), 3) for v in r['occupancy'].tolist()]
        rows.append({
            'loss': PRETTY[m],
            'ESS': round(float(r['final_ess']), 3),
            'modes': f"{r['modes_found']}/{nmodes}",
            'knn_cov': round(float(r['knn_coverage']), 3),
            'occupancy': occ,
            'aborted': bool(r['aborted']),
        })

    # --- markdown ---
    md = []
    md.append(f"# Sensor-array source localization ({data['n_src']} sources, "
              f"{nmodes} permutation modes)\n")
    md.append(f"- theta* = {[round(v,3) for v in data['theta_star'].tolist()]}, "
              f"||theta*|| = {data['theta_star'].norm():.3f}")
    md.append(f"- QT pool modes found: {round(data['qt_mode_coverage']*nmodes)}/{nmodes}, "
              f"occ = {[round(float(v),3) for v in data['qt_occupancy'].tolist()]}")
    md.append(f"- steps = {data['steps']}\n")
    md.append("| loss | ESS | modes | k=5 cov | per-mode occupancy |")
    md.append("|------|-----|-------|---------|--------------------|")
    for r in rows:
        flag = "  *(aborted)*" if r['aborted'] else ""
        md.append(f"| {r['loss']} | {r['ESS']:.3f} | {r['modes']} | "
                  f"{r['knn_cov']:.3f} | {r['occupancy']}{flag} |")
    md_text = "\n".join(md) + "\n"
    (HERE / 'results_table.md').write_text(md_text)
    print(md_text)

    # --- csv ---
    with open(HERE / 'results_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['loss', 'ESS', 'modes_found', 'nmodes', 'knn_coverage']
                   + [f'occ_{i}' for i in range(nmodes)])
        for m, r in zip(METHODS, rows):
            occ = data['runs'][m]['occupancy'].tolist()
            w.writerow([PRETTY[m], r['ESS'], data['runs'][m]['modes_found'], nmodes,
                        r['knn_cov']] + [round(float(v), 4) for v in occ])
    print(f"Saved {HERE/'results_table.md'} and {HERE/'results_table.csv'}")


if __name__ == '__main__':
    main()
