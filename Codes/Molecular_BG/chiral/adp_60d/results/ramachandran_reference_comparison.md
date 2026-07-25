# ADP Ramachandran comparison

The final KLXX stage against the published alanine dipeptide reference
data at 300 K (Zenodo record 6993124, DOI 10.5281/zenodo.6993124).
Both sets use the same 100-bin phi/psi grid, the same wrapped
smoothing, and each surface is normalized to its own most populated
bin. The reference topology matches the frozen bundle atom for atom, so
the same phi/psi atom indices apply to both.

- KLXX stage 10: `10,000,000` samples, t = `1.000`, rho = `(125, 0.1)`
- MD reference: `10,000,000` frames

## Surface extent

Occupied bins counts every bin holding at least one sample. The
maximum is taken over finite bins only.

| sample set | occupied bins | max free energy / kBT |
|---|---:|---:|
| KLXX | 6413 | 10.40 |
| MD reference (FAB) | 6481 | 10.40 |

## Free energy by region

Bins are classified by the MD reference: low means below 2 kBT, high means above 3 kBT.
Entries are the mean free energy of each set over those bins, in kBT.
Only bins finite in both surfaces are used, so the rows are directly
comparable.

| region | bins | KLXX | MD reference (FAB) |
|---|---:|---:|---:|
| low F (< 2 kBT) | 653 | 1.230 | 1.230 |
| high F (> 3 kBT) | 4496 | 6.323 | 6.293 |
| all bins | 5690 | 5.379 | 5.355 |

## Backbone basin populations

Fractions of the phi/psi histogram; alpha-L is phi > 0, and the
phi < 0 half is split by the sign of psi.

| sample set | alpha-L (phi > 0) | beta/PPII (psi > 0) | alpha-R (psi < 0) |
|---|---:|---:|---:|
| KLXX | 0.0031 | 0.7958 | 0.2012 |
| MD reference (FAB) | 0.0033 | 0.7940 | 0.2027 |
