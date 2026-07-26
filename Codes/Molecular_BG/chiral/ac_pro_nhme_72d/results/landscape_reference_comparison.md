# Ac-Pro-NHMe landscape comparison

Three sample sets on the same 100-bin torsion grid with the same
wrapped smoothing, each surface normalized to its own most populated
bin. `KLXX resampled + rejuvenated` is the KLXX set reweighted by the
regularization ESS weights, resampled, then rejuvenated with
mixed-domain MALA under the raw potential.

- Regularization ESS (RESS): `1.000002`
- Mixed-domain MALA acceptance during rejuvenation: `0.2282`

## Surface extent

| diagnostic | sample set | occupied bins | max free energy / kBT |
|---|---|---:|---:|
| backbone | KLXX | 1825 | 7.43 |
| backbone | KLXX resampled + rejuvenated | 1825 | 7.43 |
| backbone | OpenMM reference | 1698 | 7.61 |
| peptide/ring | KLXX | 1147 | 8.13 |
| peptide/ring | KLXX resampled + rejuvenated | 1149 | 8.14 |
| peptide/ring | OpenMM reference | 1067 | 8.41 |

## Free energy by region

Bins are classified by the OpenMM reference: low means below 2 kBT, high means above 3 kBT.
Entries are the mean free energy of each set over those bins, in kBT.
Only bins finite in all three surfaces are used, so the rows are
directly comparable.

| diagnostic | region | bins | KLXX | KLXX resampled + rejuvenated | OpenMM reference |
|---|---|---:|---:|---:|---:|
| backbone | low F (< 2 kBT) | 328 | 0.972 | 0.969 | 1.121 |
| backbone | high F (> 3 kBT) | 922 | 4.551 | 4.558 | 5.053 |
| backbone | all bins | 1474 | 3.394 | 3.397 | 3.789 |
| peptide/ring | low F (< 2 kBT) | 141 | 1.134 | 1.142 | 1.161 |
| peptide/ring | high F (> 3 kBT) | 682 | 4.178 | 4.193 | 5.173 |
| peptide/ring | all bins | 965 | 3.427 | 3.440 | 4.198 |

## ACE-PRO peptide torsion populations

| sample set | trans | cis |
|---|---:|---:|
| KLXX | 0.7165 | 0.2835 |
| KLXX resampled + rejuvenated | 0.7170 | 0.2830 |
| OpenMM reference | 0.9044 | 0.0956 |
