# Butanediol landscape comparison

Three sample sets on the same 100-bin torsion grid with the same
wrapped smoothing, each surface normalized to its own most populated
bin. `KLXX resampled + rejuvenated` is the KLXX set reweighted by the
regularization ESS weights, resampled, then rejuvenated with
mixed-domain MALA under the raw potential.

- Regularization ESS (RESS): `0.999981`
- Mixed-domain MALA acceptance during rejuvenation: `0.8867`

## Surface extent

| diagnostic | sample set | occupied bins | max free energy / kBT |
|---|---|---:|---:|
| heavy-atom | KLXX | 836 | 8.32 |
| heavy-atom | KLXX resampled + rejuvenated | 845 | 8.31 |
| heavy-atom | OpenMM reference | 830 | 8.44 |
| hydroxyl | KLXX | 10000 | 4.83 |
| hydroxyl | KLXX resampled + rejuvenated | 10000 | 4.80 |
| hydroxyl | OpenMM reference | 10000 | 4.93 |

## Free energy by region

Bins are classified by the OpenMM reference: low means below 2 kBT, high means above 3 kBT.
Entries are the mean free energy of each set over those bins, in kBT.
Only bins finite in all three surfaces are used, so the rows are
directly comparable.

| diagnostic | region | bins | KLXX | KLXX resampled + rejuvenated | OpenMM reference |
|---|---|---:|---:|---:|---:|
| heavy-atom | low F (< 2 kBT) | 104 | 1.125 | 1.114 | 1.239 |
| heavy-atom | high F (> 3 kBT) | 601 | 5.468 | 5.455 | 5.729 |
| heavy-atom | all bins | 816 | 4.478 | 4.465 | 4.716 |
| hydroxyl | low F (< 2 kBT) | 4182 | 1.209 | 1.177 | 1.297 |
| hydroxyl | high F (> 3 kBT) | 2536 | 3.237 | 3.203 | 3.572 |
| hydroxyl | all bins | 10000 | 2.091 | 2.056 | 2.275 |

## Central C-C-C-C rotamer populations

| sample set | gauche-minus | gauche-plus | trans |
|---|---:|---:|---:|
| KLXX | 0.1758 | 0.2091 | 0.6151 |
| KLXX resampled + rejuvenated | 0.1762 | 0.2096 | 0.6142 |
| OpenMM reference | 0.1383 | 0.1663 | 0.6953 |
