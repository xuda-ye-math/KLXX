# Ac-Pro-NHMe peptide torsion comparison

The KLXX generator and an independent OpenMM run, compared on the
cis/trans populations of the ACE-PRO $\omega$ torsion. Cis is
$|\omega| < \pi/2$, counted per sample on both sides, so no binning
grid enters the comparison.

The KLXX column is the final stage at $t=1$ of an inference-only run
with no training update. The OpenMM column is Langevin dynamics at
300 K on the same unregularized physical potential, plus a Metropolis
rigid $\pi$ rotation of the proline side of the ACE-PRO amide bond.
The rotation is an involution with unit Jacobian, so the jump is exact
and every retained sample counts once at the target temperature.

The amide barrier is about 60-65 kJ/mol, some 26 $k_{\mathrm B}T$ at
300 K. Neither column measures it. The generator cannot resolve a free
energy deeper than about $\log N$ above its most populated bin, and
the reference obtains its basin changes from the jump move rather than
from the dynamics, which is what lets it estimate populations without
deforming the target. No barrier is therefore tabulated.

### Sample sets

<div align="center">

<table>
<thead>
<tr><th>method</th><th>sampler</th><th>samples</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>inference replay</td><td>15,000,000</td></tr>
<tr><td>OpenMM</td><td>basin-jump Monte Carlo</td><td>2,000,000</td></tr>
</tbody>
</table>

</div>

The reference retained 2,000,000 samples after discarding
the first 4,000, over
501.0 ns, accepting
405,542 of the retained jump attempts
(0.2028).

### Peptide torsion populations

<div align="center">

<table>
<thead>
<tr><th>method</th><th>cis</th><th>trans</th><th>&Delta;G(cis&minus;trans)</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>0.1199</td><td>0.8801</td><td>4.97</td></tr>
<tr><td>OpenMM</td><td>0.1443 &plusmn; 0.0003</td><td>0.8557</td><td>4.44</td></tr>
</tbody>
</table>

</div>

Free energy differences are in kJ/mol. The reference error bar is its
blocking standard error at 32 blocks, not the naive binomial
one; the series is correlated, so the block spread is the honest
scale. The two cis fractions differ by
0.0244, which is 82 times that error, and the two
$\Delta G$ values differ by 0.53
kJ/mol, or 0.21
$k_{\mathrm B}T$. The disagreement is far outside what the
reference's own spread can account for.

### Reference convergence

<div align="center">

<table>
<thead>
<tr><th>blocks</th><th>samples per block</th><th>cis</th><th>standard error</th></tr>
</thead>
<tbody>
<tr><td>2</td><td>1,000,000</td><td>0.144315</td><td>0.000111</td></tr>
<tr><td>4</td><td>500,000</td><td>0.144315</td><td>0.000260</td></tr>
<tr><td>8</td><td>250,000</td><td>0.144315</td><td>0.000334</td></tr>
<tr><td>16</td><td>125,000</td><td>0.144315</td><td>0.000294</td></tr>
<tr><td>32</td><td>62,500</td><td>0.144315</td><td>0.000297</td></tr>
<tr><td>64</td><td>31,250</td><td>0.144315</td><td>0.000304</td></tr>
<tr><td>128</td><td>15,625</td><td>0.144315</td><td>0.000345</td></tr>
<tr><td>256</td><td>7,812</td><td>0.144317</td><td>0.000350</td></tr>
<tr><td>512</td><td>3,906</td><td>0.144317</td><td>0.000336</td></tr>
</tbody>
</table>

</div>

Every blocking returns the same central value to five decimal places.
The standard error is 0.000111 at 2
blocks, where the spread of so few block means is itself uncertain by
about 71 per cent and cannot carry the error bar; it settles by eight
blocks and then stays between 0.000294 and
0.000350 for every blocking from 16 to 512.
Short blocks cannot see correlations longer than themselves, so the
quoted error is the 32-block row, inside that plateau. The
plateau across more than an order of magnitude in block length is why
it is not an artifact of the block choice, and it is the scale against
which the disagreement above must be read.

