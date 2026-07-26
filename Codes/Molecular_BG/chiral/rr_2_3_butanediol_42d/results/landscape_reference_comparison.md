# (2R,3R)-2,3-Butanediol landscape comparison

The KLXX generator and an independent OpenMM run on the same
100-bin torsion grid with the same wrapped smoothing. Each
surface is normalized to its own most populated bin, so only raw
per-method values are reported; a difference between two such surfaces
would carry an arbitrary additive offset.

The KLXX column is the final stage at $t=1$ of an inference-only run
with no training update. The OpenMM column is native parallel
tempering on the unregularized physical potential at 300 K: one
continuous chain over six replicas on a geometric 300--800 K grid,
50.0 ns retained after a 0.5 ns discarded equilibration.

The two sample counts differ by two orders of magnitude, as the
sample-set table records. Generating flow samples is cheap and
parallel; advancing a single tempered trajectory is neither.

### Sample sets

<div align="center">

<table>
<thead>
<tr><th>method</th><th>samples</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>10,000,000</td></tr>
<tr><td>OpenMM</td><td>100,000</td></tr>
</tbody>
</table>

</div>

### Maximum free energy

<div align="center">

<table>
<thead>
<tr><th>diagnostic</th><th>sample count</th><th>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></th><th>OpenMM</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">heavy-atom</td><td>as sampled</td><td>12.12</td><td>7.54</td></tr>
<tr><td>thinned to 100,000</td><td>7.52</td><td>7.54</td></tr>
<tr><td rowspan="2">hydroxyl</td><td>as sampled</td><td>3.84</td><td>4.04</td></tr>
<tr><td>thinned to 100,000</td><td>4.04</td><td>4.04</td></tr>
</tbody>
</table>

</div>

Maximum free energy over finite bins, in units of $k_{\mathrm B}T$.

Each surface is normalized to its own most populated bin, so the
deepest value it can resolve is set by the bin holding a single
sample and grows as $\log N$. The `as sampled` row therefore compares
two different resolutions rather than two free-energy surfaces. The
`thinned` row removes that by histogramming the generated set at the
reference frame count, which is a measurement at matched size rather
than a correction applied to either column.

### Mean free energy by region

<div align="center">

<table>
<thead>
<tr><th>diagnostic</th><th>region</th><th>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></th><th>OpenMM</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">heavy-atom</td><td>low (&lt; 2)</td><td>1.229</td><td>1.259</td></tr>
<tr><td>high (&gt; 3)</td><td>5.126</td><td>5.353</td></tr>
<tr><td>all</td><td>4.137</td><td>4.309</td></tr>
<tr><td rowspan="3">hydroxyl</td><td>low (&lt; 2)</td><td>1.226</td><td>1.314</td></tr>
<tr><td>high (&gt; 3)</td><td>3.116</td><td>3.715</td></tr>
<tr><td>all</td><td>2.059</td><td>2.326</td></tr>
</tbody>
</table>

</div>

Regions are classified by the OpenMM surface and restricted to bins
finite in both surfaces, so the rows are directly comparable.

### Central C-C-C-C rotamer populations

<div align="center">

<table>
<thead>
<tr><th>method</th><th>gauche&minus;</th><th>gauche+</th><th>trans</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>0.1486</td><td>0.1861</td><td>0.6652</td></tr>
<tr><td>OpenMM</td><td>0.1425</td><td>0.1726</td><td>0.6849</td></tr>
</tbody>
</table>

</div>

