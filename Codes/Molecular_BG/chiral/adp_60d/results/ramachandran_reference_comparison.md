# ADP Ramachandran comparison

The KLXX generator against ground truth: the published alanine
dipeptide reference data at 300 K from the FAB study (Zenodo record
6993124, DOI 10.5281/zenodo.6993124), on the same 100-bin phi/psi
grid with the same wrapped smoothing. Each
surface is normalized to its own most populated bin, so only raw
per-method values are reported; a difference between two such surfaces
would carry an arbitrary additive offset.

The reference topology matches the frozen bundle atom for atom, so the
same phi/psi atom indices apply to both.

### Sample sets

<div align="center">

<table>
<thead>
<tr><th>method</th><th>samples</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>10,000,000</td></tr>
<tr><td>ground truth</td><td>10,000,000</td></tr>
</tbody>
</table>

</div>

### Maximum free energy

<div align="center">

<table>
<thead>
<tr><th>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></th><th>ground truth</th></tr>
</thead>
<tbody>
<tr><td>10.40</td><td>10.40</td></tr>
</tbody>
</table>

</div>

Maximum free energy over finite bins, in units of $k_{\mathrm B}T$.

### Mean free energy by region

<div align="center">

<table>
<thead>
<tr><th>region</th><th>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></th><th>ground truth</th></tr>
</thead>
<tbody>
<tr><td>low (&lt; 2)</td><td>1.230</td><td>1.230</td></tr>
<tr><td>high (&gt; 3)</td><td>6.323</td><td>6.293</td></tr>
<tr><td>all</td><td>5.379</td><td>5.355</td></tr>
</tbody>
</table>

</div>

Regions are classified by the reference surface and restricted to bins
finite in both surfaces, so the rows are directly comparable.

### Backbone basin populations

<div align="center">

<table>
<thead>
<tr><th>method</th><th>alpha-L (phi &gt; 0)</th><th>beta/PPII (psi &gt; 0)</th><th>alpha-R (psi &lt; 0)</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>0.0031</td><td>0.7958</td><td>0.2012</td></tr>
<tr><td>ground truth</td><td>0.0033</td><td>0.7940</td><td>0.2027</td></tr>
</tbody>
</table>

</div>

