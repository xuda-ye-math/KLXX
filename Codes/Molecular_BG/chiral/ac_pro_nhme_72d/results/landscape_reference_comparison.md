# Ac-Pro-NHMe landscape comparison

The KLXX generator and an independent OpenMM run on the same
100-bin torsion grid with the same wrapped smoothing. Each
surface is normalized to its own most populated bin, so only raw
per-method values are reported; a difference between two such surfaces
would carry an arbitrary additive offset.

The OpenMM column is native parallel tempering on the
unregularized physical potential at 300 K.

### Sample sets

<div align="center">

<table>
<thead>
<tr><th>method</th><th>samples</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>320,000</td></tr>
<tr><td>OpenMM</td><td>320,000</td></tr>
</tbody>
</table>

</div>

### Maximum free energy

<div align="center">

<table>
<thead>
<tr><th>diagnostic</th><th>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></th><th>OpenMM</th></tr>
</thead>
<tbody>
<tr><td>backbone</td><td>7.43</td><td>7.61</td></tr>
<tr><td>peptide/ring</td><td>8.13</td><td>8.41</td></tr>
</tbody>
</table>

</div>

Maximum free energy over finite bins, in units of $k_{\mathrm B}T$.

### Mean free energy by region

<div align="center">

<table>
<thead>
<tr><th>diagnostic</th><th>region</th><th>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></th><th>OpenMM</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">backbone</td><td>low (&lt; 2)</td><td>0.972</td><td>1.121</td></tr>
<tr><td>high (&gt; 3)</td><td>4.598</td><td>5.095</td></tr>
<tr><td>all</td><td>3.442</td><td>3.835</td></tr>
<tr><td rowspan="3">peptide/ring</td><td>low (&lt; 2)</td><td>1.134</td><td>1.161</td></tr>
<tr><td>high (&gt; 3)</td><td>4.244</td><td>5.218</td></tr>
<tr><td>all</td><td>3.486</td><td>4.246</td></tr>
</tbody>
</table>

</div>

Regions are classified by the OpenMM surface and restricted to bins
finite in both surfaces, so the rows are directly comparable.

### ACE-PRO peptide torsion populations

<div align="center">

<table>
<thead>
<tr><th>method</th><th>trans</th><th>cis</th></tr>
</thead>
<tbody>
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>0.7165</td><td>0.2835</td></tr>
<tr><td>OpenMM</td><td>0.9044</td><td>0.0956</td></tr>
</tbody>
</table>

</div>

