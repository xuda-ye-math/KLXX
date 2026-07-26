# (2R,3R)-2,3-Butanediol landscape comparison

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
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>240,000</td></tr>
<tr><td>OpenMM</td><td>240,000</td></tr>
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
<tr><td>heavy-atom</td><td>8.32</td><td>8.44</td></tr>
<tr><td>hydroxyl</td><td>4.83</td><td>4.93</td></tr>
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
<tr><td rowspan="3">heavy-atom</td><td>low (&lt; 2)</td><td>1.125</td><td>1.239</td></tr>
<tr><td>high (&gt; 3)</td><td>5.482</td><td>5.743</td></tr>
<tr><td>all</td><td>4.492</td><td>4.730</td></tr>
<tr><td rowspan="3">hydroxyl</td><td>low (&lt; 2)</td><td>1.209</td><td>1.297</td></tr>
<tr><td>high (&gt; 3)</td><td>3.237</td><td>3.572</td></tr>
<tr><td>all</td><td>2.091</td><td>2.275</td></tr>
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
<tr><td>KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub></td><td>0.1758</td><td>0.2091</td><td>0.6151</td></tr>
<tr><td>OpenMM</td><td>0.1383</td><td>0.1663</td><td>0.6953</td></tr>
</tbody>
</table>

</div>

