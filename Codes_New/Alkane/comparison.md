# Why propane 27D and n-butane 36D are much harder than methane and ethane

Measured in this folder with KL+X_pi (fixed regularization (100 kJ/mol, 0.15 nm)
for methane, ethane, propane; the diagonal path (50, 0.25) to (100, 0.15) for
butane; screen fraction 1e-4, MC_STEPS 20/100, batch 5000 to 10000):

| target | dimension | accepted stages | F_hat = prod ESS^(-1/2) | wall time | identity ESS from the source at t = 1 |
|---|---:|---:|---:|---:|---:|
| methane | 9 | 1 | 1.02 | 1.5 min | 4e-3 |
| ethane | 18 | 1 | 1.17 | 2.2 min | 8e-5 |
| propane | 27 | 2 | 1.59 | 11 min | 3e-5 |
| n-butane | 36 | 8 | 3.65 | 61 min | (below 1e-4 at t = 0.15 without the soft start) |

The difficulty is not proportional to the dimension. Two mechanisms explain
the jumps at 27D and 36D; the first is measured here, the second follows from
the molecules and from the earlier KLXX path of this target.

**1. The overlap between the source and the target decays exponentially with
the number of coordinates, so the first increment must shrink with size.** The
source is a product measure on the whitened internal coordinates; the
importance weight of a source sample against the target is a product of
per-coordinate factors, so its ESS falls roughly geometrically with the
dimension: 4e-3, 8e-5, 3e-5 above, and effectively zero for butane. The
adaptive schedule reacts by shortening the first stage (t_safe 1.0, 1.0, 0.4,
0.15), and every extra stage costs a full training run. Methane and ethane
finish in one stage; propane needs a stop at t = 0.4; butane needs eight
stages. On top of this, the number of nonbonded pairs grows quadratically with
the atom count, so a random source configuration is more likely to contain a
clash, and the singular core the flow has to learn to avoid takes a larger
share of the source volume; this is what the soft start of the regularization
path (low threshold, wide pair floor) relieves for butane.

**2. Butane is the first alkane with a backbone torsion, and its target is
multimodal with barriers that local moves do not cross.** Methane, ethane, and
propane carry only methyl rotations, whose three-fold symmetric wells are
equivalent, so their torsional marginals are close to uniform and the flow
has little to reshape on the torus block. n-Butane adds the C-C-C-C dihedral
with the trans well and the two gauche wells at different energies and
different widths, separated by barriers several kT high at 300 K; the
reference simulations of the manuscript need parallel tempering up to 800 K
to equilibrate this rotamer population, which local MALA at 300 K does not
do. The flow therefore has to build a three-well marginal, coupled to the
bond angles and to the methyl rotations, out of a uniform source marginal,
and the SMC surrogate cannot repair a wrong rotamer population by
rejuvenation: it can only reweight and resample what the current map
proposes. That is the region where the run stalls: the accepted stage ESS
dips to 0.45 to 0.63 for t between 0.15 and 0.35 before recovering to above
0.86 once the wells are in place, and the old KLXX path on this target shows
the same dip (flow ESS 0.55 to 0.59 for t from 0.17 to 0.26) with increments
of about 0.05.

**3. What does not help.** Training KL+X_pi directly from the source to the
target on a fixed set of 200000 target samples (propane, `train_data_driven.py`)
reaches a validation ESS of 0.18 with batch 10000 and 1500 steps and falls to
0.09 with twice the batch and twice the steps while its training loss keeps
decreasing: the map fits the stored samples rather than the density, and
nothing regenerates target samples through the current map. The staged
generator, with the SMC surrogate and the trained-versus-identity selection,
is what makes the larger targets tractable; the price is the number of
stages, which is set by the overlap decay and the torsional barriers, not by
the dimension.
