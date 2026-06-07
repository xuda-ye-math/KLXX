import json
src = open('main.tex').read()
s35 = src[src.index('\\subsection{Coarse training'):src.index('\\section{Adaptive-temperature')]
s56 = src[src.index('\\subsection{Bayesian screened Poisson'):src.index('\\section*{Acknowledgement}')]
prompt = ("You are an adversarial consistency auditor. Section 3.5 below states a general coarse-training/"
"fine-reweighting construction; Section 5.6 instantiates it inside an adaptive-temperature ladder (per stage, "
"whitened coordinates xi, tempered misfit t_k*Phi). Check ONLY cross-consistency between the two: (1) does the "
"per-stage fine weight eq: poisson-fine correctly instantiate eq: coarse-fine-z (note: in 5.6 the high block is "
"freshly prior-distributed each stage, so the source-end correction of coarse-fine-z vanishes and only the "
"target-end correction t_k(Phi_L - Phi) survives -- is that consistent and is it stated clearly enough); "
"(2) notation: 3.5 uses x/y with blocks L/H and potentials U_0, U, U_{0,L}, U_L; 5.6 uses xi with blocks L/H, "
"potentials U, U_L, misfits Phi, Phi_L -- any collision or undefined mapping between the two; (3) claims: does "
"either section promise something the other contradicts (storage in full coordinates vs fresh prior draws; "
"gates; what the ESS measures). Output max 4 findings with exact quotes and fixes, or CONSISTENT.\n\n"
"=== SECTION 3.5 ===\n" + s35 + "\n=== SECTION 5.6 ===\n" + s56)
json.dump({'prompt': prompt}, open('/tmp/adv.json','w'))
print('bytes', len(prompt))
