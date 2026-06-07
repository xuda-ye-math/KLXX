import json, subprocess, sys
n = sys.argv[1]; focus = sys.argv[2] if len(sys.argv) > 2 else ''
src = open('main.tex').read()
i = src.index('\\subsection{Bayesian screened Poisson')
newsec = src[i:src.index('\\section*{Acknowledgement}')]
clock = src[src.index('\\subsection{Boltzmann generator'):i]
phi4 = src[src.index('\\subsection{Lattice'):src.index('\\subsection{Boltzmann generator')]
absct = src[src.index('\\begin{abstract}'):src.index('\\end{abstract}')]
contrib = src[src.index('This paper makes four contributions'):src.index('This paper makes four contributions')+6000]
remark = src[src.index('\\begin{remark}[Coarse training'):src.index('\\end{remark}', src.index('\\begin{remark}[Coarse training'))]
truth = open('../Poisson_Inverse/comparison_referee.md').read() + "\n" + open('../Poisson_Inverse/referee.md').read()[:3000]
ledger = open('../.aris/reviews/poisson-subsection/LEDGER.md').read()
prompt = ("You are an adversarial reviewer of a NEW subsection (5.6, screened-Poisson Bayesian inversion) in a "
"J.Comput.Phys. paper on X-regularized forward KL for normalizing-flow Boltzmann generators. Round " + n +
" of an iterative improvement loop. Judge: (1) QUANTITATIVE FIDELITY -- every number in the subsection, the "
"abstract sentence, and the contributions sentence must match the ground-truth data below; (2) CONSISTENCY with "
"the rest of the paper -- terminology (fake ESS and forward KL never hyphenated, potentials named U, ESS/coverage "
"conventions, the established loss names), cross-references, theme continuity with the phi4 (fake ESS) and clock "
"(per-stage quality) subsections and the coarse-train/fine-IS remark; (3) WRITING -- terse declarative voice, "
"minimal parentheses, claims not outrunning evidence; (4) MATH -- the displayed model, whitening, bridge identity, "
"and fine-ESS formula must be correct and self-contained. Do NOT re-report ledger items.\n\n=== LEDGER ===\n"
+ ledger + "\n=== GROUND TRUTH (referee + comparison data) ===\n" + truth +
"\n=== NEW SUBSECTION ===\n" + newsec + "\n=== ABSTRACT ===\n" + absct +
"\n=== CONTRIBUTIONS PARAGRAPH ===\n" + contrib + "\n=== REMARK coarse-fine-IS ===\n" + remark +
"\n=== CLOCK SUBSECTION (style/continuity reference) ===\n" + clock +
"\n=== PHI4 SUBSECTION (theme reference) ===\n" + phi4 +
("\nROUND FOCUS: " + focus if focus else "") +
"\n\nOutput STRICTLY: numbered findings, each severity critical/major/minor + exact quote + why + concrete fix; "
"max 6; or exactly CONSISTENT.")
json.dump({'prompt': prompt}, open('/tmp/pr_round.json', 'w'))
print('bytes', len(prompt))
