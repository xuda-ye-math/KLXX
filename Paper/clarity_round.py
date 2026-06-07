import json, sys
n = sys.argv[1]
src = open('main.tex').read()
i = src.index('\\subsection{Coarse training, fine importance sampling}')
j = src.index('\\section{Adaptive-temperature Boltzmann generator}')
chunk = src[i:j]
prompt = ("You are a scientific copy editor, round " + n + " of 3. Polish the CLARITY and GRAMMAR of this LaTeX "
"subsection from a J.Comput.Phys. paper. Rules: do not change any mathematical content, symbol, claim, or "
"established terminology (forward KL, X-regularized loss, fine validation ESS); prefer referencing displayed "
"equations via \\eqref{...} over vague phrases ('the brackets above', 'this expression'); every pronoun and "
"definite description must have an unambiguous antecedent; appositions must not read as sentence splices; "
"terse declarative journal register. Output STRICTLY a numbered list: exact quote -> replacement, with a "
"one-line reason; max 6; or CLEAN if nothing remains.\n\n" + chunk)
json.dump({'prompt': prompt}, open('/tmp/adv.json','w'))
print('bytes', len(prompt))
