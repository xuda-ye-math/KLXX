import json, sys
n = sys.argv[1]
src = open('main.tex').read()
i = src.index('\\subsection{Coarse training')
j = src.index('\\section{Adaptive-temperature')
chunk = src[i:j]
prompt = ("You are a scientific copy editor, round " + n + " of 3, polishing ONE subsection of a J.Comput.Phys. "
"paper. Goal: plain, direct, journal-register prose. Remove FANCY or literary phrasing and replace with plain "
"statements; flag anything that reads as marketing or essayistic. Specific targets in this text: 'A useful "
"special structure arises when', 'The difficulty that a flow exists to resolve', 'while ... is free', 'This "
"invites', 'is the expensive part', 'changes the cost ... but not the consistency'. Hard rules: change NO "
"mathematical content, symbol, equation, label, or \\eqref; keep terminology (forward KL, X-regularized loss, "
"importance sampling, pushforward); every claim must stay exactly as strong/weak as written (no over- or "
"under-claiming). Prefer shorter declarative sentences. Output STRICTLY a numbered list: exact quote -> plain "
"replacement, one-line reason; max 8; or CLEAN.\n\n" + chunk)
json.dump({'prompt': prompt}, open('/tmp/p35.json','w'))
print('bytes', len(prompt))
