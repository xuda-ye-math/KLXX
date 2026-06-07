import json, sys
n, lo, hi, name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lines = open('main.tex').readlines()
chunk = "".join(lines[int(lo)-1:int(hi)])
prompt = ("You are a scientific copy editor. TASK: find REDUNDANT adverbs and intensifier adjectives in the LaTeX "
"prose below (a slice of a J.Comput.Phys. paper) whose removal leaves the sentence's meaning unchanged. "
"Typical targets: every/all when 'the ... (plural)' suffices, exactly, entirely, simply, clearly, very, "
"considerably, completely, essentially, fully, highly, strictly (when not mathematical), genuinely, deliberately "
"(when the deliberateness is not the point), 'in practice' fillers. DO NOT touch: mathematically load-bearing "
"words (strictly positive, exactly zero, identically, almost surely), claims' quantifiers that carry content "
"('on every rung' when per-rung universality IS the result; 'never enters a gradient step' where the negation is "
"the point), established terminology, math, numbers, or anything inside equations. Output STRICTLY a numbered "
"list: exact quote (enough context to locate uniquely) -> trimmed replacement; max 10; if none, output CLEAN. "
"Round " + n + ", section: " + name + "\n\n" + chunk)
json.dump({'prompt': prompt}, open('/tmp/adv.json','w'))
print('bytes', len(prompt))
