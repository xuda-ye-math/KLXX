---
name: paper-writing
description: Draft, revise, organize, and review rigorous scientific papers in scientific machine learning, computational mathematics, stochastic analysis, sampling, optimization, and adjacent mathematical sciences, while matching the user's established LaTeX voice. Use for abstracts, introductions, related work, methods, theorem statements, proofs, numerical experiments, captions, conclusions, supplements, rebuttals, or full-paper consistency checks.
---

# Paper Writing

Write mathematically serious SciML and analysis papers in the author's established style. Use the raw corpus as primary evidence:

- `/mnt/projects/Stochastic-Gradient/Paper/` for theorem-led stochastic analysis, assumptions, error decomposition, proof structure, and numerical verification.
- `/mnt/projects/X-regularization/Paper_Arxiv/` for modern SciML framing, mechanism-led methods, diagnostic experiments, limitations, molecular applications, and main/supplement organization.

Treat the current paper being edited as the local authority for notation, formatting, terminology, and section rhythm. Use the corpus only to resolve choices the current paper does not settle.

## Start with evidence

Before drafting or revising:

1. For every `.tex` task, inspect the document's master file and preamble area before editing prose or mathematics. Inventory the user-defined commands relevant to the requested section and treat them as the authoritative LaTeX vocabulary.
2. Read the surrounding section and identify the paper's central object, precise claim, audience, and evidence available.
3. Locate the nearest corpus analogue: theorem/proof, method, experiment, caption, abstract, or conclusion.
4. Inspect result files, tables, figures, code, and bibliography entries that support the requested claims. Never infer numerical results from filenames or memory.
5. Preserve existing macros, labels, environment choices, spelling, and notation unless the user asks to change them. Prefer project-defined commands over equivalent raw LaTeX; for example, when the preamble defines `\Ge` and `\Le`, use them instead of `\geq`, `\geqslant`, `\leq`, or `\leqslant`.

Read [references/voice.md](references/voice.md) before substantial drafting. Read [references/architecture.md](references/architecture.md) for section design. Read [references/rigor-and-evidence.md](references/rigor-and-evidence.md) for theory, experiments, citations, and final checks.

## Write in the house style

- Lead with the mathematical object, computational problem, or concrete failure—not a broad motivational hook.
- Prefer direct declarative sentences. Put the subject and main verb early; keep one principal claim per sentence.
- Introduce notation immediately before use and explain why an object is needed after defining it.
- Move in a visible chain: problem → obstacle → construction → quantitative result → interpretation → limitation.
- Use measured connectives such as “However,” “Hence,” “Specifically,” “In contrast,” and “Crucially” only where the logical relation is real.
- State rates, assumptions, dimensions, costs, and regimes explicitly. Do not hide the main theorem behind asymptotic prose.
- Distinguish mathematical guarantees, empirical observations, heuristics, and conjectures by wording.
- Explain what each displayed equation accomplishes. Do not leave equations as an unconnected derivation transcript.
- Prefer connected prose. Use lists only for genuinely parallel objects, algorithm steps, assumptions, or a compact conclusion where the current paper already uses them.
- Keep claims conservative. Prefer “shows,” “suggests,” “tends to,” or “in our experiments” according to the evidence; avoid hype and intensifiers.
- Do not narrate draft history or failed writing attempts in the paper. Scientific comparisons belong only when they answer a research question.
- Reuse the preamble's semantic and typographic commands consistently across prose, displays, tables, captions, and algorithms. Do not bypass a user macro with an equivalent built-in command.

## Ground every claim

Never invent a theorem, proof step, citation, number, baseline, architecture, hyperparameter, hardware detail, or experimental outcome. If essential material is absent, leave a visible LaTeX comment such as:

```latex
% [MATERIAL GAP: verify the number of independent runs and report uncertainty]
```

Do not soften an unsupported claim into polished ambiguity. Either support it, qualify it accurately, or mark the gap.

For citations, verify that the bibliography key exists and the cited source supports the exact nearby statement. Separate priority claims, standard background, and methodological comparisons; each needs different evidence.

## Edit safely

- Make the smallest change that fulfills the request. Do not combine a prose revision with unrelated notation or formatting cleanup.
- Preserve manual edits and nearby content. Use targeted patches, never regenerate an entire working `.tex` file.
- Before a multi-site replacement, enumerate the exact matched sites and ensure the requested rule applies to all of them.
- Treat publication projects as read-only during inspection and checking unless the user explicitly asks for edits.
- Compile only in a temporary copy unless the user asks for an in-place build. Keep auxiliary files, logs, and rendered artifacts out of the source tree.
- When a PDF is produced, inspect the rendered pages containing the changes. A successful LaTeX exit code is not a visual check.

## Draft by section purpose

Give each section one job and each paragraph one local claim. For a new section, first form a terse internal outline of paragraph purposes. Draft prose only after the logical order is sound. Do not expose the outline unless useful to the user.

For theory, state assumptions before dependence on them, place the headline result early, unpack every term after the theorem, and make the proof skeleton visible before technical estimates. For SciML methods, define the model and orientation, state the loss/algorithm, identify how every required measure or datum is obtained, then discuss cost and failure modes. For experiments, formulate the question each benchmark isolates, specify the protocol once, report complementary diagnostics, and interpret the result without restating every table entry.

## Review in passes

Review touched text in this order:

1. **Logic:** Does each sentence follow, and does every displayed result support the stated conclusion?
2. **Evidence:** Are claims traceable to theorems, result artifacts, figures, tables, or verified citations?
3. **Notation:** Is each symbol defined once and used consistently across text, equations, captions, and supplement?
4. **Voice:** Is the prose direct, restrained, mechanism-aware, and free of generic filler?
5. **Compression:** Remove repetition, throat-clearing, and equations or sentences that do no work.
6. **LaTeX:** Check labels, references, bibliography keys, delimiters, environments, and rendering.

Report unresolved mathematical or evidentiary issues plainly. Do not declare a section finished while a material gap remains.
