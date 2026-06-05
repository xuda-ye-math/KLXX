# writing-principles.md

The five-pass scientific-editing pipeline applied by `paper-write` after the initial draft. Inspired by scientific-writing pedagogy. Run the passes **in order** — each assumes the previous one is done.

## Pass 1 — Clutter removal
Remove filler phrases, redundant words, and unnecessary hedging. "In order to" → "to"; "it is important to note that" → delete; "very", "quite", "somewhat" → usually delete.

## Pass 2 — Active voice
Convert passive constructions to active where appropriate. "The model was trained by us" → "We train the model." Keep passive only where the agent is genuinely irrelevant.

## Pass 3 — Sentence structure
Improve topic positioning and local coherence: put the subject the sentence is *about* early; connect each sentence's opening to the prior sentence's close. Do **not** force a single sentence template — vary structure for readability.

## Pass 4 — Terminology consistency
Extract domain-specific key terms and verify consistent usage across all sections. If Methods says "validation split," later sections must say "validation split" — not "val set", "dev set", or "holdout". One concept, one name.

## Pass 5 — Numerical consistency
Cross-check every repeated numerical statement against its source table/figure/result file. A number stated in the abstract, the intro, and a table must agree. This pass feeds directly into the Stage-3 `paper-claim-audit` — fix mismatches here rather than letting the audit block submission.

---

These passes handle *prose and reporting fidelity*. They do **not** replace the evidence-to-claim audit cascade (`assurance-stack.md`): polished prose around an unsupported claim is still an unsupported claim.
