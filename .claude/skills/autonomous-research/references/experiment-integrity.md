# experiment-integrity.md

System-wide rules for keeping experiments honest. Applies the moment you write eval code (W1.5), not just at audit time (Stage 1). The point: make it impossible for a "good number" to be an artifact of how the number was computed.

## The five integrity failure modes (never introduce these)

1. **Model-derived reference labels.** Reference/ground-truth targets must come from the dataset or another *declared external* source — never synthesized from the model's own outputs.
2. **Self-normalized scores.** A metric's denominator must not be derived from the model's own predictions. Normalize against fixed, external quantities.
3. **Phantom results.** Every reported number must trace to an actual output file on disk. If it isn't in a file, it isn't a result.
4. **Dead-code / unused-metric inflation.** Don't describe metrics or analysis branches that the code never actually executes. The analysis must reflect what ran.
5. **Scope inflation.** State claims only over the datasets, seeds, and settings actually tested. Generalization beyond them is a separate, unsupported claim.

## Defensive practices

- **Write metrics to files** (JSON/CSV) keyed by run id + seed + config, so claims are traceable and Stage 3 can verify them.
- **Fix seeds and log them.** Report aggregate ± spread across seeds, not a single best seed (best-seed cherry-picking is a Stage-3 finding).
- **Keep the eval script and the manuscript config in sync** — config mismatches between what ran and what's written are a blocking audit failure.
- **Separate "ran" from "planned."** Never describe a planned-but-not-run experiment as completed (also a Workflow-4 honesty gate).

## Handoff to the audit cascade

After experiments, Stage 1 (`experiment-audit`) checks code/results against the five modes and emits an `integrity_status` ∈ `pass|warn|fail`. That status propagates: a claim touched by a `fail` cannot be marked `supported` in the claim ledger (Stage 2) until resolved. See `assurance-stack.md`.
