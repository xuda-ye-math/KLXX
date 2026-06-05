# workflows.md

Step-by-step procedures for the five workflows. Each step that produces a reviewable artifact runs the **critique-to-action loop** (see `reviewer-independence.md`). Artifacts live under `.aris/artifacts/`; reviews under `.aris/reviews/<step>/`. Resume from whichever artifacts already exist.

---

## Workflow 1 — Idea Discovery
**In:** research direction. **Out:** `IDEA_REPORT.md` (ranked ideas).

1. **`research-lit`** — survey literature (multi-source: arXiv, Semantic Scholar, web). Ingest each kept paper into the wiki `papers/` (see `research-wiki.md`). Scale paper count by effort.
2. **`idea-creator`** — read the wiki `query_pack.md` (≤8 000 chars: open gaps as search seeds, rejected ideas as a banlist) and brainstorm candidate ideas. Generate via cross-model where possible.
3. **`novelty-check`** (reviewer) — for each top idea, the reviewer verifies novelty against the surveyed work and the wiki. Downgrade or drop non-novel ideas; record rejected ones in the wiki so they are not re-proposed next session.
4. **`experiment-plan`** — for the top-ranked idea, sketch the minimal experiment that could *falsify* it: datasets, baselines, metrics, seeds, compute.
5. **Review gate** (reviewer, document-only): score the ranked report for novelty, feasibility, and whether the planned experiment can actually substantiate the claim. Loop to convergence.

→ Write `IDEA_REPORT.md`.

---

## Workflow 1.5 — Experiment Bridge
**In:** `IDEA_REPORT.md` / `EXPERIMENT_PLAN.md`. **Out:** running code + `EXPERIMENT_LOG.md`.

0. **Environment bootstrap** — snapshot the environment in one pass before writing code: GPU (`nvidia-smi`), CUDA/driver, Python + key package versions, CPU/RAM, dataset/checkpoint paths → `.aris/artifacts/ENV_SNAPSHOT.md`. Reference it while coding so you match what's actually installed instead of probing for it run-time (Meta-Harness's highest-value agentic-coding change).
1. **`experiment-bridge`** — turn the plan into runnable code. Keep eval code honest from the start (see `experiment-integrity.md`): real reference labels, no self-normalized denominators, metrics computed from actual outputs.
2. **Code review** (reviewer, repository-level): score for correctness *and* integrity-failure modes before any GPU time is spent. Loop to convergence.
3. **Sanity run** — one short run on a single GPU to confirm the pipeline executes end-to-end and metrics are written to files. (Cheap validation gate: a malformed run should fail here in seconds, not after hours of GPU time.)
4. **Full run** — **(autonomous in this environment)** check `nvidia-smi` is free, then launch with `Bash run_in_background: true`; monitor to completion. Scale repetitions/seeds by effort.
   - **Auto-debug:** on failure, classify the error → apply a class-specific fix → retry (≤3). If two distinct strategies both fail, spawn a **separate rescue-diagnosis sub-agent** for an independent read before giving up.
5. **Collect** results into `EXPERIMENT_LOG.md` (commands, configs, seeds, raw metric file paths). Record an `experiments/` wiki node.

→ Then run **Stage 1 `experiment-audit`** (see `assurance-stack.md`).

---

## Workflow 2 — Auto Review Loop
**In:** draft + results. **Out:** improved paper. (The headline "research in sleep" loop.)

Each round:
1. Submit the current draft to a **fresh** reviewer for structured scoring (rubric: soundness, clarity, evidence support, novelty, presentation).
2. Extract action items; sort by severity.
3. If the reviewer requests new evidence and execution is permitted → run follow-up experiments (W1.5 sub-steps), audit them (Stage 1), update the claim ledger (Stage 2).
4. Revise affected sections.
5. **Convergence check:** stop when score ≥ threshold (default 6/10, often raised to 7.5 for this loop) and all `critical` items resolved, or after **4 rounds**.

Persist each round. Report the score trajectory (e.g. 5.0 → 7.5) and any claims pruned for lack of evidence.

---

## Workflow 3 — Paper Writing
**In:** `NARRATIVE_REPORT.md`. **Out:** compiled `paper.pdf`. Three phases:

**Plan & Generate**
1. **`paper-plan`** — structural outline + a **claims–evidence matrix** (every claim → the result that backs it).
2. **`paper-figure`** — publication-quality plots and comparison tables from the raw result files (deterministic; same data → same figure).

**Draft & Assure**
3. **`paper-write`** — section-by-section LaTeX with citation lookup, then the **five-pass scientific edit** (see `writing-principles.md`).
4. **`proof-checker`** *(theory papers only)* — 20-category taxonomy, two-axis severity, counterexample red-team → proof-obligation ledger.
5. **`paper-claim-audit`** (Stage 3, **fresh zero-context** reviewer, repository-level) — cross-check every quantitative claim against the ledger + raw files.

**Compile & Improve**
6. **`paper-compile`** — multi-pass LaTeX build; auto-repair common errors.
7. **`auto-paper-improvement-loop`** — **2 rounds** of reviewer critique on **both LaTeX source + compiled PDF** (visual review), then revise. Plus **citation audit** (see `citation-discipline.md`).

→ `paper.pdf`. Setting `auto_write: true` feeds W2 output straight into W3.

---

## Workflow 4 — Rebuttal
**In:** paper + reviewer comments. **Out:** paste-ready `REBUTTAL.md`. 7-phase pipeline with 3 safety gates:

1. **Parse** reviews into discrete, addressable points.
2. **Classify** each: factual misunderstanding / missing experiment / presentation / fundamental concern.
3. **Gate 1 — honesty:** never claim an experiment was run that wasn't, or a result that doesn't exist. Map every promised number to a real (or to-be-run) result.
4. **Draft** point-by-point responses grounded in the claim ledger.
5. **Gate 2 — new evidence:** if a response needs new results, run + audit them (W1.5 + Stage 1/2) before citing.
6. **Stress test** (reviewer): an adversarial sub-agent attacks each response as a skeptical reviewer would; patch weak spots.
7. **Gate 3 — final review** + human approval before anything is sent.

→ `REBUTTAL.md`.

---

## Full pipeline
W1 → W1.5 → (W2 ⇄ experiments) → W3 → (W4 on external feedback). Honor `human checkpoint: true` to pause at each boundary; otherwise run autonomously but always surface invalidated claims, unresolved `critical` items, and citation REPLACE/REMOVE recommendations.
