# assurance-stack.md

The assurance stack is the harness's answer to **plausible unsupported success**. The general critique loop catches obvious problems; the stack catches the specific ways an executor inflates results to please a reviewer. Each stage is independently invocable; in the full pipeline they run in order.

Conceptually the cascade moves: **code-level integrity → evidence-to-claim interpretation → manuscript reporting fidelity.**

---

## Stage 1 — `experiment-audit` (reviewer, repository-level)

A cross-model/independent reviewer audits eval code + result files for these integrity failure modes:

1. **Model-derived reference labels** — reference targets synthesized from model outputs instead of the dataset or another declared source.
2. **Self-normalized scores** — metrics whose denominator is derived from the model's own predictions (can silently inflate performance).
3. **Phantom results** — claimed numbers that don't match any actual output file.
4. **Dead-code / unused-metric inflation** — metrics or branches described in the analysis but never actually executed.
5. **Scope inflation** — claims generalized beyond the tested datasets/seeds/settings.

**Output:** `EXPERIMENT_AUDIT.md` (human-readable) + `EXPERIMENT_AUDIT.json` (machine-readable, with an `integrity_status` ∈ `pass|warn|fail` per checked item). **Advisory** — it does not halt execution, but its statuses propagate into Stage 2.

---

## Stage 2 — `result-to-claim`

For each candidate experimental claim, assign exactly one verdict against the available evidence:

- **`supported`** — evidence directly substantiates the claim.
- **`partially_supported`** — holds under narrower conditions than stated.
- **`invalidated`** — evidence contradicts or fails to support it.

Propagate Stage-1 `integrity_status` onto each claim: **a claim touched by a `fail` cannot be marked `supported`** until the integrity issue is resolved.

**Output:** the **claim ledger** `CLAIM_LEDGER.md` — each claim → supporting/qualifying/contradicting evidence + verdict + integrity status. This ledger is the single source of truth that Stage 3 audits the manuscript against. (Template: `templates/CLAIM_LEDGER.md`.)

---

## Stage 3 — `paper-claim-audit` (reviewer, FRESH zero-context, repository-level)

Spawn a **brand-new** `Agent` (no prior conversation context) so neither your framing nor accumulated reviewer expectations bias it. It reads the manuscript LaTeX **plus** raw result/config files and cross-checks every quantitative claim.

Per-claim status:
- `exact_match` — manuscript number matches the source exactly.
- `rounding_ok` — differs only by acceptable rounding.
- `number_mismatch` — manuscript number ≠ source.
- `config_mismatch` — manuscript describes a different config than the experiment used.
- `missing_evidence` — no source file backs the number.

Representative checks: numerical mismatches, best-seed cherry-picking, config mismatches, aggregation / delta-arithmetic errors, scope overclaim. Any `number_mismatch` / `config_mismatch` / `missing_evidence` is a blocking fix before submission.

---

## Manuscript assurance (Workflow 3)

- **Five-pass scientific editing** — see `writing-principles.md`. Applied by `paper-write` after drafting.
- **Proof verification** (`proof-checker`, theory papers) — 20-category issue taxonomy; **two-axis severity**: *proof status* (invalid / unjustified / unclear) × *impact* (global / local / cosmetic). Verifies theorem applications against side-condition checklists; runs a **counterexample red-team** on key lemmas and main guarantees. Output: a **proof-obligation ledger** recording the status of each theorem/lemma/derived obligation.
- **Visual PDF review** (`auto-paper-improvement-loop`) — send **both** the LaTeX source **and** the compiled PDF to the reviewer. Source → substantive content; PDF → presentation: figure readability, caption↔figure alignment, layout (orphaned headers, misplaced floats), table formatting, color consistency across figures. Dual-input catches what source-only review misses.
- **Citation audit** — see `citation-discipline.md`.

---

## Limits
The cascade catches *common* integrity failures; it is **not** formal verification and cannot detect every fabrication. Treat every `warn`/`fail`/`invalidated`/mismatch as a thing to fix or surface to the human — never as noise to suppress.
