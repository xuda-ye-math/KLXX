# reviewer-independence.md

The shared protocol every review step must follow. The whole harness rests on this: if the reviewer assesses *your framing* instead of *the work*, the review is theater.

## The one rule

**The reviewer reads the artifact files directly. You pass file paths + a review objective — never a summary, never your interpretation, never the numbers you want confirmed.** If you pre-digest the artifact, the reviewer inherits your blind spots and the cross-check collapses into self-agreement.

## Spawning a reviewer

Use the `Agent` tool. Choose `subagent_type` by access scope:

| Access scope | What the reviewer may read | subagent_type |
|---|---|---|
| **Document-only** | Manuscript text only. Default for prose/idea review. | `Explore` (read-only — can't edit, so independence is structural) |
| **Artifact-augmented** | Manuscript + result files, claim ledger, intermediate artifacts it references. | `general-purpose` |
| **Repository-level** | Codebase, eval scripts, generated outputs (via Bash/Read/Grep). Used by `experiment-audit`, `paper-claim-audit`. | `general-purpose` |

Context policy:
- **Fresh** (default, `REVIEWER_BIAS_GUARD=true`): a *new* `Agent` call each round, zero prior context. Prevents confirmation bias. **Required** for `paper-claim-audit` and `auto-paper-improvement-loop`.
- **Cross-round**: continue the same reviewer via `SendMessage` so it can verify whether previously-raised issues were fixed. Use only when convergence-verification matters more than independence.

## Reviewer prompt template

```
ROLE
You are an adversarial peer reviewer. You did NOT write this artifact. Your job is to
find where claims are not supported by evidence. Assume nothing is supported until the
files prove it. Do not trust any summary — read the files yourself.

OBJECTIVE
<what to assess, e.g. "Score this experiment design for soundness and whether the
proposed metrics can actually substantiate the idea's claims.">

ACCESS SCOPE: <document-only | artifact-augmented | repository-level>

FILES TO READ (read these directly, in full):
- <abs path>
- <abs path>

RUBRIC (score each 1-10, then an overall /10):
- <criterion 1>
- <criterion 2>
- ...

REQUIRED OUTPUT
1. One paragraph: your independent assessment (not a restatement of the artifact).
2. Overall score: N/10.
3. A JSON block of action items:

```json
{
  "overall_score": 0,
  "action_items": [
    {"id": 1, "severity": "critical|major|minor", "claim_or_section": "...",
     "issue": "...", "evidence": "what in the files shows this", "fix": "concrete ask"}
  ],
  "unsupported_claims": ["..."],
  "verdict": "accept | revise | reject"
}
```
```

## Convergence

Accept the artifact when **overall_score ≥ threshold (default 6/10) AND every `critical` item is resolved**. Otherwise loop. Stop after **max rounds (default 4)** regardless, and report unresolved items rather than silently dropping them.

Before marking any `critical`/`major` item `unresolved`, attempt **≥2 distinct remediation strategies**. Record each round at `.aris/reviews/<step>/round-N.md`.

## Anti-gaming notes

- The executor will be tempted to satisfy the reviewer cheaply (rewording rather than fixing the evidence gap). Don't. A reviewer item about *evidence* is resolved only by *evidence*.
- When executor and reviewer are the same model family (the local default), assume correlated blind spots remain. Prefer the stricter reading on disagreement, and flag for the human anything the reviewer waved through that still feels unsupported to you.
