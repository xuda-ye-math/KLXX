# citation-discipline.md

Rules for citations and the citation audit. LLM-drafted bibliographies fabricate references and misattribute claims; grounding reduces but does not eliminate this.

## While drafting

- **Look up before you cite.** Resolve every reference against a canonical source (arXiv, DBLP, CrossRef, ACL Anthology, OpenReview, Nature) before adding the `\cite`. Never invent an arXiv ID, DOI, or author list.
- **Cite for the claim you're making.** The cited paper must actually establish what you use it to support — not merely be topically adjacent.

## Citation audit (Workflow 3)

Verify **every** `\cite` along three independent axes, using **fresh cross-family reviewers with web access** where available:

1. **Existence** — the cited paper resolves at the claimed arXiv ID / DOI / venue.
2. **Metadata correctness** — authors, year, venue, title match canonical sources.
3. **Context appropriateness** — the cited paper actually establishes the claim it supports. *This is the most diagnostic axis:* a real paper used to back a wrong claim is a credibility failure that metadata-only checks miss.

## Verdicts → per-entry ledger

Each entry gets one recommendation, surfaced for **human approval before submission**:

- **KEEP** — all three axes pass.
- **FIX** — exists and is appropriate, but metadata is wrong → correct the entry.
- **REPLACE** — claim is real but this citation doesn't support it → find the correct source.
- **REMOVE** — fabricated, unresolvable, or unsupportable.

Record verdicts per entry (id, axes results, recommendation, note). REPLACE/REMOVE are never auto-applied — the human decides.
