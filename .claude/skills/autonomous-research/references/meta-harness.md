# meta-harness.md

Harness-engineering principles drawn from **Meta-Harness: End-to-End Optimization of Model Harnesses** (Lee, Nair, Zhang, Lee, Khattab, Finn, 2026 — `lee2026metaharness`, the harness paper ARIS itself cites). The *harness* is the code that decides what to store, retrieve, and present to the model; it can swing performance by up to 6× on a fixed model. This harness — `SKILL.md` + the reference docs — is exactly such code, so these lessons apply to running *and* improving it.

## The central empirical result: raw traces ≫ summaries

Meta-Harness searches over harness code with a coding-agent proposer that has **full filesystem access** to every prior candidate's source, scores, and **raw execution traces**, retrieved adaptively with `grep`/`cat` rather than pre-digested.

The ablation that matters (online text classification, median / best accuracy):

| Proposer sees | Median | Best |
|---|---|---|
| Scores only | 34.6 | 41.3 |
| Scores + LLM summaries | 34.9 | 38.7 |
| **Full raw traces** | **50.0** | **56.7** |

Summaries barely beat scores-only and can *hurt* (best-acc drops) — they compress away the diagnostic detail needed to trace a downstream failure back to an earlier decision. **Lesson: never replace raw evidence with a summary.** This is the single principle behind both of ARIS's cross-checks:
- **reviewer-independence** — the reviewer reads artifact files directly, never your summary (`reviewer-independence.md`);
- **meta-optimization** — diagnosis reads raw traces across many runs, never a statistics digest.

## Causal reasoning over prior failures (why full history pays off)

Meta-Harness's qualitative log shows the proposer doing more than random mutation: after several regressions it *identifies a confound* ("the regressions came from the prompt rewrite, not the structural fix"), *isolates* the proven change, and *pivots to a minimal, additive, lower-risk modification* — even transferring lessons across separate runs. This is the behavior to emulate in `/meta-optimize`: read enough raw history to form a causal hypothesis, then make the smallest safe change. Digest-only optimizers structurally cannot do this.

## Optimize the harness in code space

- The harness is a program; small changes to retrieval/memory/prompt logic affect behavior many steps later, so local-edit heuristics fit poorly. Search/edit at the level of structure, not template-filling.
- Keep the **outer loop minimal**: don't hard-code the diagnosis procedure. The skill should constrain *what is forbidden, what artifacts to produce, and what to optimize* — and otherwise let the agent inspect traces and decide what to change. (This is why ARIS's research workflows are prescriptive about integrity gates but meta-optimization is left open-ended.)
- Evaluate candidates under **Pareto dominance** over multiple objectives (e.g. review quality vs. token/context cost), not one scalar. Report the frontier; don't commit to one operating point prematurely.
- The proposer **never sees test results** — improvement signal comes only from a held-out-from-final **search set**.

## Practical implementation tips (Meta-Harness Appendix D), mapped to ARIS

- **The skill text is the primary steering lever.** Its quality matters more than iteration count. Constrain outputs and safety, not the diagnosis path. → keep `SKILL.md` sharp; this is the thing meta-optimization edits.
- **Environment bootstrap.** Gather an environment snapshot (OS, languages, package versions, GPU, dir listing) in one guarded command *before* the agent loop, and prepend it. Eliminates 2–4 wasted probing turns; biggest single win on long-horizon agentic tasks. → ARIS does this in W1.5 as `ENV_SNAPSHOT.md` (`SKILL.md` experiment-execution policy).
- **Lightweight validation before expensive evaluation.** A cheap interface/sanity test should reject malformed candidates in seconds. → ARIS's W1.5 sanity run, and the parse/invariant check before evaluating a harness patch.
- **Log everything navigably.** Machine-readable (JSON), hierarchical, consistent names, regex-friendly. → the `.aris/` layout; keep raw traces, treat `events.jsonl` as an index only.
- **Start from a baseline + a search set that's *hard* for it.** A fast, discriminative eval beats a large one; ~50 instances is enough. → meta-optimize against past traces where the current harness underperformed.
- **(Optional) a small CLI over the logs** — list the Pareto frontier, show top-k, diff runs — makes the experience store easier to query as it grows.
- **Automate evaluation outside the proposer.** A separate scorer writes results to the filesystem; the proposer just reads them.

## Caveat

Full-history inspection is **expensive** (Meta-Harness uses ~10 MTok/iteration vs. ≤0.03 for digest-based optimizers). So treat deep `/meta-optimize` passes as occasional, not per-run, and use the cheap validation gate to keep the cost of bad candidates near zero.
