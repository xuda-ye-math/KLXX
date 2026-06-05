# effort-contract.md

How effort presets and reviewer routing scale a run. The governing invariant: **effort scales coverage and iteration — never the reviewer's reasoning depth.**

## Effort presets

| Preset | Multiplier | Effect |
|---|---|---|
| `lite` | ~0.4× | Fewer papers surveyed, fewer ideas, fewer review rounds. Quick exploration. |
| `balanced` | 1× (default) | Standard behavior. |
| `max` | ~2.5× | Deeper literature search, more thorough review, more experiment repetitions/seeds. |
| `beast` | ~5–8× | Breadth/iteration toward upper bounds. Long autonomous runs. |

What scales: papers surveyed, ideas generated, seeds/repetitions, review rounds (within the max-round cap), search depth.

**Invariant:** the reviewer always reasons at maximum depth (the paper's `xhigh`-equivalent) regardless of preset. A `lite` run reviews *less often*, not *less carefully*. Convergence threshold (6/10) and max rounds (4) are defaults you may raise per workflow (e.g. 7.5 for the auto-review loop) but should not silently lower.

Override inline: `effort: max`.

## Reviewer routing

- **Default (this environment):** independent **fresh-context Claude sub-agent** via the `Agent` tool (same family — treat scores as advisory; see SKILL.md "Reviewer model policy").
- **Cross-family upgrade (preferred when available):** route to a non-Claude reviewer — a `codex`/`gemini`/`llm` CLI, or an OpenAI-compatible endpoint. Detect at run start (`command -v codex gemini llm`); record what's available in `.aris/config.md`. Cross-family review produces less-correlated critiques and is the paper's recommended configuration.
- **High-stakes:** for critical reviews, use the strongest available reviewer and require human sign-off. Override inline: `reviewer: <backend>`.

Whatever the backend, the **reviewer-independence protocol** (`reviewer-independence.md`) always applies: paths + objective in, structured score + action items out, reviewer reads files directly.

## Human checkpoints

- `human checkpoint: true` → pause for approval at every workflow boundary.
- Otherwise run autonomously, but **always** surface (never auto-decide): invalidated claims, unresolved `critical` review items, citation REPLACE/REMOVE recommendations, and any harness/meta-optimization patch.

## `.aris/config.md` keys

```
effort: balanced
reviewer_backend: claude-subagent   # or codex / gemini / llm / oracle-pro
review_threshold: 6
max_review_rounds: 4
reviewer_bias_guard: true           # fresh context each round
human_checkpoint: false
experiment_execution: autonomous    # autonomous | confirm | dry-run
gpu: local
```
