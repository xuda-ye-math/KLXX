# .aris/config.md — run configuration (copy to project root .aris/config.md)

```
effort: balanced              # lite | balanced | max | beast
reviewer_backend: claude-subagent   # claude-subagent | codex | gemini | llm | oracle-pro
review_threshold: 6           # accept artifact at >= this score
max_review_rounds: 4
reviewer_bias_guard: true     # fresh reviewer context each round
human_checkpoint: false       # true => pause at each workflow boundary
experiment_execution: autonomous   # autonomous | confirm | dry-run
gpu: local                    # local | ssh | vast | modal
```

## Detected reviewer backends
<!-- filled at run start via: command -v codex gemini llm -->
- cross-family CLI available: <none | codex | gemini | llm>
- note: default is fresh-context Claude sub-agent (same family — scores advisory)

## Always-surface items (never auto-decided, even when human_checkpoint=false)
- invalidated claims
- unresolved `critical` review items
- citation REPLACE / REMOVE recommendations
- proposed harness / meta-optimization patches
