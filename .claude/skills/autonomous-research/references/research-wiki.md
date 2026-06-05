# research-wiki.md

Persistent, cross-session project memory. Turns one-shot research into **spiral learning**: failed ideas become a banlist, validated claims become foundations for the next round. Without it, an ideation pipeline re-proposes the same dead end every session.

Stored as plain Markdown under `.aris/wiki/` (file-system-as-state — any new session resumes from these files; no database, no in-memory cache).

## Layout

```
.aris/wiki/
  index.md                 # node registry + the relationship graph
  query_pack.md            # ≤8000-char compressed summary fed to idea-creator
  papers/<id>.md
  ideas/<id>.md            # INCLUDING rejected ideas (the banlist)
  experiments/<id>.md
  claims/<id>.md
```

Node IDs are canonical and stable (e.g. `paper-2024-vaswani-attention`, `idea-007`, `exp-012`, `claim-031`).

## Four entity types

- **papers** — surveyed literature. Fields: id, title, authors, year, venue, key contribution, relevance, gaps it leaves open.
- **ideas** — candidate research directions. Fields: id, statement, motivating gap, status (`active | rejected | validated`), rejection reason (if any). **Never delete rejected ideas** — they are the banlist.
- **experiments** — runs. Fields: id, hypothesis, setup (datasets/seeds/config), command, raw result file paths, outcome, `experiment-audit` integrity_status.
- **claims** — tracked claims. Fields: id, statement, verdict (`supported | partially_supported | invalidated`), evidence pointers, integrity_status.

## Eight typed relationships (the knowledge graph)

`extends`, `contradicts`, `addresses_gap`, `inspired_by`, `tested_by`, `supports`, `invalidates`, `supersedes`.

Record edges in `index.md`, e.g.:
```
idea-007 addresses_gap paper-2024-vaswani-attention
idea-007 tested_by exp-012
exp-012 supports claim-031
claim-031 supersedes claim-018
idea-003 status=rejected (reason: subsumed by paper-2025-x)
```

## Integration points

- **`research-lit`** ingests discovered papers as `papers/` pages.
- **`idea-creator`** reads `query_pack.md` *before* ideating — uses listed gaps as search seeds and rejected ideas to avoid revisiting dead ends.
- **`result-to-claim`** updates `claims/` verdicts after each experiment.

## Session protocol

1. **On start:** read `index.md` + `query_pack.md`. Do not re-propose anything in the rejected list; do not re-derive an already-`supported` claim.
2. **During:** create/update nodes and edges as artifacts are produced.
3. **On end:** refresh `query_pack.md` (open gaps + rejected ideas + validated claims) so the next session starts informed, not blank.
