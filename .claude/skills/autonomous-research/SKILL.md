---
name: autonomous-research
description: Run autonomous ML/scientific research end-to-end using adversarial multi-agent collaboration (an ARIS-style harness). Invoke for "/autonomous-research", or when the user wants to go from a research direction/idea to validated experiments and a written paper with independent review at every step — idea discovery, experiment bridge, auto-review, paper writing, or rebuttal. The executor (you) drives progress; an independent fresh-context reviewer sub-agent critiques every artifact before it is accepted.
---

# Autonomous Research (ARIS-style harness)

A **research harness**: a stateful system that decomposes long-horizon research into reviewable sub-workflows and gates every artifact behind an *independent* reviewer. You are the **executor**. You spawn **reviewer** sub-agents that critique your work adversarially before any artifact is accepted.

## Operating assumption (do not relax)

> **Any long-horizon task done by a single agent is unreliable.** Divide the workflow into sub-workflows and have an independent reviewer check the output of each step.

The failure mode you are defending against is **not** visible breakdown — it is **plausible unsupported success**: results that are real but misreported, claims that outrun their evidence, and a reader who silently inherits your framing. Your job as executor is to *make progress*; the reviewer's job is to *find where the evidence does not license the claim*. Never let the executor role grade its own homework.

## Three invariants

1. **Independent assurance.** Every artifact (code, results, manuscript section, claim) is reviewed by a *separate* agent before acceptance. The reviewer reads the artifact files **directly** — you pass file paths + a review objective, never a summary. See `references/reviewer-independence.md`.
2. **Modular execution.** The trajectory is split into replaceable steps that exchange **plain-text artifact contracts** (Markdown/JSON files under `.aris/`), not hidden in one opaque transcript. Any step can be re-run from saved artifacts.
3. **Persistent state.** A per-project **research wiki** under `.aris/wiki/` records papers, ideas, experiments, and claims across sessions — including *rejected* ideas, so dead ends are not re-tried. See `references/research-wiki.md`.

> **Raw evidence over summaries** (the principle behind all three invariants). Never substitute a compressed summary for the underlying evidence. A reviewer that reads your summary inherits your blind spots; a meta-optimizer that reads a statistics digest cannot trace a failure to its cause. Meta-Harness (Lee et al. 2026) shows this empirically: giving an agent raw execution traces beats giving it scores+summaries by **~15 points** (50.0 vs 34.9 median) — summaries *hurt* by compressing away diagnostic signal. So: reviewers read artifact files directly, traces are kept raw and queried with `grep`/`cat`, and diagnosis works over full history, not digests. See `references/meta-harness.md`.

## Reviewer model policy

The paper's recommended configuration is **cross-family** review (e.g. Claude executor + GPT/Gemini reviewer) because mixed-family critiques are less correlated. In this environment the reviewer is an **independent fresh-context Claude sub-agent** spawned via the `Agent` tool with an adversarial persona. This breaks *context* and *role* blind spots but shares model family — treat reviewer scores as advisory, not ground truth, and prefer the strictest interpretation when executor and reviewer disagree.

**Upgrade path (do this if available):** if a non-Claude reviewer is reachable — a `codex`/`gemini`/`llm` CLI on `PATH`, or an OpenAI-compatible endpoint via env keys — route reviews there instead for true cross-family review. Detect with `command -v codex gemini llm` at the start of a run and record what you find in `.aris/config.md`.

## How to run a reviewer (the core primitive)

Spawn reviewers with the `Agent` tool. **The orchestrator never reviews its own output.**

- **Document-only** review (manuscript prose, ideas): `subagent_type: "Explore"` (read-only — it cannot edit, which enforces independence).
- **Artifact-augmented / repository-level** review (experiment audit, claim audit): `subagent_type: "general-purpose"` (needs Bash/Read/Grep to inspect code and outputs).
- **Fresh context** (default, `REVIEWER_BIAS_GUARD=true`): start a *new* `Agent` each round so it carries no prior expectations. Required for `paper-claim-audit` and the paper-improvement loop. Use **cross-round** (continue the same agent via `SendMessage`) only when you specifically need it to verify that previously-raised issues were fixed.

Every reviewer prompt must contain, in order:
1. **Role:** "You are an adversarial peer reviewer. You did not write this artifact. Be skeptical. Assume claims are unsupported until the files prove otherwise."
2. **Objective + scope:** what to assess and which access scope (document-only / artifact-augmented / repository-level).
3. **File paths to read directly** — never your summary of them.
4. **Rubric** and the **required structured output** (score `/10`, then a JSON block of action items). See the template in `references/reviewer-independence.md`.

## The critique-to-action loop

Every workflow step that produces a reviewable artifact runs this loop:

1. **Executor** produces the artifact and writes it to its `.aris/` contract path.
2. **Reviewer** (fresh sub-agent) reads the files directly, scores against the rubric, returns structured action items (each tagged `critical` / `major` / `minor`).
3. **Executor** addresses items. For any `critical`/`major` item you must attempt **≥2 distinct remediation strategies** before marking it `unresolved`.
4. **Convergence check** — accept the artifact when **score ≥ threshold (default 6/10) AND all `critical` items resolved**, else loop again.
5. **Stop** at convergence or after **max rounds (default 4)**. Persist each round's review under `.aris/reviews/<step>/round-N.md`.

Watch for **reviewer-bias amplification**: if the reviewer keeps demanding one methodology, you may be overfitting to its preferences rather than improving the science. Note divergences for the human instead of blindly complying past diminishing returns.

## Workflows

Drive the workflow the user asks for; if they just give a direction/idea, run the full pipeline in order. Each is detailed step-by-step in `references/workflows.md`.

| # | Workflow | Input | Output | Artifact contract |
|---|----------|-------|--------|-------------------|
| 1 | **Idea Discovery** | Research direction | Ranked idea report | → `IDEA_REPORT.md` |
| 1.5 | **Experiment Bridge** | Experiment plan | Running code + results | `IDEA_REPORT.md` → `EXPERIMENT_PLAN.md`, `EXPERIMENT_LOG.md` |
| 2 | **Auto Review Loop** | Draft + results | Improved paper | `EXPERIMENT_LOG.md` → revised draft |
| 3 | **Paper Writing** | Narrative report | Compiled PDF | `NARRATIVE_REPORT.md` → `paper.pdf` |
| 4 | **Rebuttal** | Paper + reviews | Paste-ready rebuttal | reviews → `REBUTTAL.md` |

**Full pipeline:** W1 → W1.5 → (W2 ⇄ experiments) → W3 → (W4 on feedback). Chain them through the artifact contracts; resume from whichever artifacts already exist.

### Experiment execution policy (this environment)

Experiments **run autonomously** on the local GPU. At the W1.5 execution step:
1. **Environment bootstrap (do this once, first).** Before writing experiment code, capture the environment in one snapshot and append it to your working context: GPU (`nvidia-smi`), CUDA/driver, Python + key package versions (`torch`, etc.), CPU/RAM, and dataset/checkpoint paths. Write it to `.aris/artifacts/ENV_SNAPSHOT.md`. This eliminates the 2–4 wasted turns agents otherwise spend probing what's installed — Meta-Harness found this the single highest-value harness change for autonomous long-horizon runs.
2. **Sanity-check** the script on a single short run.
3. Confirm the GPU is free (`nvidia-smi`), then launch the full job(s) with `Bash run_in_background: true`, monitor to completion, and collect results into `EXPERIMENT_LOG.md`.

**Auto-debug on failure:** classify the error, apply a class-specific fix, retry up to **3 times**; if two distinct strategies both fail, spawn a *separate* "rescue diagnosis" sub-agent for an independent read before giving up. (`sudo` is never run by you — print such commands for the user.)

### Process discipline — long-running GPU jobs must outlive their launcher

The harness can silently reap process trees when a session closes, and `dmesg`/stderr show **nothing**: no kill log, no error, the python just disappears. Discovered the hard way (2026-05-27 run; multiple pythons simultaneously killed mid-training with no trace). Symptoms and rules:

1. **Never have an agent launch a long-running python that must outlive its session.** A python started inside an agent via `Bash run_in_background: true` (or `nohup &`) is a child of that *agent's bash*. When the agent's turn ends — which the executor cannot directly control, and which can happen at any tool-budget boundary — the harness reaps the agent's process tree, killing the python.
   - **Rule:** GPU work is launched by the **executor** (root agent), from the executor's session, via `Bash run_in_background: true`. The harness tracks these and emits a clean completion notification on exit; they survive agent-spawn/death cycles.
   - **If an agent must run a python**, it runs it **synchronously** (no `run_in_background`, no `nohup`), so the python's lifetime is bounded by the agent's session. Death-by-cleanup becomes structurally impossible.

2. **Foreground Bash with `nohup python &` is also fragile.** Even from the executor, `nohup` does not reliably keep the python alive across the harness's session-cleanup boundaries. Use `Bash run_in_background: true` instead — it is the only documented-stable path.

3. **Zombie agent wake-ups produce hallucinated "completion" notifications.** After a reaped agent's python is gone, the agent's notification process may wake up briefly and emit `<status>completed</status>` messages with confidently-stated numbers (ESS, mode counts, etc.) that **do not correspond to any real run** — they are the agent's pre-death expectations replayed. **Never trust agent-summary numbers**; verify by reading the saved `data.pth` / results CSV via the project's `build_table.py`. The files are ground truth.

4. **Periodic `nvidia-smi` checks are mandatory during long runs.** Every few minutes during a multi-hour sweep, run:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
   pgrep -af "python train" | grep -v "claude\|grep"
   ```
   to confirm (a) the expected pythons (and *only* those) are alive, (b) no OOM brewing, (c) no zombie reruns clobbering completed `data.pth`. Identify any unknown python via `/proc/<PID>/cwd` and `cat /proc/<PID>/cmdline`; kill zombies with plain `kill <PID>` (avoid `pkill -f` patterns that match the kill command itself and self-kill the shell).

5. **Same-folder duplicate launches race on `data.pth`.** If two pythons in the same folder finish near-simultaneously, the later writer wins, possibly with a partial / wrong result. Before launching a full run, `pgrep -af python` and kill any in-folder duplicate.

6. **Stale Monitor markers must be cleared between launches.** A polling Monitor that touches `/tmp/<marker>` to dedupe "RUN DONE" events will silently miss subsequent completions (e.g. sanity → full) in the same folder unless the marker is `rm`'d between phases.

### User-side monitor & visualization — make results handy, not buried

The harness happily writes everything to `.pth` files and per-step logs the executor can read, but **the human on the other end opens the project folder in an IDE/file viewer** and wants the headline result to be immediately legible without parsing tensors or scrolling 20 k lines of training log. Make the project folder self-explanatory:

1. **Always emit, in the project's own folder, three classes of artifact at minimum:**
   - **`results_table.md` + `results_table.csv`** — the headline numbers in a Markdown table the user reads in-place plus a CSV for re-plotting. Not just buried in `data.pth`.
   - **`summary.md`** — one page: setup, headline table, one-paragraph interpretation, caveats, file index. The user should be able to grok the experiment in <60 s by opening this file. (Distinct from the raw status log.)
   - **At least one PNG figure** — render the headline visualization with `matplotlib`, write next to `data.pth`. For an ESS-discriminator experiment that means an **ESS-over-step trajectory** plot (one colored line per method, like Chessboard / Figure 3 of the X-functional paper): raw per-step ESS faint, moving-average bold. For a mode-discovery experiment, a **per-mode occupancy bar chart** (4 methods × N modes) makes the fake-ESS pitfall and the X-rescue visible at a glance. CSV/MD tables convey numbers; PNGs convey shape — the user wants both.
2. **Status log in the project folder, not stdout.** Long runs append `train_status.log` (or per-tag variants) in the project's own folder; the user `tail -f`'s it to see live progress without reaching into `/tmp` or the harness's internal task transcripts. One line per progress event with timestamp + method + percent + ms/step + ETA + the key live metric (e.g. running ESS); flush each write. No `tqdm` (won't tail cleanly).
3. **A single live `SUMMARY.md`/`STATUS.md` at the sweep root** for multi-project sweeps. Updated at each project boundary (per-k DONE or per-method DONE) with the latest table-so-far so the user can refresh one file and see all completed columns. This is the "dashboard" — distinct from per-project `summary.md`.
4. **Don't put results in tool-call output the user has to scroll the transcript to find.** Console summaries from `build_table.py` etc. are fine as a duplicate, but the *canonical* place is always a file in the project folder. Rule of thumb: if you'd point a human at a result, you should be able to give them a *file path*, not a tool-call ID.
5. **Auto-trigger the plot + table rebuild at run end.** The training script should end by calling its `build_table.py` and `plot_results.py` (or invoke them in the same shell after the python run), so the user finds an up-to-date PNG/MD next to the `.pth` without a separate manual step.
6. **PDFs/figures referenced from a paper draft live next to the data, not in `/tmp`.** Generated figure paths should be stable relative paths (e.g. `Project/figures/headline.png`) the LaTeX `\includegraphics` line can point at and the user can preview in their IDE.

The litmus test: a user opening the project folder cold should be able to (a) see the headline number in a table, (b) see *why* in a figure, (c) read a one-page interpretation in `summary.md`, and (d) tail the live log if a run is in progress — without ever opening a `.pth`, the executor's transcript, or a `/tmp/` path.

## Assurance stack — run these as gates, not afterthoughts

The critique loop is general; the assurance stack catches the specific ways an executor inflates results. Full procedures in `references/assurance-stack.md`.

**Evidence-to-claim audit cascade** (run in order):
- **Stage 1 — `experiment-audit`** (reviewer, repository-level): audits eval code + result files for the 5 integrity failure modes (model-derived reference labels, self-normalized scores, phantom results, dead-code/unused-metric inflation, scope inflation). Writes `EXPERIMENT_AUDIT.md` + `.json`. Advisory: doesn't halt, but its `integrity_status` propagates downstream.
- **Stage 2 — `result-to-claim`**: maps each candidate claim to a verdict — `supported` / `partially_supported` / `invalidated` — producing the **claim ledger** `CLAIM_LEDGER.md`. A claim with a Stage-1 `fail` cannot be `supported` until resolved.
- **Stage 3 — `paper-claim-audit`** (reviewer, **fresh zero-context**, repository-level): a brand-new sub-agent cross-checks every quantitative claim in the manuscript against the ledger + raw result files. Per-claim status: `exact_match` / `rounding_ok` / `number_mismatch` / `config_mismatch` / `missing_evidence`.

**Manuscript assurance:**
- **Five-pass scientific editing** (in `paper-write`): clutter removal → active voice → sentence structure → terminology consistency → numerical consistency. See `references/writing-principles.md`.
- **Proof verification** (theory papers): 20-category issue taxonomy, two-axis severity (status × impact), counterexample red-team → proof-obligation ledger.
- **Visual PDF review**: send **both** LaTeX source *and* compiled PDF to the reviewer — catches figure readability, caption/figure alignment, float placement, table formatting, color consistency.
- **Citation audit**: verify every `\cite` on three axes — existence, metadata correctness, **context appropriateness** (the cited paper actually supports the claim). Verdicts: KEEP/FIX/REPLACE/REMOVE for human approval. See `references/citation-discipline.md`.

## Effort & configuration

Read `.aris/config.md` at the start of a run (create it from the defaults below if missing). Honor inline overrides like `effort: max`, `reviewer: <backend>`, `human checkpoint: true`. Full table in `references/effort-contract.md`.

- **Effort presets** scale breadth/depth/iteration only — **never** the reviewer's reasoning depth (that stays maxed): `lite` (~0.4×), `balanced` (1×, default), `max` (~2.5×), `beast` (~5–8×).
- **Human checkpoints**: if `human checkpoint: true`, pause for approval at each workflow boundary. Otherwise run autonomously but **always** surface: invalidated claims, unresolved `critical` review items, and citation REPLACE/REMOVE recommendations.

## Bootstrapping a run

1. Ensure the project scaffold exists (create if missing):
   ```
   .aris/
     config.md              # effort, reviewer backend, gates
     wiki/{papers,ideas,experiments,claims}/  + index.md
     artifacts/             # the artifact contracts above
     reviews/<step>/        # per-round reviewer reports
     meta/events.jsonl      # passive event log (see below)
   ```
2. Detect available reviewer backends (`command -v codex gemini llm`); record in `.aris/config.md`.
3. Load the wiki `index.md` so you don't re-propose rejected ideas or re-derive known claims.
4. Run the requested workflow via the critique-to-action loop, persisting every artifact and review.
5. End with a status summary: artifacts produced, claim-ledger verdicts, unresolved critical items, and any human-decision items.

## Meta-optimization (advisory only)

The harness *is* code — this `SKILL.md` and its reference docs. Meta-optimization is outer-loop improvement of that code, and it follows the **raw-evidence** principle above.

- **Keep raw traces, not digests.** Append a one-line index entry per significant step to `.aris/meta/events.jsonl` (`{ts, step, tool, status, overrides, trace_path}`) — but treat it only as an *index* into the full artifacts (`reviews/`, `EXPERIMENT_LOG.md`, debug attempts, prior artifact versions). Do **not** throw away the raw traces; the diagnostic signal lives there, not in the digest.
- **Diagnose over full history, openly.** When asked to `/meta-optimize`, read *raw* prior traces across many past runs (`grep`/`cat` what's relevant — don't ingest everything) and reason about *why* a default failed: which override recurs, which tool fails repeatedly, where review scores plateau, what confound links them. Form a causal hypothesis (Meta-Harness's qualitative finding: this kind of "isolate the confound, then make the minimal safe change" reasoning is exactly what raw history enables and digests cannot). The skill constrains *what's forbidden and what to produce* — it does not script your diagnosis.
- **Edit in code space + validate cheaply.** Propose concrete patches to `SKILL.md` / reference docs. Before evaluating a patch, run a cheap sanity check (does it still parse / preserve the invariants?); then judge it against a small set of past traces where the current harness underperformed — under **Pareto dominance over (quality, token cost)**, not a single scalar.
- **Reviewer-gated, never auto-applied.** A fresh reviewer scores each proposed patch; surface only those ≥7/10. The human makes the final call — `aris` never auto-applies harness changes.

See `references/meta-harness.md` for the full design and Meta-Harness's implementation tips.

## Hard limits

- This harness cannot guarantee correctness, novelty, or soundness. The audit cascade is an advisory safety net, not formal verification.
- Repository-level review may send source to an external model — never enable it on repos with secrets/sensitive code without an approved local-only path.
- Humans own research direction, evidence validation, and the final submission decision. You automate execution and review loops, not accountability.
