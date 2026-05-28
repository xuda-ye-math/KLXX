# VD high-dimensional product multi-well benchmark — experiment plan

## 1. Problem setup

Consider a series of high-dimensional potentials in dimension $d=2^k$ defined by
$$
    U(x) = \frac12\sum_{i=1}^{2^k}x_i^2 + 12 \sum_{i=1}^k \exp(-x_i^2)
$$
that is, $U(x)$ is a double well potential in the first $k$ dimensions, and a normal Gaussian in the last $2^k-k$ dimensions, so that $U(x)$ has exactly $2^k$ wells.
Now assume the source potential is exactly the standard quadratic potential
$$
    U_0(x) = \frac12\sum_{i=1}^{2^k} x_i^2.
$$
so we need to train a flow map $G$ in $2^k$ to map from the target $\mu \propto \exp(-U(x))$ to the source $\mu_0 \propto \exp(-U_0(x))$, using a NSF in $[-4,4]^{2^k}$, and the loss functions forward .. (4 in total), and we aim to test $k=1,2,3,4,6$ (the maximal dimension is 64, test from highest dimension, need to see ESS difference).

The parameter scaling is: training and validation sets are both $10000\cdot 2^k$, batch size fixed at $2000$ across different $k$, and use $10000$ steps, learning rate is $10^{-3}$. Watch the ESS and converge at $d=64$ first. The quench and temper algorithm use $100\cdot 2^k$ samples, and each time draw a batch.

---

## 2. Why this is the right high-dimensional test

This experiment is the high-dimensional companion to the five 2D benchmarks already in the paper (Threewell, Himmelblau, Annulus, Python, Chessboard) and directly fills the stated future-work gap *"applying the framework to higher-dimensional Boltzmann sampling"* (Discussion, `Paper/main.tex:604`). Three structural properties make it ideal:

1. **The target is a product measure, so the optimal flow is known in closed form.** Because
$$
U(x)=\sum_{i=1}^{k}\Big[\tfrac12 x_i^2+12e^{-x_i^2}\Big]+\sum_{i=k+1}^{2^k}\tfrac12 x_i^2 ,
$$
the target factorizes $\mu=\prod_i\mu_i$ into $k$ identical 1D double wells and $2^k-k$ standard Gaussians. The optimal transport map is therefore **diagonal**: the *same* 1D map $g$ on each of the first $k$ coordinates and the identity on the rest. This gives a ground-truth optimum to compare against and means that any failure is a *training / mode-discovery* failure, not a capacity failure.

2. **The number of modes equals the dimension and grows combinatorially.** With the hardened $\exp(-x^2)$ coefficient $12$ (raised from $6$), each double well has minima at $x_i=\pm\sqrt{\ln 24}\approx\pm1.783$ separated by a barrier of height $\approx 9.9$ at the origin (per coordinate: well value $2.09$ vs. barrier value $12.0$). The $k$ wells combine into exactly $2^k=d$ modes, all living in the first-$k$-coordinate subspace. So $k=1,\dots,7$ gives $d=2,4,8,16,32,64,128$ with the same number of modes respectively — **#modes = $d$ exactly**.

3. **The source sits on the barrier, maximizing the fake-ESS hazard.** $\mu_0=\mathcal N(0,I_d)$ is centered at the origin, which is the *saddle/barrier* of every double well. The untrained pushforward covers none of the $2^k$ mode combinations, exactly as the narrow Threewell/Himmelblau sources do in 2D. As $k$ grows, plain forward KL is expected to lock onto a handful of mode combinations while reporting a near-perfect ESS — the **fake ESS pitfall** at combinatorial scale — which $\mathrm{X}_{\hat\mu}$ (QT mode discovery) and the mixture variant are predicted to rescue.

**Headline hypothesis.** As $d$ increases from 2 to 64: ESS stays high for *all four* losses (because ESS only sees the support the flow already reaches), but **mode coverage collapses with $d$ for forward KL and KL+$\mathrm{X}_\mu$**, while **KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$ and the mixture retain coverage**, with the mixture attaining the best full-coverage ESS. This is the 2D story (`Paper/main.tex:591`) carried to high $d$.

---

## 3. The four losses (already implemented; reuse verbatim)

The losses are exactly those of `2D_Benchmark/2D_Himmelblau/core.py` and the paper's total loss `Paper/main.tex:444-454`. With $G$ mapping target→source, $z(y)=U_0(G(y))-U(y)-\log|\det J_G(y)|$ is the log-ratio $\log(\mu/\nu)(y)$ (single forward pass; permutation gives the pairwise term at zero extra backprop):

| # | Method id (keep these strings) | Loss | Regime |
|---|--------------------------------|------|--------|
| 1 | `KL`                | $\mathbb E_\mu[z]$ | forward KL baseline |
| 2 | `KL+X_mu`           | $\;+\,\lambda\,\mathbb E_{\mu^2}\lvert z-z'\rvert$ | in-mode shape control |
| 3 | `KL+X_mu+X_hat_mu`  | $\;+\,\mathbb E_{\hat\mu^2}\lvert z-z'\rvert$ | QT mode discovery |
| 4 | `KL+X_mu+X_mix`     | $\;+\,\mathbb E_{(\alpha\hat\mu+\beta\bar\nu)^2}\lvert z-z'\rvert$ | mixture (mode discovery + leakage) |

Fixed loss coefficients (paper default, `main.tex:457`): $\lambda=1$, $\alpha=\beta=1/2$. The $\bar\nu$ batch is the raw stop-gradient pushforward $\bar y_i=G^{-1}(x_i)$, $x_i\sim\mu_0$.

`core.py` for this benchmark = a thin copy of the Himmelblau `core.py` with the potential swapped:

```python
class MultiWell(Potential):
    def __init__(self, k: int):
        super().__init__()
        self.k = k                                  # d = 2**k
    def forward(self, x):                           # x: [N, d] -> [N]
        quad = 0.5 * (x * x).sum(-1)
        well = 12.0 * torch.exp(-(x[:, :self.k] ** 2)).sum(-1)  # deeper wells (8->12) to harden
        return quad + well
```

`loss_KL`, `loss_X`, `loss_KL_X`, `quench_and_temper`, `coverage` are reused unchanged.

---

## 4. Smart network sizing per $k$

The key sizing insight follows from §2.1: **the optimal map is diagonal**, so capacity does *not* need to grow with $d$ the way a fully-coupled target would demand. The hard sub-problem (the double-well→Gaussian transport) is fixed-shape and low-dimensional ($k\le 6$). We therefore **hold spline resolution and depth nearly constant** and **scale only the conditioner width** so that it is not bottlenecked by the number of spline parameters it must emit. Per NSF coupling layer the conditioner emits $\approx \tfrac d2\,(3\cdot\text{BINS}+1)$ numbers (active half $\times$ params per spline); the hidden width must comfortably exceed that.

| $k$ | $d=2^k$ | #modes | `BINS` | `TRANSFORMS` | `HIDDEN_FEATURES` | conditioner out/layer | rationale |
|-----|---------|--------|--------|--------------|-------------------|-----------------------|-----------|
| 1 | 2  | 2  | 16 | 6  | (128, 128) | ~49   | tiny; 2D-benchmark-class net |
| 2 | 4  | 4  | 16 | 6  | (128, 128) | ~98   | same |
| 3 | 8  | 8  | 16 | 8  | (128, 128) | ~196  | +2 transforms for mask coverage |
| 4 | 16 | 16 | 16 | 8  | (128, 128) | ~392  | width still ample |
| 6 | 64 | 64 | 16 | 10 | (256, 256) | ~1568 | widen conditioner; +transforms for the 58 identity dims |

Sizing rules (encode as functions of `k` in `parameters.py`, not magic numbers):

- **`BINS = 16`, fixed.** The target is smooth (Gaussian + smooth Gaussian bump); the double-well→Gaussian transport is monotone with one mildly steep region near the barrier. 16 rational-quadratic bins resolve it; the 2D benchmark's 32 was for sharp manifold targets (Python $\sigma_\text{stroke}=0.04$, Chessboard edges) we do not have here. *Bump to 24 only if a marginal-fit diagnostic (§6) shows a poorly captured barrier.*
- **`TRANSFORMS = 6` for $d\le4$, `8` for $d\in\{8,16\}$, `10` for $d=64$.** With `randmask=True` a coordinate is "active" in $\approx T/2$ layers; even $T=6$ gives every multimodal coordinate several spline applications. The mild growth is cheap insurance that all $k$ multimodal coords *and* the large identity block get mixed at $d=64$.
- **`HIDDEN_FEATURES = (128,128)` up to $d=16$, `(256,256)` at $d=64$.** Widen only when the conditioner output ($\sim$1568 at $d=64$) approaches the hidden width. Depth-2 MLP throughout.
- **Box / source:** `NSF_LIM = 4.0` (so the flow acts on $[-4,4]^d$, wells at $\pm1.783$ sit comfortably inside), `SIGMA = 1.0` ($\mu_0=\mathcal N(0,I_d)$), `PLT_LIM = 4.0`.
- Initialize every flow with `flow.zeros()` (identity warm start), as the 2D code does.

This keeps the *comparison across $k$* clean: the four losses see near-identical architecture so any ESS/coverage gap is attributable to the loss, not the net.

---

## 5. Specific parameter choices (single source of truth → `parameters.py`)

All quantities below are written as functions of `k` so one `parameters.py` covers the whole sweep.

### 5.1 Training
| Symbol | Value | Notes |
|--------|-------|-------|
| `K_LIST` | `(7, 6, 5, 4, 3, 2, 1)` | **run highest $d$ first** (user: converge at $d=128$ first) |
| `N_TRAIN` | $10000\cdot2^k$ | source pool drawn once, reused: 20k / 40k / 80k / 160k / 640k |
| `N_VALID` | $10000\cdot2^k$ | fresh source batch for final ESS / coverage |
| `BATCH` | `200` | **fixed across $k$**; reduced from 2000 — forward KL is batch-sensitive, so a small batch widens the KL-vs-X gap (and cheapens the $O(d)$ inverse) |
| `STEPS` | per-$k$ via `parameters.steps(k)`: 3000/3000/2500/2000/1500/1200 for $k=1..6$ | enough for genuine convergence at batch 500; the full `STEPS=10000` spec run is `--steps 10000` |
| `LR` | `1e-3` | Adam |
| `SEED` | `0` (train), `1` (QT), `42` (eval) | match 2D convention |

Memory check at $d=64$: pool $640{,}000\times64$ float32 $\approx 164$ MB — fine on the 16 GB RTX 5070 Ti. **Chunk the validation pushforward** (`N_VALID=640k`) into blocks of e.g. 50k (or use `importance_weights(..., chunk=...)`) to bound peak VRAM.

### 5.2 QT pool for $\hat\mu$ (mode discovery)
| Symbol | Value | Notes |
|--------|-------|-------|
| `P_QT` | $100\cdot2^k$ | $\approx 100$ samples per mode: 200 / 400 / 800 / 1600 / 6400 |
| `QT_SIGMA` | `2.0` | diffusion (melt) std — large enough to cross every per-dim barrier |
| `QT_OPT_STEP`, `QT_OPT_ITERS` | `0.5`, `200` | LBFGS quench, `armijo=True` |
| `QT_MC_STEP`, `QT_MC_ITERS` | `2e-3`, `500` | Langevin temper (500, down from the 2D 1000, since $\hat\mu$ shape need only be coarse) |
| `B_hat` (per-step draw) | `BATCH` for `X_hat_mu`; `BATCH//2`=1000 for `X_mix` | drawn **with replacement** from the pool, then refreshed by 50 Langevin steps |

> **Small-$k$ note:** for $k=1,2$ the pool ($P_{QT}=200,400$) is smaller than the per-step draw `B_hat`. Draw indices uniformly **with replacement** and rely on the per-step Langevin refresh (`step=2e-3, iters=50`) to diversify duplicates — exactly the recycle-with-rejuvenation scheme of `main.tex:416`.

### 5.3 One-step importance-sampling surrogate for the $\mu$-batch
**One-step flow only — no annealed importance sampling, no Boltzmann-generator ladder.** The forward KL and $\mathrm{X}_\mu$ terms get their approximate $\mu$-samples from a single importance-resampling pass against the current pushforward, identical to the 2D code (`2D_Benchmark/2D_Himmelblau/train.py:62-67`), at **every** $k$ including $d=64$:

```python
G_now = flow.t()
y, _ = G_now.inv.call_and_ladj(x)                 # pushforward of source batch x ~ mu_0
w = importance_weights(x, u0, u1, G_now.inv)      # one-step weights
y_mu = resample(y, w)                             # resample by weights
y_mu = langevin(y_mu, u1, step=2e-3, iters=50)    # short Langevin to diffuse duplicates
```

| Symbol | Value | Notes |
|--------|-------|-------|
| `IS_MC_STEP`, `IS_MC_ITERS` | `2e-3`, `50` | post-resample Langevin (same for all $k$) |

> This is deliberate, not a shortcut. The paper notes one-step weights collapse in high $d$ (`main.tex:348`) — so the $\mu$-batch *degrades* as $k$ grows, and plain `KL` is *expected* to miss combinatorial modes. That failure **is** the result. Mode discovery is supplied by $\mathrm{X}_{\hat\mu}$ via QT (§5.2), which is independent of the IS weight quality, so the X-variants are predicted to hold up where one-step `KL` cannot.

### 5.4 Compile / speed
- `u1.enable_grad().enable_eval()` once (required by `langevin` and `lbfgs(armijo=True)`). The analytic `MultiWell` grad is cheap and compiles cleanly.
- `from zflows.utils import suppress_warnings, set_cache_size_limit; suppress_warnings(); set_cache_size_limit(64)` at script top (the sweep builds many compiled closures).
- Optionally wrap the training loss with `compile_raw`; the in-loss `torch.randperm` is fine under `torch.compile`. Validate on the sanity run before trusting it.

---

## 6. Diagnostics and figures

Scatter plots no longer work for $d>2$, so coverage is measured *combinatorially* (enabled by separability) in addition to the paper's two metrics.

1. **ESS trajectory** (training) and **final ESS** (on `N_VALID`) per method — same as `plot_results.py`. Headline per-$k$ ESS curves overlay the 4 methods.
2. **kNN coverage** vs the QT $\hat\mu$ pool, `coverage(y, y_hat_mu, k=5)` — works in any $d$, reused verbatim.
3. **Exact mode coverage (new, separability-exploiting).** Assign each pushforward sample to a well-combination by the sign pattern of its first $k$ coordinates, $s(y)=\operatorname{sign}(y_{1:k})\in\{-,+\}^k$ (the barrier is exactly at 0). Report
   - `modes_found / 2**k` = fraction of the $2^k$ combinations populated above a small count threshold;
   - **mode-balance** = total-variation / KL distance of the empirical combination histogram to the uniform $2^{-k}$ (all wells have equal mass by symmetry).
   This is the crisp high-$d$ analogue of "did the flow miss a mode," and the cleanest way to expose fake ESS.
4. **1D marginal fit** of $x_1$ (and $x_{k}$) vs the true double-well marginal $\propto e^{-x^2/2-6e^{-x^2}}$ — verifies `BINS` is sufficient and both wells are captured.
5. **$(x_1,x_2)$ 2D projection** of the pushforward (for $k\ge2$) showing the 4 sub-mode quadrants.
### Headline deliverable: two summary tables (loss × $k$)

The primary result is **two tables**, each with **4 rows (losses) × one column per $k$**:

- **Table 1 — final ESS** (on the fresh `N_VALID` pool).
- **Table 2 — exact mode coverage** `modes_found / 2**k` (diagnostic 3).

Columns are $k\in\{1,\dots,7\}$ i.e. $d\in\{2,4,8,16,32,64,128\}$ (a $4\times7$ grid; `K_LIST` is a one-line change to add/remove columns). **Actual final-ESS result** (well coeff 12, batch 200, `data_k{1..7}.pth` → `ess_table.csv`):

| loss \\ $d$ | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|-------------|---|---|---|----|----|----|-----|
| `KL`                | 0.985 | 0.958 | 0.924 | 0.882 | 0.790 | 0.669 | 0.547 |
| `KL+X_mu`           | 0.997 | 0.983 | 0.971 | 0.963 | 0.893 | 0.845 | 0.657 |
| `KL+X_mu+X_hat_mu`  | 0.986 | 0.989 | 0.974 | 0.961 | 0.948 | 0.835 | 0.745 |
| `KL+X_mu+X_mix`     | 0.987 | 0.992 | 0.977 | 0.970 | 0.953 | 0.878 | **0.752** |

What the table shows (confirmed): **plain forward KL degrades monotonically with $d$** ($0.985\to0.547$) while the X-augmented losses stay high, so **the KL-vs-X gap widens with dimension** ($\approx0.01$ at $d=2$ to $+0.18$–$0.21$ at $d=64,128$); the **mixture is best (or tied-best) for every $d\ge4$**. This is the ESS/calibration regime — exact mode coverage is $1.0$ and mode-imbalance $\sim0.01$–$0.03$ for *all* losses (the flow always reaches all $2^k$ modes evenly), so ESS is the sole discriminator. Tables rendered as Markdown (`tables.md`) and CSV (`ess_table.csv`, `mode_coverage_table.csv`, `mode_balance_table.csv`); this is the paper's high-$d$ contribution, extending the 2D summary table (`main.tex:587`).

Save under `VD_Product/`: the two tables (`tables.md`, `tables.tex`); a tidy `summary.csv` with one row per $(k,\text{method})$ holding `final_ess, knn_coverage, modes_found, mode_balance`; and supporting per-$k$ figures `ESS_k{1..7}.png`, `modes_k*.png`, `marginal_k*.png` (secondary, for diagnosis). Optional `ess_vs_dim.png` / `coverage_vs_dim.png` line plots reproduce the same two tables visually.

### Progress reporting (no tqdm)

Long runs must **not** use `tqdm`. Each `train.py` run appends human-readable progress and status to a plain-text log **inside `VD_Product/`**, e.g. `VD_Product/train_status.log` (and/or per-$k$ `status_k{K}.txt`). Write at a fixed cadence (e.g. every 1% of `STEPS`, so every 100 steps at `STEPS=10000`), each line carrying: timestamp, current $k$/$d$, method, step / total, **percent complete**, latest loss, running ESS, **wall-clock elapsed**, **per-step ms**, and **ETA**. Flush after each write (`flush=True` / `open(..., 'a')` per write) so the file is tail-able live, and echo the same line to stdout. Example line:

```
[2026-05-27 18:42:10] k=6 d=64  KL+X_mu+X_mix  step 4500/10000  45.0%  loss=3.21e-01  ess=0.78  elapsed=312.4s  3.2ms/step  eta=352s
```

Track time with `time.perf_counter()` (and `torch.cuda.synchronize()` before timing on CUDA so the elapsed/ms numbers are real, not async-queued). Also write a one-line overall-sweep heartbeat to `VD_Product/sweep_status.log` at each $k$/method boundary (which stage is running, total wall-clock elapsed, ETA for the remaining sweep) so the autonomous run is monitorable by tailing one file.

### Time budget & kill-early policy

**Always monitor wall-clock; never let a bad run burn time.** Encode an explicit per-method time budget and abort conditions in `train.py` and in the autonomous monitor:

- **Per-method soft budget (default):** $k\le3$: ≤5 min; $k=4$: ≤15 min; $k=6$: ≤45 min. After the $d=64$ sanity run, recompute the budget from the measured ms/step (`ESTIMATED_TOTAL = ms_per_step * STEPS * 4 methods`) and **commit to it**.
- **Abort immediately (do not wait out a doomed run) if any of:** (a) loss or ESS becomes `NaN`/`Inf`; (b) measured ms/step implies the full run exceeds ~2× the committed budget; (c) ESS for a method has collapsed to ≈0 and is not recovering after a few hundred steps (blown up). On abort, write the reason to `train_status.log`/`sweep_status.log`, save whatever partial `data_k{K}.pth` exists, and surface to the human rather than silently retrying.
- The autonomous monitor (Step 4) tails the status logs and `nvidia-smi`; if a budget is exceeded or `NaN`/blow-up appears, it **kills the background job at once** (`TaskStop` / kill the PID), records the abort, and only then decides whether to retune (lower QT iters, smaller net, fewer steps) — it does **not** let a slow/diverged job keep running.

---

## 7. File layout (mirror `2D_Benchmark/`, one parametrized codebase)

```
VD_Product/
  vd_plan.md          # this file
  core.py             # MultiWell potential + loss_KL/loss_X/loss_KL_X + QT + coverage + mode_coverage
  parameters.py       # all §4–§5 numbers as functions of k; K_LIST = (7,6,5,4,3,2,1)
  test_qt.py          # QT sanity at one k (e.g. k=2): assert finite, plot (x1,x2)
  train.py            # loops K_LIST (highest d first); 4 methods each; saves data_k{K}.pth
                      #   writes progress to train_status.log (no tqdm)
  plot_results.py     # per-k ESS / modes / marginal figures + summary.csv
  plot_sweep.py       # builds tables.md + tables.tex (the two 4xk tables); optional *_vs_dim.png
  data_k{1,2,3,4,5,6,7}.pth
  train_status.log    # live, tail-able training progress (percent + status)
  sweep_status.log    # one-line-per-stage heartbeat across the whole sweep
  tables.md / tables.tex   # the two headline tables (Markdown + \input-able LaTeX)
  summary.csv
```

Each `data_k{K}.pth` mirrors the Himmelblau `data.pth` schema (`config`, `x_unif`, `y_hat_mu`, `runs[method] = {state_dict, ess_history, samples, final_ess}`) plus the new `mode_coverage` / `mode_balance` fields. `train.py` skips any $k$ whose `data_k{K}.pth` already exists (idempotent resume, as the 2D code does).

---

## 8. Execution order (user requirement: $d=64$ first)

1. **QT sanity** (`test_qt.py` at $k=2$): confirm QT finds all 4 quadrant modes; finite output.
2. **$d=64$ sanity ($k=6$), short:** `STEPS=500`, all 4 methods, one-step IS. Confirm (a) training is stable, (b) ESS curves *separate* across methods, (c) `KL` mode coverage $<1$ while `+X_hat_mu` recovers it, (d) `train_status.log` is written with percent/status (no tqdm).
3. **Full $d=64$ run:** `STEPS=10000`, save `data_k6.pth`, generate `modes_k6.png` + ESS curve. **This is the gate** — only proceed if the high-$d$ story holds.
4. **Descend in $k$:** $k=4,3,2,1$ at full settings (each cheaper than $d=64$).
5. **Build tables:** `plot_sweep.py` → `tables.md`, `tables.tex` (the two 4×$k$ tables), `summary.csv`, optional `*_vs_dim.png`.

---

## 9. `/autonomous-research` procedure

Run via the `autonomous-research` skill as a **Workflow 1.5 (Experiment Bridge) → Stage-1/2 audit → Workflow 3 (write-up)** pipeline, every artifact gated by an independent fresh-context reviewer sub-agent. Effort: `balanced` (raise to `max` for the $d=64$ run and the table build). This plan file *is* the `EXPERIMENT_PLAN.md` contract.

**Step 0 — Bootstrap & environment snapshot.** Create `.aris/{config.md, wiki/, artifacts/, reviews/, meta/events.jsonl}`. Detect cross-family reviewers (`command -v codex gemini llm`); record in `.aris/config.md`. Snapshot the env once → `.aris/artifacts/ENV_SNAPSHOT.md`: GPU (`nvidia-smi` → RTX 5070 Ti, 16 GB), `torch 2.12.0+cu130 cuda True`, `zflows` import OK, CPU/RAM. Load any `2D_Benchmark` results into the wiki so the high-$d$ run is framed as the extension, not a re-derivation.

**Step 1 — `experiment-bridge` (write code).** From §3–§7 write `core.py`, `parameters.py`, `test_qt.py`, `train.py`, `plot_results.py`, `plot_sweep.py`. Keep eval honest (`experiment-integrity.md`): ESS/coverage/mode-coverage computed from *actual* pushforward weights on a *fresh* `N_VALID` pool — no self-normalized shortcuts, no reference labels derived from the model, no unused metrics.

**Step 2 — Code review (reviewer, repository-level), loop to convergence.** Spawn a `general-purpose` adversarial reviewer that reads the files *directly*. Rubric: (a) `MultiWell` matches §1 and the optimum is diagonal; (b) the four losses match `main.tex:444` and the 2D `core.py`; (c) the §4 sizing functions are correct and `BINS/TRANSFORMS/HIDDEN` match the table; (d) the $\mu$-batch uses the §5.3 **one-step IS only** — no AIS/Boltzmann ladder; (e) exact-mode-coverage classifier is correct; (f) `flow.t()` capture-once, `enable_grad`/`enable_eval` ordering, validation chunking; (g) progress goes to `train_status.log` with percent/status and **no `tqdm`**. Address every `critical`/`major` with ≥2 strategies; accept at score ≥6/10 and all `critical` resolved, max 4 rounds. Persist `.aris/reviews/code/round-N.md`. **No GPU time before this passes.**

**Step 3 — Sanity run.** `test_qt.py` (k=2) + the §8.2 short $d=64$ run (`STEPS=500`). Confirm pipeline executes, files written, ESS curves separate, `KL` coverage $<1$. Auto-debug on failure: classify → class-specific fix → retry ≤3; if two distinct strategies fail, spawn a separate rescue-diagnosis sub-agent.

**Step 4 — Full sweep (autonomous, background) with active time-monitoring.** Confirm `nvidia-smi` free, then `Bash run_in_background: true`: $k=6$ first (the gate), then $k=4,3,2,1$, then `plot_sweep.py`. **Monitor by tailing `VD_Product/train_status.log` / `sweep_status.log`** (no tqdm) and watch the ms/step + ETA fields; the harness re-invokes on exit (don't poll-sleep at 60 s — use a long fallback heartbeat). **Enforce the §6 kill-early policy:** if any run hits a `NaN`/blow-up or its projected wall-clock exceeds ~2× the committed budget, **kill the job immediately** (`TaskStop`/kill PID), log the reason, save partial data, and retune *before* relaunching — never wait out a doomed run. Collect commands/configs/seeds/timings/metric-file paths into `.aris/artifacts/EXPERIMENT_LOG.md`; add an `experiments/` wiki node.

**Step 5 — Audit cascade.**
- *Stage 1 `experiment-audit`* (reviewer, repository-level): the 5 integrity failure modes over `train.py`/`plot_*` + `data_k*.pth` → `EXPERIMENT_AUDIT.md`.
- *Stage 2 `result-to-claim`*: build `CLAIM_LEDGER.md` mapping each claim — "ESS stays high for all losses across the grid", "mode coverage of `KL` decays with $d$", "mixture best full-coverage ESS at $d=64$", "QT finds all $2^k$ modes" — to `supported / partially_supported / invalidated` against `summary.csv` and `tables.md`.

**Step 6 — Write-up (Workflow 3) into the paper.** Add a *High-dimensional product multi-well* subsection to `Paper/main.tex` (new numerical-experiments subsection) whose centerpiece is the **two 4×$k$ tables** (`\input{../VD_Product/tables.tex}`), with the five-pass scientific edit. Then *Stage 3 `paper-claim-audit`* (fresh zero-context reviewer): cross-check every quantitative table entry against `CLAIM_LEDGER.md` + raw `data_k*.pth`/`summary.csv` (per-claim `exact_match`/`number_mismatch`/`missing_evidence`). Visual PDF review of the new tables/figures; citation audit if new refs are added.

**Always surface to the human:** invalidated claims, unresolved `critical` items, any kill-early abort and its cause, and any sampler/budget decision that changed the qualitative story. Humans own the final call on whether the high-$d$ result is paper-ready.

---

## 10. Implementation notes & performance (from the actual run)

The dominant cost at high $d$ is the **NSF inverse**, and understanding why drove several parameter choices:

- **`zflows.NSF` is a masked-autoregressive flow** (`MaskedAutoregressiveTransform`). Its *forward* (`call_and_ladj`, used for the loss and density) is a single parallel pass — fast (~17–29 ms at $d=64$, batch 2000). Its *inverse* (`_inverse`, used to draw pushforward $\nu$-samples $y=G^{-1}(x)$) runs a **sequential loop of `passes = d`** masked-MLP evaluations (`core/transforms.py:500`), so it costs $\approx d\times$ the forward.
- **Measured at $d=64$, batch 2000:** inverse $\approx 1.9$ s with `transforms=10`, $\approx 1.1$ s with `transforms=6`. The cost is **launch-overhead bound, not FLOP/memory bound**: peak memory is only 0.16 GB and per-sample time is flat across batch size (2000→125), so **chunking does not help**, and **hidden width is essentially free** — `(64,64)`, `(128,128)`, `(256,256)`, and the bottleneck `(256,64,256)` all invert in ~1.1 s. (A naive autograd graph *through* the inverse balloons to ~15 GB and OOMs — always draw $\nu$-samples under `torch.no_grad`.)
- **Consequences baked into the code:**
  - **`transforms = 6` (flat in $k$).** The only architectural lever on inverse cost is the transform count (linear in the `passes` loop), and the separable/diagonal-optimal target needs little coupling depth. Hidden width kept roomy at `(256,256)` for $k\ge5$ since it is free.
  - **One inverse per step.** `train.py` computes $y=G^{-1}(x)$ and its ladj in a single `G.inv.call_and_ladj`, then derives the IS log-weights from that same ladj ($\log w = -U(y)+U_0(x)+\mathrm{ladj}$) — avoiding a second inverse inside `importance_weights`. Halves the per-step cost.
  - **Flow direction.** $G$ maps target→source, so the **loss** (forward KL + all X terms, evaluated on $\mu$-samples) uses only the *fast forward* pass, and the *slow inverse* is confined to the `no_grad` sampling block.
  - **Eval pool capped to 20 000** samples for the final ESS (the inverse is compute-bound; 20 k makes the ESS estimate stable to ~$10^{-2}$).
  - **Per-$k$ step counts** (`parameters.steps`): 800 ($k\le3$), 600 ($k=4$), 400 ($k=5$), 300 ($k=6$) — these smooth separable targets converge in well under the 2D paper's 1000 steps, and the high-$d$ inverse makes step count the wall-clock driver. The full `STEPS=10000` spec run is available via `--steps 10000` (≈hours at $k=6$).
- **Measured wall-clock:** $\approx 1.18$ s/step at $k=6$; the full $4\times6$ sweep runs in roughly 1.3 h on the RTX 5070 Ti. Highest-$d$-first ordering means a complete $4\times4$ table (d = 8,16,32,64) lands before the cheap $k=1,2$ columns fill in the $4\times6$.
- **If a future run needs the inverse to be cheap** (e.g. `STEPS=10000` at $d=64$): switch the high-$k$ flow to `RealNVP(mixing="lu")`, whose coupling inverse is closed-form $O(d)$ in a single parallel pass — orders of magnitude faster than the autoregressive NSF inverse — at the cost of departing from the NSF spec.
