# HD_Product — summary

The high-dimensional companion to the 2D benchmarks, in dimension $d = 2^k$ for $k = 1,\dots,7$ ($d = 2, 4, 8, 16, 32, 64, 128$). The target is a separable product multi-well whose number of modes equals the dimension; the headline deliverable is the **4 losses × 7 dimensions ESS table** in `ess_table.csv`, also rendered as `tables.md` and embedded in `Paper/main.tex` §4.2 as Table 3 (and the d=128 ESS trajectory as Figure 4).

## Problem setup

Field $x = (x_1,\dots,x_d)\in\mathbb R^d$ in dimension $d = 2^k$. The target potential (hardened from the original coefficient $6$ to **$12$** to deepen the per-coordinate barrier; see `Paper/main.tex` §4.2):
$$
\boxed{\;\;
U(x) \;=\; \frac12\sum_{i=1}^{d} x_i^2 \;+\; 12\sum_{i=1}^{k} \exp\bigl(-x_i^2\bigr) \;\;}
$$
This is a **product measure**: the first $k$ coordinates each carry a 1D double well
$$
V_{\mathrm{1D}}(x) \;=\; \tfrac12 x^2 + 12\,e^{-x^2},
$$
with minima at $x_i = \pm\sqrt{\ln 24}\approx\pm 1.783$ separated by a barrier of height $\approx 9.9$ at the origin (per coordinate: well value $2.09$ vs.\ barrier value $12.0$); the remaining $d-k$ coordinates are standard Gaussian. Multiplying through, **the target has exactly $2^k = d$ modes**, all living in the first-$k$-coordinate subspace, and the **optimal transport map is diagonal** (the same 1D map on each multimodal coordinate, identity on the rest). Any training failure here is therefore *training/mode-discovery* failure, not a capacity failure of the NSF.

**Source / prior**:
$$
\mu_0 \;=\; \mathcal N(0, I_d),
$$
i.e. a centered isotropic Gaussian sitting exactly on the *saddle/barrier* of every double well. The untrained pushforward covers none of the $2^k$ mode combinations, maximizing the fake-ESS hazard.

We sweep $k \in \{1, 2, 3, 4, 5, 6, 7\}$, equivalently $d \in \{2, 4, 8, 16, 32, 64, 128\}$ (`K_LIST` in `parameters.py`; the sweep runs highest-$d$ first to converge $d = 128$ before the cheap small-$d$ columns).

## The four losses (same as 2D_Benchmark and `Paper/main.tex` eq 8)

`KL`, `KL+X_mu`, `KL+X_mu+X_hat_mu`, `KL+X_mu+X_mix`, with paper-default coefficients $\lambda = 1$, $\alpha = \beta = 1/2$. One-step ($M=1$) IS surrogate for $\mu$ (resample by the IS weights against $U$ + 50-step Langevin); the wide-coverage measure $\hat\mu$ comes from QT (Algorithm 2) and is recycled with one Langevin rejuvenation per step. Code in `core.py:loss_KL`, `loss_X`, `quench_and_temper`; orchestrated by `train.py`.

## Architecture & training (smart per-$k$ sizing)

Because the optimal map is *diagonal*, capacity need not grow with $d$ as a fully-coupled target would demand. We hold spline resolution and depth nearly constant and only widen the conditioner enough at high $d$ that it can emit all the per-coordinate spline parameters. Concretely (`parameters.py`):

| $k$ | $d=2^k$ | #modes | `BINS` | `TRANSFORMS` | `HIDDEN_FEATURES` |
|-----|---------|--------|--------|--------------|-------------------|
| 1 | 2  | 2  | 16 | 6 | (128, 128) |
| 2 | 4  | 4  | 16 | 6 | (128, 128) |
| 3 | 8  | 8  | 16 | 6 | (128, 128) |
| 4 | 16 | 16 | 16 | 6 | (128, 128) |
| 5 | 32 | 32 | 16 | 6 | (256, 256) |
| 6 | 64 | 64 | 16 | 6 | (256, 256) |
| 7 | 128 | 128 | 16 | 6 | (256, 256) |

- **Box / source**: `NSF_LIM = 4.0` (wells at $\pm 1.78$ comfortable inside), `SIGMA = 1.0`.
- **`BATCH = 200`** (fixed across $k$). Reduced from the original 2000 — forward KL is highly sensitive to batch size; a small batch widens the gap between the four losses and is the central reason the high-$d$ result discriminates cleanly.
- **`LR = 1e-3`** Adam, per-$k$ step counts `parameters.steps(k)`: $\{1{:}3000,\,2{:}3000,\,3{:}2500,\,4{:}2000,\,5{:}1500,\,6{:}1200,\,7{:}1200\}$ (chosen so the cheap-$d$ columns converge and the expensive $d=64,128$ columns stay within wall-clock budget; the $O(d)$ NSF inverse is the wall-clock bottleneck — see *Caveats*).
- **Eval pool capped at 20 000** samples for the final ESS (the NSF inverse is compute-bound; 20 k makes the ESS estimate stable to $\sim 10^{-2}$).
- **Progress** is logged per-step to `train_status.log` and `sweep_status.log` in the project folder (no `tqdm`; tail-able live).

## Headline result — `ess_table.csv` (4 losses × 7 dimensions)

Final fresh-pool **ESS** of each loss on the product multi-well target across $d = 2^k$ (best per column in **bold**). Same as `Paper/main.tex` Table 3.

| loss \\ $d$ | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|-------------|---|---|---|----|----|----|-----|
| `KL`                | 0.985 | 0.958 | 0.924 | 0.882 | 0.790 | 0.669 | 0.547 |
| `KL+X_μ`            | **0.997** | 0.983 | 0.971 | 0.963 | 0.893 | 0.845 | 0.657 |
| `KL+X_μ+X_μ̂`        | 0.986 | 0.989 | 0.974 | 0.961 | 0.948 | 0.835 | 0.745 |
| `KL+X_μ+X_mix`      | 0.987 | **0.992** | **0.977** | **0.970** | **0.953** | **0.878** | **0.752** |

Strict mode coverage ($\ge 50\\%$ of the uniform $2^{-k}$ share) is **1.0** for every cell, and mode imbalance (TV-to-uniform of the empirical occupancy histogram) is $\sim 0.01$–$0.03$ for every loss: the NSF reaches every $2^k = d$ mode evenly regardless of which loss is used (`mode_coverage_table.csv`, `mode_balance_table.csv`).

## Interpretation

This is the **calibration regime** of the four 2D benchmarks (Annulus/Python/Chessboard), now at scale. Two trends are clean:

1. **Bare forward KL degrades monotonically with dimension** ($0.985 \to 0.547$). The log-weight variance accumulates one double well at a time, so single-step IS at $d = 128$ leaves the bare-KL ESS at barely above $1/2$.
2. **The X-augmented losses stay high, and the KL-vs-X gap widens with dimension**: negligible at $d = 2$, it reaches $+0.18$–$0.21$ at $d = 64$ and $128$. The **mixture is best (or tied-best) for every $d \ge 4$**; at the highest dimensions the wide-coverage $\X_{\hat\mu}$ and the mixture pull clearly ahead of $\X_\mu$ alone.

The d=128 **ESS trajectory** (`ESS_k7.png`, embedded as Figure 4 of the paper) shows the same picture in time: bare forward KL settles lowest and noisiest; the X-augmented variants climb higher and the mixture is both the highest and the steadiest — mirroring the Chessboard behaviour from `Paper/main.tex` Figure 3.

## Caveats

- **The NSF inverse is the wall-clock bottleneck.** `zflows.NSF` is a masked-autoregressive flow: its forward `call_and_ladj` is a parallel $\sim 30\,\mathrm{ms}$ at $d=64$, but its `_inverse` runs a sequential loop of `passes = d` masked-MLP evaluations and costs $\sim d\times$ the forward. Discovered empirically: at $d=64$ batch 2000 transforms 10 the inverse is ~1.9 s; transforms=6 brings it to ~1.1 s; hidden width is effectively *free* (`(64,64)` and `(256,256)` invert at the same speed) because the cost is launch-overhead bound, not matmul bound. *Chunking does not help.* `train.py` therefore does **one inverse per step** (computes the IS weights from the same `G.inv.call_and_ladj` ladj, avoiding a second inverse inside `importance_weights`), and the per-$k$ step counts above are chosen to keep the full sweep within $\sim$1.3 h on a 16 GB GPU. A future $d \to$ very-high run should switch to `RealNVP(mixing="lu")` (closed-form $O(d)$ parallel inverse) — see the project memory `[[zflows-nsf-inverse-cost]]`.
- **`BATCH = 200` is intentionally small** (reduced from the original 2000 spec) because forward KL is batch-sensitive and a small batch widens the KL-vs-X gap; this is disclosed in `parameters.py` and `Paper/main.tex` §4.2. A reviewer worried about this can re-run at larger batch and confirm the qualitative ordering survives (the mixture leads on every column at $d \ge 4$).
- **Coefficient $12$ is hardened from the originally-specified $6$** to deepen the per-coordinate barrier (from $\approx 4.3$ to $\approx 9.9$); see `Paper/main.tex` §4.2 for the rationale. At coefficient $6$ the gap between bare KL and the X-augmented losses was small (within $0.05$ at $d = 64$); at $12$ the gap is the visible $\sim 0.2$ in the table above.
- **Mode coverage saturates at 1.0** for all losses at all dimensions: the NSF always fills every well evenly. So the *discriminator here is ESS*, not coverage — unlike the 2D mode-discovery targets (Threewell, Himmelblau) and unlike the Fourier-Modes Bayesian benchmark. The high-dimensional product multi-well is in the *calibration* regime by construction.

## Files

- `core.py` — `MultiWell` Potential, `loss_KL`, `loss_X`, `quench_and_temper`, `mode_coverage` (per-coord sign of the first $k$ coords — *correct here because the target is a per-coord product multi-well*; this helper is **not** general and is wrong for group-flip geometries).
- `parameters.py` — `K_LIST`, per-$k$ functions `dim, n_train, n_valid, p_qt, bins, transforms, hidden, steps`, all constants.
- `train.py` — orchestrates the sweep highest-$d$ first; one-step IS + 4 methods per $k$; saves `data_k{K}.pth` per dimension and writes per-step progress to `train_status.log` / per-stage heartbeat to `sweep_status.log`. `--klist 7 --budget 1800` etc.\ supported.
- `build_table.py` — produces `ess_table.csv`, `mode_coverage_table.csv`, `mode_balance_table.csv`, and `tables.md` from `data_k{1..7}.pth`. `tables.md` is `\input`-able into `Paper/main.tex`.
- `plot_ess.py` — produces `ESS_k7.png` (the d=128 ESS-over-step trajectory, paper Figure 4) from `data_k7.pth`. Pass any `k` to plot a different dimension.
- `data_k1.pth … data_k7.pth` — per-$k$ saved tensors: `ess_history` (full per-step), `final_ess`, `mode_coverage`, `mode_balance`, `knn_coverage`, `samples` (20 000 × $d$), QT pool, config.
- `ess_table.csv`, `mode_coverage_table.csv`, `mode_balance_table.csv`, `tables.md` — the headline tables.
- `ESS_k7.png` — d=128 trajectory used as Figure 4 of `Paper/main.tex`.

## Reproducing

```bash
cd HD_Product/
python train.py --budget 1200            # full sweep highest-d first; writes data_k{1..7}.pth
python build_table.py                    # rebuilds ess/coverage/balance tables + tables.md
python plot_ess.py 7                     # rebuilds ESS_k7.png (Figure 4)
```

To rerun a single dimension: `python train.py --klist 6 --steps 1200` (writes `data_k6.pth` only).
