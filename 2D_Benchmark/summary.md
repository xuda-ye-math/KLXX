# 2D_Benchmark — summary

Five two-dimensional targets that each isolate a different failure mode of the bare forward KL and let us read off, in pictures, where the cross-regularization functionals $\X_\mu$, $\X_{\hat\mu}$, $\X_{\bar\nu}$ and the mixture variant earn their keep. This folder is the empirical core of `Paper/main.tex` §4.1; see Figures 1–3 and Table 2 there for the rendered headline.

## The four losses (kept identical across all five targets)

For a flow map $G:\mathbb R^d\to\mathbb R^d$ trained so that $\nu := G^{-1}_{\\#}\mu_0 \approx \mu$ with source $\mu_0$ and target $\mu\propto\exp(-U)$, write the log-ratio along the flow as
$$
z(y) \;=\; U_0(G(y)) - U(y) - \log|\det J_G(y)|,
$$
so the **four methods compared are** (paper eq. 8):

| id | loss | role |
|----|------|------|
| `KL`                | $\mathbb E_{y\sim\mu}[z(y)]$ | bare forward KL baseline |
| `KL+X_mu`           | $\;+\,\lambda\,\mathbb E_{\mu^2}|z(y)-z(y')|$ | in-mode shape control |
| `KL+X_mu+X_hat_mu`  | $\;+\,\mathbb E_{\hat\mu^2}|z(y)-z(y')|$ | QT-driven mode discovery |
| `KL+X_mu+X_mix`     | $\;+\,\mathbb E_{(\alpha\hat\mu+\beta\bar\nu)^2}|z(y)-z(y')|$ | mixture (mode discovery + leakage suppression) |

Paper-default coefficients $\lambda = 1$, $\alpha = \beta = 1/2$ throughout. $\bar\nu$ is the detached stop-gradient pushforward of the current model; $\hat\mu$ is built once before training by **Quench and Temper** (Algorithm 2): diffuse a cloud of source samples, run LBFGS on $U$ to collapse onto mode centers, rejuvenate with a short Langevin chain.

## The five targets and their potentials

(Same as `Paper/main.tex` Table 1.)

| target | potential $U(x_1, x_2)$ | minima | source $\mu_0 = \mathcal N(0,\sigma^2 I)$ |
|--------|------------------------|--------|-------------------------------------------|
| **Threewell**  | $6\bigl[(x_1^2-1)^2 + (x_2^2-1)^2 + \sin(x_1+2x_2)\bigr]$ | three asymmetric wells | $\sigma=0.25$ (narrow, far from all wells) |
| **Himmelblau** | $(x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2$ | four isolated points | $\sigma = 1.0$ |
| **Annulus**    | $20\bigl(r^6 - 8r^4 + 16r^2 + 1\bigr)^{1/3}$, $r^2=x_1^2+x_2^2$ | $r=0$ + ring at $r=2$ | $\sigma = 0.8$ |
| **Python**     | 1024 stroke Gaussians of width $\sigma_{\mathrm{stroke}}=0.04$ along the Python-logo skeleton + 2 eye Gaussians of width $\sigma_{\mathrm{eye}}=0.30$ | 1026 mode centers on a 1D manifold | $\sigma = 2.0$ (larger box) |
| **Chessboard** | Gaussian mixture on 26 of the 32 black squares of an 8×8 chessboard (6 cells removed) | 26 centers | $\sigma = 1.0$ |

These were chosen so that **each target isolates a different forward-KL failure mode**: Threewell and Himmelblau test mode discovery (the source covers none of the modes); Annulus tests in-manifold shape calibration (a thin 1D ring of minima at $r=2$ plus an isolated central minimum); Python tests calibration on a strongly anisotropic 1D manifold (the snake-body skeleton); Chessboard tests off-support leakage (the 6 removed cells must stay empty).

## Architecture & training (same recipe, light per-target hyperparameters)

The flow is a **neural spline flow** with 6 coupling transforms, 32 bins, hidden widths $(128, 128)$, on a square box matched to each target. Training uses Adam (learning rate $10^{-3}$ to $5\times 10^{-3}$) for **1000 steps** (2000 on Python, whose narrow tubular support takes longer to saturate), batch size 500, enlarged to **1000–2000** on the multi-mode Threewell and Chessboard so a single batch can cover all modes at once. At each step the AIS surrogate for $\mu$ is **one importance-weighted resampling pass** against the current pushforward followed by **50 Langevin steps** — the one-step ($M=1$) regime of `Paper/main.tex` §3.2. The wide-coverage measure $\hat\mu$ is built once before training by QT (Algorithm 2) and recycled with one Langevin rejuvenation per step.

## Headline result (`Paper/main.tex` Table 2)

Final **ESS / $k=5$ Naeem coverage against $\hat\mu$** for each loss on each target. Best full-coverage ESS per row in **bold**.

| target | `KL` | `KL+X_μ` | `KL+X_μ+X_μ̂` | `KL+X_μ+X_mix` |
|---|---|---|---|---|
| Threewell  | $0.99 / 0.75$ | $1.00 / 0.75$ | $0.98 / \mathbf{1.00}$ | $\mathbf{0.99}/\mathbf{1.00}$ |
| Himmelblau | $0.92 / 0.53$ | $0.97 / 0.48$ | $\mathbf{0.95}/\mathbf{1.00}$ | $\mathbf{0.95}/\mathbf{1.00}$ |
| Annulus    | $0.66 / 1.00$ | $0.97 / 1.00$ | $0.94 / 1.00$ | $\mathbf{0.97} / 1.00$ |
| Python     | $0.62 / 1.00$ | $0.75 / 1.00$ | $0.73 / 1.00$ | $\mathbf{0.84} / 1.00$ |
| Chessboard | $0.77 / 1.00$ | $0.76 / 1.00$ | $0.78 / 1.00$ | $\mathbf{0.84} / 1.00$ |

## Interpretation

Two regimes emerge.

**Mode-discovery regime (Threewell, Himmelblau).** Bare `KL` and `KL+X_μ` post ESS up to 1.00 while silently missing whole modes — coverage 0.75 on Threewell, 0.53 and 0.48 on Himmelblau. This is the **fake-ESS pitfall** in its starkest form: ESS read in isolation would license a "successful training" verdict, yet a quarter to half of the target's mass is absent from the trained $\nu$. Only the QT-driven `X_μ̂` and mixture variants restore full coverage; on Himmelblau the two full-coverage losses tie at the best ESS, with the mixture pushforward the visibly cleaner of the two.

**Calibration / leakage regime (Annulus, Python, Chessboard).** Coverage is already $1.00$ for every loss, so ESS becomes the discriminator. `X_μ` raises ESS sharply on Annulus ($+0.31$) and Python ($+0.13$) but is flat on Chessboard ($-0.01$). The mixture improves on bare forward KL by $+0.31$, $+0.22$, $+0.07$ respectively and attains the best ESS on all three. On Chessboard the $\bar\nu$ half of the mixture is the key: it lets the off-support penalty $|z(y) - z(y')|$ pick up the leakage into the 6 removed cells that $\hat\mu$ alone cannot see.

Across the five targets the **mixture variant is the only loss that simultaneously matches or beats the best full-coverage ESS and reaches full mode coverage on every target**, never trading one off against the other.

## Caveats

- The per-target hyperparameters (NSF box, learning rate, batch, source $\sigma$) were tuned independently per target — these are not held fixed across the five. The comparison is *across losses on each target*, not *across targets*.
- The one-step AIS surrogate ($M=1$) is the contract for these benchmarks; an AIS ladder might shift the ESS numbers up for every method, but the qualitative ordering is what the X functionals are designed to produce, not the absolute ESS.
- On Threewell, the narrow source ($\sigma=0.25$) at the origin is the worst case for forward KL's mode-discovery blindness; on a wider source the fake-ESS pitfall would be milder. This is deliberate — Threewell exists in this benchmark to make the failure visible.
- The `X_{\bar\nu}` contribution is only realized inside the mixture; we do not run a stand-alone `KL+X_μ+X_{\bar\nu}` to keep the comparison to four methods. The mixture is strictly stronger than the sum $\alpha^2 \X_{\hat\mu}+\beta^2 \X_{\bar\nu}$ because of the cross terms with $x\sim\hat\mu, y\sim\bar\nu$ (see `Paper/main.tex` line 113).

## Per-target sub-folders

Each `2D_<target>/` folder is self-contained:

- `core.py` — the target potential, `loss_KL`, `loss_X`, `quench_and_temper`, `coverage`, and target-specific helpers.
- `parameters.py` — pinned hyperparameters (source $\sigma$, NSF box, batch, steps, learning rate).
- `train.py` — the one-step IS training loop for the 4 methods; saves `data.pth`.
- `plot_results.py` — produces `samples.png` (Row 1 of paper figure: pushforward $y = G^{-1}(x)$), `resample.png` (Row 2: $\mathrm{resample}(y, w)$ + Langevin rejuvenation), and `ESS.png` (the Chessboard-style ESS trajectory).
- `test_qt.py` — sanity-checks the Quench and Temper output for the target.
- `data.pth` — per-method `ess_history`, `final_ess`, `samples`, `state_dict`; consumed by `plot_results.py`.

Paper figure pages (`Paper/main.tex` Figures 1, 2, 3) reference these `samples.png` / `resample.png` / `ESS.png` directly via `\includegraphics{../2D_Benchmark/2D_<target>/<png>}`.

## Reproducing

```bash
cd 2D_Benchmark/2D_<target>/
python train.py            # full run; writes data.pth
python plot_results.py     # rebuilds samples.png, resample.png, ESS.png
python test_qt.py          # optional: QT sanity check
```
