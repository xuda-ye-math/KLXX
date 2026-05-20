# 2D Himmelblau experiment

Reference implementation for Section 5.5 of the paper. Trains a normalizing flow on the four-mode Himmelblau potential under three losses (forward KL, +$\mathrm{X}_\mu$, +$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$) to expose when the wide-coverage regularizer $\mathrm{X}_{\hat\mu}$ is actually needed.

## Target and source

- **Target.** Himmelblau, $U_1(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2$. Four well-separated minima near
  $(3, 2)$, $(-2.81, 3.13)$, $(-3.78, -3.28)$, $(3.58, -1.85)$.
- **Source.** Isotropic Gaussian $\mathcal{N}(0, \sigma^2 I)$, with $\sigma = 1.0$. Almost all mass sits near the central saddle, far from any mode.

## Files

| file | role |
| --- | --- |
| `core.py` | shared building blocks: `Himmelblau` potential, `loss_KL` / `loss_X` / `loss_KL_X`, `quench_and_temper`, `coverage` |
| `train.py` | full training run with $\sigma = 1.0$; saves `data.pth` |
| `plot_results.py` | loads `data.pth`; writes `ESS.png`, `samples.png`, `resample.png` |
| `test_qt.py` | standalone sanity check for the Quench-and-Temper utility (4-mode coverage) |
| `parameters.py` | *legacy* — kept for historical reference; not used by the current scripts |

`data.pth` and `__pycache__/` are gitignored (`*.pth`, `__pycache__/` in the repo `.gitignore`).

## Methods

Each training run trains a single neural spline flow (NSF) by Adam minimization of one of three losses, all sharing the AIS-surrogate target samples produced by the current pushforward:

| key | loss | provenance |
| --- | --- | --- |
| `KL` | `loss_KL(y_mu, u0, u1, G)` | forward KL on AIS target samples |
| `KL+X_mu` | `loss_KL_X(y_mu, u0, u1, G, lambda_=1)` | adds $\mathrm{X}_\mu$ regularizer at $\lambda = 1$ |
| `KL+X_mu+X_hat_mu` | `loss_KL_X(y_mu, ...) + loss_X(y_hat_mu, u0, u1, G)` | also adds $\mathrm{X}_{\hat\mu}$ at $\hat\lambda = 1$ against QT-generated samples |

Inside `loss_KL_X` the $\mathrm{X}_\mu$ term reuses the same $z = U_0(G(y)) - U_1(y) - \log|J_G|$ scalars computed for the KL term — so $\mathrm{X}_\mu$ is essentially free given a forward KL pass. (Implementation note: a random permutation gives an unbiased estimator of $\mathbb{E}_{x,y \sim \mu}|z(x) - z(y)|$.)

## Training loop (per step)

```
sample x ~ u_0 (batch of 500 from a pool of 50,000)
y = G^{-1}(x)                          # current pushforward
w = importance_weights(x, u0, u1, G^{-1})
y_mu = resample(y, w)                  # one-step importance resample
y_mu = langevin(y_mu, u1, step=2e-3, iters=50)
if method has X_hat_mu:
    y_hat_mu = langevin(y_hat_mu, u1, step=2e-3, iters=50)
loss = ...                             # see methods table
loss.backward(); optimizer.step()
```

Hyperparameters: `BATCH=500`, `STEPS=1000`, `LR=1e-3`. NSF has $6$ coupling transforms, $32$ bins, hidden widths $(128, 128)$, domain $[-6, 6]^2$. `flow.zeros()` warm-starts the bijection to identity.

## Quench-and-Temper (QT) for $\hat\mu_1$

`quench_and_temper(x, target, sigma, opt_step, opt_iters, mc_step, mc_iters)` runs three stages:

1. **Diffuse (melt):** $x \leftarrow x + \sigma \cdot \xi$, $\xi \sim \mathcal{N}(0, I)$.
2. **Optimize (quench):** L-BFGS with Armijo line search drives each particle to the nearest mode of $U_1$.
3. **Rejuvenate (temper):** Langevin around each mode spreads the deterministic optima into a sample.

In `train.py`, QT is invoked **once** at the start of training to produce `y_hat_mu` (500 samples), with `sigma=2.0, opt_step=0.5, opt_iters=200, mc_step=2e-3, mc_iters=1000`. The samples are then **recycled** across training steps; one extra Langevin rejuvenation step is applied per iteration (gated to the third method only) so the set drifts under $U_1$ rather than staying frozen at the QT output.

Code-side gotchas you actually need to know:

- **`u1.enable_grad()`** is mandatory before calling `langevin`. **`u1.enable_eval()`** is mandatory before calling QT (because its `lbfgs(armijo=True)` step needs the compiled energy evaluation). Both calls live at the top of the train script.
- **`y_hat_mu` rebinding is local to `train()`**. The langevin rejuvenation inside the loop reassigns the local name, so each `train()` call starts from the original (caller-passed) QT output and accumulates drift only within that call. Different methods do not pollute each other.
- **The `target(y)` term in `loss_KL`** is $\nu$-independent and could be dropped without changing the gradient. We keep it for symmetry with `loss_X` (which subtracts a permuted version) and for cleaner printouts.

## Reported numbers

After training:

| forward KL | + $\mathrm{X}_\mu$ | + $\mathrm{X}_\mu$ + $\mathrm{X}_{\hat\mu}$ |
| --- | --- | --- |
| Cov = 0.53 | Cov = 0.53 | **Cov = 1.00** |

Coverage is computed against the QT-generated `y_hat_mu` at $k = 5$ using `core.coverage`. The pattern matches the paper's claim: the ESS alone is *fake* (it is high but two of four modes are missing), and only $\mathrm{X}_{\hat\mu}$ rescues coverage.

## How to run

```bash
cd 2D_Himmelblau
python train.py          # produces data.pth
python plot_results.py   # writes ESS.png, samples.png, resample.png; prints coverage
```

`train.py` skips training and prints a message if `data.pth` is already present, so `plot_results.py` is the only step that needs to be re-run for figure regeneration.

## The QT sanity test (`test_qt.py`)

A standalone, no-flow sanity check that QT alone discovers all four Himmelblau modes from a unimodal Gaussian start:

```python
source = Gaussian(mean=[0., 0.], variance=[1., 1.])
x = source.samples(2048)
x_out = quench_and_temper(x, target, sigma=2.0, opt_step=0.5, opt_iters=200, mc_step=2e-3, mc_iters=1000)
```

Assertions:
1. `torch.isfinite(x_out).all()` — no NaN/Inf escaped the optimizer.
2. `median(min distance to a known mode) < 0.5` — most particles land near a mode.
3. `argmin distance has 4 unique values` — every mode is hit by at least one particle.
4. `coverage(x_out, ref, k=5) > 0.8` against a reference of 512 points drawn from $\mathcal{N}(\mathrm{mode}_i, 0.2^2 I)$ — 5-NN balls of the reference cluster are reached by the QT cloud.

The fourth check is the most informative: it quantifies *how tightly* the QT cloud concentrates relative to a $\sigma_\text{ref} = 0.2$ Gaussian per mode. Coverage of $\approx 0.88$ in our runs reflects that the Langevin rejuvenation deliberately spreads samples *wider* than a $0.2$-Gaussian — by design — so a few reference points on the outer rim of each cluster fall just outside any candidate's 5-NN ball. Tightening the rejuvenation (smaller `mc_step`, fewer iters) would push the number toward $1$ without changing mode discovery.

Run:

```bash
python test_qt.py        # prints intermediate diagnostics, writes qt.png
```
