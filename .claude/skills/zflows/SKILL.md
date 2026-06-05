---
name: zflows
description: Write correct code that uses the `zflows` package — self-contained PyTorch normalizing flows (inspired by zuko, but no longer depends on it) with energy-based / SMC sampling utilities. Invoke when the user imports from `zflows.flow`, `zflows.potential`, `zflows.loss`, or `zflows.utils`, or asks you to build a flow, train with reverse/forward KL, run Langevin/HMC/L-BFGS, or do importance sampling / resampling against an unnormalized target.
---

# zflows

`zflows` is a self-contained PyTorch library for normalizing flows on $\mathbb R^d$ (and the torus), oriented at **energy-based sampling**. It is strongly inspired by [`zuko`](https://github.com/probabilists/zuko) and reuses its design vocabulary (lazy transforms, `ComposedTransform`, `call_and_ladj`), but since v0.3.2 it no longer has `zuko` as a runtime dependency — the core machinery is vendored under `zflows.core` (do **not** import from there directly; always go through the four public modules below).

Installation (PyPI release v0.5.x onward):

```bash
pip install zflows                  # pinned tagged release
# or, for the bleeding edge + demo scripts:
git clone https://github.com/xuda-ye-math/zflows.git && cd zflows && pip install -e .
```

Four modules, all importable directly:

```python
from zflows.flow      import Flow, NSF, NCSF, CNF, RealNVP, OTFlow, ComposedTransform
from zflows.potential import (
    Potential, Uniform, Gaussian, Gaussian_Mixture,
    Linear_Combination, linear_combination, potential_from,
)
from zflows.loss      import reverse_KL, forward_KL, OT_loss, loss_compile, loss_compile_beta
from zflows.utils     import (
    importance_weights, importance_weights_F, importance_weights_G,
    importance_weights_log, importance_weights_log_F, importance_weights_log_G,
    compute_ESS, compute_ESS_log, compute_CESS, compute_CESS_log,
    resample, langevin, rejuvenation, stochastic_heun, hmc, lbfgs, optimization,
    annealed_importance_sampling, annealed_importance_sampling_F, annealed_importance_sampling_G,
    check_compile_available, set_cache_size_limit, suppress_warnings,
)
```

All flows are **unconditioned** (`context=0`). The motivating use case is one fixed target distribution.

---

## 1. The `Flow` contract

Every flow class subclasses `Flow(nn.Module, ABC)` and exposes exactly one access path to its bijection:

```python
F = flow.t()                        # ComposedTransform
y, ladj = F.call_and_ladj(x)        # forward and log|det J_F(x)|
y       = F(x)                      # forward only
x_back  = F.inv(y)                  # inverse
x_back, ladj_inv = F.inv.call_and_ladj(y)
```

Rules — break these and you write wrong code:

- **Always use `flow.t()`** to get the bijection. `flow.t()` is the only supported entry point; `Flow` does not implement `forward()` (it is abstract), so calling `flow(...)` or reaching for a zuko-style `flow().transform` will not work.
- `F.call_and_ladj(x)` is the canonical forward — it returns *both* `y` and `log|det J_F(x)|` in one ODE / one pass.
- `flow.zeros()` initialises the flow to the identity bijection (`T(x) == x`, `ladj == 0`) by zeroing the last conditioner layer (plus LU/rotation mixing in RealNVP). Useful as a warm start.
- **`flow.t()` is capture-once safe.** The returned `ComposedTransform` holds *references* to `flow`'s `nn.Parameter`s and re-reads them by attribute access on every forward pass, so `F = flow.t()` captured ONCE survives subsequent `optimizer.step()` updates without rebuilding. This is the central contract `loss_compile` / `loss_compile_beta` rely on. Inside training loops you may either re-call `flow.t()` each step (slightly cheaper Python overhead) or capture once and reuse (`loss_compile` requires it).
- **Optional compiled fused fast paths** (on `ComposedTransform`, mirror `Potential.enable_grad`): capture `F = flow.t()`, then `F.enable_for_ladj()` / `F.enable_inv_ladj()` `torch.compile` the transform, so `F.for_ladj(x)` returns the compiled `F.call_and_ladj(x)` → `(y, ladj)` and `F.inv_ladj(y)` the compiled `F.inv.call_and_ladj(y)` → `(x_pre, ladj_inv)` (both return the fused points **and** log|det J|; `inv_ladj`'s ladj is the inverse map's, `= -log|det J_F|` at the pre-image). These live on `ComposedTransform` (`zflows/core/transforms.py`), **not** on `Flow`. Calling `for_ladj`/`inv_ladj` before the matching `enable_` raises `RuntimeError`. **Re-enable after changing the flow?** — **No** for in-place param updates (`optimizer.step()`, `load_state_dict()`, `zeros()`): the captured `F` re-reads the same tensors, so the compiled path reflects them automatically. **Yes** if the params became *different* tensors (`.to(device)`/`.to(dtype)`, swapping a submodule, rebuilding the flow): the old compile is stale — fetch a fresh `flow.t()` and enable on it. `enable_*` is deliberately **not idempotent** (each call recompiles; re-calling on the same `F` also refreshes). Use when a forward/inverse map is the repeated-fixed-shape bottleneck (e.g. the `G⁻¹` source-pushforward in `utils.annealed_importance_sampling_G`, which auto-uses these when the passed `flow.t()` has them enabled). Both directions speed up ~3–20× and the gain grows with `d` — `tests/compare_compiled_inverse.py` (GPU). The `NSF`/`NCSF` bisection inverse emits a `torch.compile` graph-break warning but remains correct and still gets the large speedup; closed-form/ODE maps (`RealNVP`, `CNF`, `OTFlow`) compile cleanly.

### Picking a flow class

| Class     | Domain         | Inverse cost      | Use when                                                                            |
|-----------|----------------|-------------------|-------------------------------------------------------------------------------------|
| `NSF`     | $[a, b]^d$ box | bisection         | Default for bounded / well-localised targets.                                       |
| `NCSF`    | $[a, b]^d$, periodic per axis (default $[-\pi, \pi]^d$) | bisection | Angular / toroidal targets.                                                         |
| `CNF`     | $\mathbb R^d$  | RK4 ODE (fixed-step) | Topologically complex targets (e.g. interlocking rings) where splines struggle. Hutchinson divergence (optional).         |
| `OTFlow`  | $\mathbb R^d$  | RK4 ODE (fixed-step) | Continuous flow with **closed-form** Hessian trace ($O(d \cdot m)$); typically faster + more stable than `CNF`. Optional OT regularizers via `OT_loss`. |
| `RealNVP` | $\mathbb R^d$  | closed-form O(d)  | Want fast closed-form inverse (latent interpolation, repeated `F.inv`).             |

### Constructors (exact signatures)

```python
NSF(a, b, bins=8, slope=1e-3, transforms=4, randmask=True, hidden_features=(64, 64), activation=nn.SiLU)
NCSF(a, b, bins=8, slope=1e-3, transforms=4, randmask=True, hidden_features=(64, 64), activation=nn.SiLU)
CNF(dimension, frequency=3, nt=16, exact=True, hidden_features=(64, 64), activation=nn.SiLU)
OTFlow(dimension, hidden=64, layer=3, rank=10, nt=8, time_bound=(0.0, 1.0))
RealNVP(dimension, transforms=4, randmask=True, mixing=None,
        hidden_features=(64, 64), activation=nn.SiLU)
```

- `a`, `b` are 1-D length-`d` lists/tensors (lower/upper box bounds). `assert a.shape == b.shape and a.ndim == 1`.
- `activation` is a **class**, not an instance — pass `nn.SiLU`, not `nn.SiLU()`.
- `hidden_features` is a tuple of MLP widths.
- `randmask=True` (the default since v0.5) draws a fresh `torch.randperm(d)` per layer. Pass `randmask=False` to recover the legacy `arange / arange.flip` alternating mask.
- For `CNF`, `exact=False` switches to the Hutchinson trace estimator (faster for large d, biased gradients). Integration is fixed-step RK4 (no adaptive solver) with `nt=16` RK4 steps by default; bump `nt` for tighter inverse round-trip / log-det accuracy at linear cost.
- For `OTFlow`, the velocity field is `−∇Φ_θ` for a learnable scalar potential `Φ_θ` (antiderivative-of-tanh ResNet + low-rank quadratic head). `nt=8` RK4 steps is a typical training default; bump for tighter inverse round-trip.
- For `RealNVP`, `mixing` interleaves a learnable linear "1×1 conv" between coupling layers: `None` (default; pure couplings), `"rotation"` (orthogonal `R = exp(A − A^T)`, log|det|≡0), or `"lu"` (PLU map; `L`'s diagonal contributes a non-trivial log|det|). Recommended `"lu"` at $d \gtrsim 8$; identity-initialised so it's a no-op until training moves it.

---

## 2. The `Potential` contract

`Potential(nn.Module)` represents $U(x) = -\log\mu(x)$ up to an additive constant.

```python
class MyU(Potential):
    def __init__(self):
        super().__init__()        # MANDATORY
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d]  ->  return [N]
        return ...
```

**Shape contract is strict:** input `[N, d]`, output `[N]` — not `[N, 1]`, not scalar. The fast `enable_grad` path uses `vmap(grad(lambda x: forward(x.unsqueeze(0)).squeeze(0)))` and depends on this.

### Two opt-in compiled fast paths

```python
u = MyU().to(device).enable_grad().enable_eval()   # chainable, idempotent
g = u.grad(x)    # [N, d] -> [N, d];  vmap(grad(forward)) under torch.compile
v = u.eval(x)    # [N, d] -> [N];     compiled forward
u.eval()         # NO ARG -> standard nn.Module.eval() switch (returns self)
```

- `enable_grad()` builds `_grad_fn = torch.compile(vmap(grad(single)), mode='reduce-overhead')`.
- `enable_eval()` builds `_eval_fn = torch.compile(forward, mode='reduce-overhead')`.
- Both are **idempotent** — second call returns `self` without recompiling.
- Calling `.grad(x)` before `.enable_grad()` raises `RuntimeError`. Same for `.eval(x)` / `.enable_eval()`.
- **`u.eval()` (no arg) is the nn.Module switch; `u.eval(x)` is the fast path.** Do not confuse them.

#### `reduce-overhead` static-buffer hazard

The default `mode="reduce-overhead"` captures a CUDA graph; **the output tensor is a static buffer that the next compiled call overwrites**. If you need a value across calls (e.g. compare `g_old` and `g_new`), `.clone()` it:

```python
g = u.grad(x).clone()        # safe across the next .grad() / .eval() call
U_start = u.eval(x).clone()  # safe across the leapfrog
```

`utils.langevin`, `utils.hmc`, `utils.lbfgs` already handle this internally — but follow the same rule in your own code.

#### Other `mode=` options

`enable_grad(mode=...)` / `enable_eval(mode=...)` accept the same modes as `torch.compile`: `"reduce-overhead"` (default, fastest, fixed shape), `"default"` (no CUDA graph, lower VRAM, ~10–30% slower), `"max-autotune"` (longest compile, extra kernel tuning).

### Built-in potentials

All take `device=` and the obvious shape arguments. Only `Gaussian` exposes the tempered draw `.samples(N, beta=1.0) -> [N, d]` (the `beta` knob temperatures the sampling distribution to $\mathcal N(\mu, \Sigma / \beta)$); `Uniform` and `Gaussian_Mixture` expose plain `.samples(N)` (no `beta`):

```python
Uniform(a, b, device="cpu")                     # constant U(x) = 0 on a box
  .samples(N)                                   # uniform on [a, b]

Gaussian(mean, variance, device="cpu")          # diagonal Gaussian; variance positive
  .samples(N, beta=1.0)                         # draws from N(mean, variance / beta)

Gaussian_Mixture(weights, mean, variance, device="cpu")  # K diagonal modes
  # weights: [K]   mean: [K, d]   variance: [K, d]  (variance positive)
  .samples(N)

# v0.5 N-potential signature (previously: Linear_Combination(U0, U1, c0, c1=None)).
# Prefer the lowercase factory `linear_combination` in user code (returns an
# instance); reach for the `Linear_Combination` class only for isinstance
# checks / subclassing.
linear_combination([u0, u1, ...], [c0, c1, ...])     # U(x) = sum_k c_k * U_k(x)
  # coeffs default to uniform 1/N when omitted. Stored as a plain list[float]
  # regardless of input (list, tuple, 1-d Tensor, or None): a Tensor input is
  # detached + .tolist()'d at construction — NOT registered as a buffer, and a
  # requires_grad=True tensor is silently detached (not rejected). The coeffs
  # are immune to .to(device) (float * Tensor lifts to the child's device).
  # Mutate one in place (lc.coeffs[k] = ...) or retune all at once via
  # lc.set_coeffs([...]); the grad/eval closures read the new coeffs with no recompile.
```

Potential lifetimes are handled by Python refcounting + GC; there is no manual `.release()`.

### Defining a `Potential` — minimum correct skeleton

```python
class MyU(Potential):
    def __init__(self, param: float = 1.0):
        super().__init__()           # don't forget
        self.param = param           # plain attribute is fine
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x ** 2).sum(dim=-1)   # [N, d] -> [N]
```

Register learnable tensors via `self.register_buffer(...)` / `nn.Parameter`. `.to(device)` will recurse.

### One-liner: wrap a callable as a Potential

For stateless `(x) -> Tensor` callables that don't need a full subclass, use the module-level factory `potential_from` in `zflows.potential`. It writes the `Potential` subclass boilerplate for you and returns a **ready-to-use instance** (lowercase name = instance, per the project convention) — no manual instantiation:

```python
from zflows.potential import potential_from

def my_U(x):
    return 0.5 * (x ** 2).sum(-1) + 2 * torch.cos(x[:, 0])

u = potential_from(my_U).to(device).enable_grad()   # instance, chainable
```

`potential_from(my_U)` already IS the instance — do **not** call it again (`potential_from(my_U)()` would invoke `forward` with no argument).

For stateful potentials (physical constants, learnable sub-modules) subclass `Potential` directly.

(Note: in `zflows < 0.5.5` this was the classmethod `Potential._from`; in `< 0.5.6` there were briefly two module-level factories — `potential_from` returning a *subclass* and `potential_instance_from` returning an *instance*. v0.5.6 consolidated them into the single `potential_from`, which now returns an **instance** — `potential_instance_from` no longer exists.)

---

## 3. Losses

```python
from zflows.loss import reverse_KL, forward_KL

loss = reverse_KL(x, target=u1, F=flow.t())            # beta=1.0 default
loss = reverse_KL(x, target=u1, F=flow.t(), beta=0.5)  # tempered: minimise KL against exp(-beta*U_1)
loss = forward_KL(y, source=u0, F=flow.t())
```

All four KL losses take `beta: float = 1.0` (inverse temperature scaling the potential). Both return a scalar `torch.Tensor` ready for `.backward()`. They estimate

- `reverse_KL`: $\mathbb E_{x \sim \mu_0}[\beta \cdot U_1(F(x)) - \log|\det J_F(x)|]$, drops the parameter-independent $U_0$ term — use when you only have an unnormalized $U_1$.
- `forward_KL`: $\mathbb E_{y \sim \mu_1}[\beta \cdot U_0(F^{-1}(y)) + \log|\det J_{F^{-1}}(y)|]$ — use when you have samples from $\mu_1$.

(Also exported: `reverse_KL_F` ≡ `reverse_KL`, `forward_KL_F` ≡ `forward_KL`, plus the `G = F^{-1}` variants `reverse_KL_G`, `forward_KL_G` for when your "flow" maps target → source.)

### Canonical training loop

```python
flow = NSF(a=[-4, -4], b=[4, 4], bins=8, transforms=4, hidden_features=(64, 64)).to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

x = u0.samples(N)
for epoch in range(EPOCH):
    perm = torch.randperm(N, device=device)
    for start in range(0, N, BATCH):
        x_batch = x[perm[start:start + BATCH]]
        loss = reverse_KL(x_batch, target=u1, F=flow.t())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

### `OT_loss` — reverse KL plus OT regularizers (for `OTFlow` only)

`OT_loss(x, target, otflow, beta=1.0, alpha_C=1.0, alpha_R=1.0)` is the full
OT-Flow training objective: reverse KL **plus** the two optimal-transport
regularizers, all integrated in a single pass via the augmented 4-channel
ODE (`position, log-det, transport cost, HJB residual`).

```python
from zflows.loss import OT_loss
loss = OT_loss(x, target=u1, otflow=flow,              # flow is the OTFlow instance,
               beta=1.0, alpha_C=1.0, alpha_R=1.0)     # NOT flow.t() — see below
```

- The `otflow` argument is the **`OTFlow` instance itself**, not its
  `.t()` transform: `OT_loss` needs the underlying `OTFlowTransform` to
  reach the augmented `call_full(x)` path. Reusing `reverse_KL(x,
  target, flow.t())` would only integrate the first two channels.
- `alpha_C = alpha_R = 0` recovers `reverse_KL(x, target, flow.t())` exactly.
- Only usable with `flow: OTFlow`; the four-channel ODE is OTFlow-specific.

### `loss_compile` / `loss_compile_beta` — torch.compile'd training step

For heavy-load training (annealed Boltzmann generators with thousands of steps per bridge), wrap the loss once with one of two helpers in `zflows.loss`. Both capture `(potential, transform)` as closure constants and fuse the forward into a single CUDA graph — typically **4–10× faster per training step** on NSF/NCSF/RealNVP at small $d$.

```python
from zflows.loss import reverse_KL, loss_compile, loss_compile_beta

F = flow.t()                                # capture once; lazy machinery sees post-step weights

# (A) Fixed beta = 1.0. Single-input fast path, no wrapper overhead.
loss_fn = loss_compile(reverse_KL, target, F)         # mode='default' is the default
for x_batch in batches:
    loss = loss_fn(x_batch)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# (B) Adaptive / annealed beta. beta becomes a runtime arg of the closure;
#     cast to a 0-d tensor inside the wrapper so Dynamo treats it as a
#     dynamic input — one compiled artifact handles every value of beta.
loss_fn = loss_compile_beta(reverse_KL, target, F)
for x_batch, beta in batches:
    loss = loss_fn(x_batch, beta)           # beta: float | 0-d Tensor (NOT shape [1]/[N])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

- **Default `mode='default'`** (safe; no CUDA Graph). Pass `mode='reduce-overhead'` to capture a CUDA Graph — fastest at small $d$, but needs static batch shape and stable parameter memory (e.g. don't call `flow.zeros()` mid-training). `mode='max-autotune'` adds extra kernel tuning at the cost of a longer first compile.
- `loss_compile_beta` asserts `beta.dim() == 0`; pass either a Python float or a pre-allocated 0-d Tensor. Shape `[1]` / `[N]` would break the single-graph cache and is rejected.
- Both helpers are variadic — `loss_compile(loss_fn, *captured)` works for any callable `(x, *captured) -> scalar`, not just the four built-in KL losses. For a fixed non-default `beta` baked into the graph, pass it as a captured positional: `loss_compile(reverse_KL, potential, transform, 0.7)`.
- Pair with `zflows.utils.suppress_warnings()` to silence Triton / Inductor / Dynamo log noise during the first 1-2 compile passes.
- (Note: in `zflows < 0.5.5` these were named `compile_raw` / `compile_beta`. They have been renamed to `loss_compile` / `loss_compile_beta` and the default mode changed from `'reduce-overhead'` to `'default'`.)

---

## 4. SMC / sampling utilities

All sampler utilities accept a `chunk: int = 1` argument that splits the input along dim 0 and runs sequentially to bound peak VRAM. Statistically equivalent to `chunk=1` (each sample's update only depends on itself / its own noise).

### Importance sampling

```python
log_w = importance_weights_log(samples, source=u0, target=u1, F=flow.t(), chunk=1)
# log_w_i = -u1(F(x_i)) + u0(x_i) + log|det J_F(x_i)|

# Tempered IS: weight between mu_0^{beta_source} and mu_1^{beta_target} for
# SMC ladders. Default (1.0, 1.0) recovers the standard case.
log_w = importance_weights_log(samples, source=u0, target=u1, F=flow.t(),
                               beta_source=1.0, beta_target=0.4)
# log_w_i = -beta_target * u1(F(x_i)) + beta_source * u0(x_i) + log|det J_F|

w     = importance_weights(samples, source=u0, target=u1, F=flow.t(), chunk=1)
# == exp(log_w - log_w.max()), in [0, 1]; ready for resample()

# F/G variants (same convention as reverse_KL_F / _G): `importance_weights_log`
# and `importance_weights` ALIAS the forward-map _F versions. The _G twins take
# the INVERSE map G = F^{-1} (target -> source, e.g. flow.t().inv) instead:
log_w = importance_weights_log_G(samples, source=u0, target=u1, G=flow.t().inv)
w     = importance_weights_G(samples, source=u0, target=u1, G=flow.t().inv)
# If the passed transform has compiled fused maps enabled (F.enable_for_ladj()
# for _F, G.enable_inv_ladj() for _G), the routines use them automatically.
```

### Diagnostics

```python
ess  = compute_ESS(weights)             # weights >= 0, not required normalized
ess  = compute_ESS_log(log_weights)     # numerically stable variant
cess = compute_CESS(source_weights, importance_weights)
cess = compute_CESS_log(source_weights, log_importance_weights)
# all return a scalar Tensor in [0, 1]
```

Prefer `_log` variants on very-low-overlap proposals.

### Resampling

```python
resampled = resample(samples, weights)   # multinomial, with replacement; returns same [N, d]
```

### Rejuvenation (Langevin / MALA / tamed ULA)

```python
from zflows.utils import langevin, rejuvenation   # rejuvenation is an alias for langevin

# samples ~ exp(-beta * U) (tempered target)
x = langevin(samples, potential=u, beta=1.0, step=1e-3, iters=100,
             adjust=False, taming=0, chunk=1)
```

- `beta: float = 1.0` scales the drift to `beta * grad U` so the stationary distribution is `exp(-beta * U)`. Default recovers the un-tempered scheme.
- `adjust=False` (default): unadjusted Langevin (ULA), O(step) bias, **1 grad call per iter**.
- `adjust=True`: MALA, exactly samples `exp(-beta * U)`, **2 grad calls per iter**. If `potential.enable_eval()` was called, the accept/reject `U(x)`/`U(y)` go through the compiled fast path.
- `taming > 0`: replaces `beta * grad U(x)` with `beta * grad U / (1 + taming * ||beta * grad U||)`. Stabilises ULA on super-linearly growing potentials. **Incompatible with `adjust=True`** — raises `ValueError`.
- Requires `potential.enable_grad()`. Missing → `RuntimeError`.

### Stochastic Heun — unadjusted Langevin, lower-bias integrator

```python
from zflows.utils import stochastic_heun

x = stochastic_heun(samples, potential=u, beta=1.0, step=1e-3, iters=100, chunk=1)
```

Stratonovich predictor–corrector integrator for the overdamped SDE
$\mathrm d\theta = -\beta \nabla U(\theta)\,\mathrm dt + \sqrt 2\,\mathrm dB$.
One step reuses a single Wiener increment $\mathrm dW$:

  - predictor:  `x̃ = x − step · β · ∇U(x) + dW`
  - corrector:  `x' = x − 0.5 · step · (β · ∇U(x) + β · ∇U(x̃)) + dW`

- **Two gradient calls per iter** (vs. one for `langevin(adjust=False)`), but the trapezoidal drift cancels the leading $O(\text{step})$ Euler-Maruyama bias — so as `step → 0` the residual bias shrinks faster than ULA.
- Still **unadjusted** (no Metropolis correction) — a small step-dependent bias persists; use `langevin(adjust=True)` or `hmc` if you need exactly unbiased sampling.
- Requires `potential.enable_grad()`. `potential.enable_eval()` is **not** used (no energy evals).
- Picks no MH gate, so no NaN guard either — keep `step` small enough that the predictor stays finite on stiff targets.

### Hamiltonian Monte Carlo

```python
x = hmc(samples, potential=u, beta=1.0, step=1e-2, iters=10, burns=10, chunk=1)
```

- One `burn` = one momentum refresh + `iters` leapfrog steps + one MH accept/reject decision.
- `beta: float = 1.0` scales the potential in the Hamiltonian (kicks use `beta * grad U`); stationary distribution is `exp(-beta * U)`.
- Cost per burn: `iters + 1` compiled grad calls (combined half-kicks).
- Tune `step` so MH acceptance is ~0.6–0.8 (Beskos et al. 2013 sweet spot).
- Divergent trajectories (NaN/Inf energies) are clamped to `log_alpha = -inf` and rejected — the returned tensor is always finite.
- Requires `potential.enable_grad()`. `potential.enable_eval()` is optional; when present, MH energy evals use the fast path.
- "Safer than MALA on stiff targets": unbiased + NaN guard.

### Annealed importance sampling — `_F` / `_G` (flow-proposal SMC: `F_# source` → target)

```python
from zflows.utils import (annealed_importance_sampling,          # alias of _F
                          annealed_importance_sampling_F, annealed_importance_sampling_G)

# `samples` are drawn from source; the routine returns samples in TARGET space.
# Only `target` needs enable_grad() (Langevin rejuvenates in the target);
# `source` is forward-only. Calling target.enable_eval() (and source.enable_eval())
# routes the per-rung energy reweighting through the compiled .eval fast path.
# `annealed_importance_sampling` is an alias for the forward-map _F variant.
# _F: pass the FORWARD flow F (source -> target); needs .inv and .call_and_ladj.
y = annealed_importance_sampling_F(samples, source=u0, target=u1, F=flow.t(),
                                   beta_source=1.0, beta_target=1.0,
                                   ladder=20, step=5e-3, iters=30, chunk=1)
# _G: pass the INVERSE flow G = F^{-1} (target -> source); identical otherwise
#     (a flow trained target->source, or literally flow.t().inv of a forward flow).
y = annealed_importance_sampling_G(samples, source=u0, target=u1, G=flow.t().inv,
                                   ladder=20, step=5e-3, iters=30, chunk=1)
```

Uses the trained flow `F` as the proposal and anneals along the geometric path between the pushforward proposal and the target,

  `pi_k(y) = mu_1(y)**(k/M) * (F_# mu_0)(y)**(1 - k/M)`   (so `pi_0 = F_# source`, `pi_M = target`),

where `mu_0 ~ exp(-beta_source·source)`, `mu_1 ~ exp(-beta_target·target)`. Algorithm:
0. `y ← F(samples)` — push source particles to `pi_0 = F_# mu_0`.
1. for `k = 1 … M`: refresh the latent pre-images `x ← F⁻¹(y)`; incremental self-normalised weights `log w = (1/M)·(-beta_target·target(y) + beta_source·source(x) + log|det J_F(x)|)` — exactly `1/M ×` the `importance_weights_log` rule; multinomial `resample`; then `langevin` (ULA) rejuvenation **in `mu_1`** (`langevin(y, target, beta=beta_target, …)`).

- **No `linear_combination` / bridge potential is built** — weights come straight from the raw `source`/`target` energies and the flow Jacobian (via `F.inv`).
- Rejuvenation targets `mu_1` directly, not the exact `pi_k`: evaluating/differentiating `log pi_k` would need the pushforward density of `F` (so `F⁻¹` and its Jacobian gradient), far more expensive; the incremental weights already follow the `pi_k` path, so the `mu_1` kernel adds no essential deviation.
- **`_F` vs `_G`** (same convention as `reverse_KL_F` / `reverse_KL_G`): `_F` takes the forward map `F` (source → target) and uses `F(x)` / `F.inv(y)`; `_G` takes the inverse map `G = F⁻¹` (target → source) and uses `G.inv(x)` / `G(y)`, reading the inverse Jacobian (`log|det J_G(y)| = −log|det J_F(x)|`, hence a flipped sign). They are mathematically identical — `_G(…, G=F.inv)` reproduces `_F(…, F)`'s weights to round-trip tolerance.
- Only `target.enable_grad()` is required (`RuntimeError` otherwise); `source` needs none (forward-only), `enable_eval()` is not used.
- Returns particles in **target space** (`~ exp(-beta_target·target)`). `ladder=1` is a single reweight + resample + Langevin hop from the flow proposal to the target; raise `ladder` when the proposal `F_# source` overlaps the target poorly (low ESS).
- Distinct from the §5 annealed Boltzmann generator, which trains a flow *per rung*; here `F` is a single fixed trained flow and the annealing is in the importance weights. See `tests/4D_Boltzmann_generator.py` for the per-rung-training variant.

### L-BFGS (mode finding / MAP)

```python
from zflows.utils import lbfgs, optimization   # optimization is `lbfgs`

x = lbfgs(samples, potential=u, step=1.0, iters=100, memory=6,
          armijo=False, chunk=1)
```

- Batched: every particle carries its own (s, y) curvature history, all step in lockstep.
- `armijo=False` (default): fixed-step Newton update under BFGS Hessian approximation. Fast, can overshoot on first iter / non-convex.
- `armijo=True`: per-particle masked backtracking line search (K_MAX=6 trials, C1=1e-4, SHRINK=0.5). Guarantees `U(x_{k+1}) <= U(x_k)`. **Requires `potential.enable_eval()`** in addition to `.enable_grad()`.
- `memory` = Nocedal's m (typically 3–20). Larger = better Hessian approx, `N * d * 2 * memory` extra floats.
- `chunk > 1` is **bit-identical** to `chunk == 1` (no noise, per-particle history independent).

### Environment helpers (one-shot, top-of-script)

```python
from zflows.utils import suppress_warnings, set_cache_size_limit, check_compile_available

suppress_warnings()             # silences Triton autotune stderr, Inductor worker chatter,
                                # Dynamo recompile logs, and Python UserWarnings in one call

set_cache_size_limit(64)        # bump Dynamo's per-code-object cache size from the default 8.
                                # Use when a script builds many compiled closures sharing one
                                # code body (sweeps, annealed bridges, hyperparameter grids).

ok = check_compile_available()  # one-shot diagnostic: prints PASS/WARN/FAIL for
                                # (1) OS == Linux, (2) nvcc reachable on $PATH /
                                # /usr/local/cuda/bin / /usr/local/cuda-*/bin, and
                                # (3) the authoritative sanity test (actually
                                # torch.compile()'s a small probe and runs it).
                                # Returns True iff (3) succeeded. Run once, NEVER
                                # from inside a training loop (consumes a cache slot).
```

---

## 5. Canonical propose → reweight → resample → rejuvenate pipeline

```python
import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # cleaner compile logs

import torch
from zflows.flow import NSF
from zflows.potential import Potential, Gaussian
from zflows.loss import reverse_KL
from zflows.utils import (
    importance_weights_log, compute_ESS_log, resample, rejuvenation,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

u0 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)

class U1(Potential):
    def forward(self, x):
        return 0.5 * (x ** 2).sum(dim=-1) + 2 * torch.cos(x[:, 0])
u1 = U1().to(device)

flow = NSF(a=[-4, -4], b=[4, 4], bins=8, transforms=4, hidden_features=(64, 64)).to(device)

# (a) train by reverse KL
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
x = u0.samples(10000)
for epoch in range(10):
    for start in range(0, 10000, 1000):
        x_b = x[start:start + 1000]
        loss = reverse_KL(x_b, target=u1, F=flow.t())
        opt.zero_grad(); loss.backward(); opt.step()

# (b) importance sampling and diagnostics
with torch.no_grad():
    x_plot = u0.samples(10000)
    y_plot, _ = flow.t().call_and_ladj(x_plot)
    log_w = importance_weights_log(x_plot, source=u0, target=u1, F=flow.t())
    ess = compute_ESS_log(log_w)
    w = (log_w - log_w.max()).exp()
    y_resampled = resample(y_plot, w)

# (c) rejuvenate via MALA
u1.enable_grad().enable_eval()
y_fresh = rejuvenation(y_resampled, potential=u1, adjust=True, chunk=4)
```

For **annealed** Boltzmann-generator pipelines, build each bridge with `linear_combination([u_target, u_source], [c_k, 1.0 - c_k])` (v0.5 N-potential signature) and warm-start the same `flow` across all rungs `c_0=0 → c_M=1`. (To instead anneal *in the importance weights* using one already-trained flow as the proposal, reach for `annealed_importance_sampling_F` / `_G` from §4.) See `tests/4D_Boltzmann_generator.py`.

---

## 6. Common pitfalls / required behaviours

- **Always call `super().__init__()`** in `Potential` subclasses; otherwise nn.Module machinery (parameters, `.to(device)`) silently breaks. The shortcut for stateless potentials is `potential_from(fn)`, which returns a ready-to-use **instance** (no subclass written by hand, no manual instantiation).
- **`forward` returns `[N]`**, not `[N, 1]` or scalar.
- **`flow.t()` is the only supported entry**; `flow().transform` is the zuko-native path and is not part of the zflows contract.
- **`F = flow.t()` is capture-once safe** — re-reads `flow.parameters()` on every forward, survives `optimizer.step()`. `loss_compile` / `loss_compile_beta` require this. Re-calling `flow.t()` per iteration is also fine (cheap; just builds a fresh `ComposedTransform` shell that delegates to the same lazy machinery).
- **`.enable_grad()` is required before `.grad(x)`**; `.enable_eval()` is required before `.eval(x)`. Otherwise `RuntimeError`.
- **Don't pass tensors with `requires_grad=True` into `.grad(x)`** — `vmap(grad(...))` handles autograd internally; the input does not need `requires_grad_`.
- **Static-buffer hazard:** if you cache the output of `.grad(x)` / `.eval(x)` across another `.grad()` / `.eval()` call, `.clone()` it first (reduce-overhead CUDA graph reuses the same buffer).
- **`loss_compile_beta(...)(x, beta)` requires `beta.dim() == 0`** — a Python float or a 0-d Tensor. Shape `[1]` / `[N]` for per-sample tempering is rejected at the wrapper (would silently trigger per-shape recompiles otherwise).
- **`OT_loss` takes the `OTFlow` instance, not `flow.t()`** — it dispatches to the 4-channel augmented ODE via the OTFlow object directly. Using `flow.t()` would silently fall back to a 2-channel path.
- **`linear_combination` / `Linear_Combination` store coeffs as a plain `list[float]`** — a 1-d tensor input is `.detach().cpu().tolist()`'d at construction (NOT registered as a buffer, so the coeffs are immune to `.to(device)`; a `requires_grad=True` tensor is silently detached, not rejected). Retune per rung with `set_coeffs([...])` or `lc.coeffs[k] = ...`.
- **`langevin(adjust=True, taming>0)` raises `ValueError`** — MH correction assumes the true overdamped proposal, not the tamed one.
- **`lbfgs(armijo=True)` requires `enable_eval()`**, not just `enable_grad()`.
- **`flow.zeros()` is non-stochastic identity initialisation** — call it before training as a warm start; do **not** expect post-training `zeros()` to mean anything.
- **`CNF` / `OTFlow` use fixed-step RK4** — round-trip error and ladj-vs-slogdet agreement scale with `nt` (`OTFlow` nt) / internal step count (`CNF`); bump for tighter accuracy.
- **Logs noisy?** Call `zflows.utils.suppress_warnings()` once at script start, before any `torch.compile` / `enable_grad` / `compile_raw` invocation. Bumps cache headroom with `set_cache_size_limit(64)` if you're sweeping many compiled closures.
- **Platform:** `enable_grad` / `enable_eval` / `loss_compile` / `loss_compile_beta` all need `torch.compile`, which is Linux+CUDA only (Windows users: WSL or skip the opt-ins and use plain autograd + raw `reverse_KL(x, target, flow.t())`).

---

## 7. Mathematical reference (for picking the right loss)

For source $\mu_0 \propto \exp(-U_0)$, target $\mu_1 \propto \exp(-U_1)$, learned bijection $F: \mathbb R^d \to \mathbb R^d$ with pushforward $F_\# \mu_0$:

- **Reverse KL** (only $U_1$ needed, energy-based):
  $$\mathcal L_{\mathrm{rev}}[F] = \mathbb E_{x \sim \mu_0}[U_1(F(x)) - \log |\det J_F(x)|].$$
- **Forward KL** (only samples from $\mu_1$ needed, data-driven):
  $$\mathcal L_{\mathrm{fwd}}[F] = \mathbb E_{y \sim \mu_1}[U_0(F^{-1}(y)) + \log |\det J_{F^{-1}}(y)|].$$
- **Importance log-weights**:
  $$\log w(y) = -U_1(F(x)) + U_0(x) + \log |\det J_F(x)|, \quad y = F(x), x \sim \mu_0.$$
- **ESS** in $[0, 1]$: $\mathrm{ESS} = (\sum w_i)^2 / (N \sum w_i^2)$.

---

## 8. End-to-end working examples (in the repo)

Located under `/mnt/projects/zflows/tests/`:

- `2D_reverse_KL.py` — `NSF` + reverse KL on a 2D energy with cosine perturbation; ESS diagnostic.
- `2D_forward_KL.py` — `NSF` + forward KL on a 3-mode Gaussian mixture.
- `3D_periodic.py` — `NCSF` on the 3-torus; full IS → resample → MALA rejuvenation pipeline. Uses `loss_compile` for the training step.
- `4D_Boltzmann_generator.py` — annealed `NSF` with `linear_combination` bridges; warm-started flow across rungs; one hoisted `loss_compile(reverse_KL, u_curr, F)` retuned per rung via `u_curr.set_coeffs([c_k, 1 - c_k])`.
- `2D_two_moon_CNF.py` — `CNF` + forward KL on the two-moons distribution.
- `2D_RealNVP_latent_interpolation.py` — `RealNVP` (with `mixing="lu"`) and closed-form `F.inv` for latent-space interpolation.
- `compare_compiled_loss.py` — performance benchmark of `loss_compile` vs. raw `reverse_KL` across a `dimension × hidden_features` grid (writeup in `compare_compiled_loss.md`).
- `multi_well_compare.py` — `CNF` vs. `OTFlow` ESS / wall-clock comparison on a multi-well target.

Verification suites (in the same directory): `_verify_flow.py`, `_verify_potential.py`, `_verify_utils.py`. Run from the repo root with `.venv/bin/python -m tests._verify_flow` etc.

When in doubt, copy one of these as a starting skeleton and modify.
