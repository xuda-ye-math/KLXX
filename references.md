# IS-Enabled Methods for Boltzmann Generator Training: A Survey of Mode-Collapse Mitigation

**Scope.** This document surveys methods that train normalizing flows
(or other models with **tractable density** $q_\theta(x)$) for sampling
from $\pi(x) \propto \exp(-U(x))$ while addressing mode collapse. Methods
without tractable density (DiKL, iDEM, PIS, DDS, PDNS) are excluded
because they preclude test-time importance-sampling reweighting — the
single most important property for quantitative physical observables.

For each method we state:

1. **The loss** (mathematically rigorous).
2. **The mode-collapse mitigation strategy.**
3. **Data-driven vs. energy-driven** classification.
4. **Cost** relative to vanilla reverse-KL flow training.
5. **IS compatibility** at test time.

Notation throughout: target $\pi(x) = Z^{-1}\tilde\pi(x)$ with
$\tilde\pi(x) = \exp(-U(x))$ on $\mathbb{R}^d$; flow density $q_\theta$
with base $\pi_0 = \mathcal{N}(0, I_d)$; baseline cost $C_{\mathrm{KL}}$
denotes one reverse-KL gradient step (one flow forward + one $U$
evaluation, batched).

---

## 1. Reference: Reverse KL Training

The default Boltzmann generator objective (Noé et al., 2019):
$$
\mathcal{L}_{\mathrm{KL}}(\theta) \;=\; \mathrm{KL}(q_\theta \,\|\, \pi)
\;=\; \mathbb{E}_{x \sim q_\theta}\!\bigl[\log q_\theta(x) + U(x)\bigr] \;+\; \log Z.
$$

- **Strategy against mode collapse**: none. Reverse KL is mode-seeking
  (Bishop, 2006; Minka, 2005); if $q_\theta$ misses a mode, no gradient
  pulls it back.
- **Classification**: pure energy-driven.
- **Cost**: $1\times C_{\mathrm{KL}}$ (definition).
- **IS at test time**: yes — flow density is tractable.

All methods below extend this baseline. We measure cost in multiples of
$C_{\mathrm{KL}}$.

---

## 2. FAB: Flow Annealed Importance Sampling Bootstrap

**Reference**: Midgley, Stimper, Simm, Schölkopf, Hernández-Lobato, ICLR 2023
(arXiv:2208.01893).

### 2.1 Loss

The $\alpha = 2$ divergence
$$
D_2(\pi \,\|\, q_\theta) \;=\; \frac{1}{2}\int \frac{\pi(x)^2}{q_\theta(x)}\,\mathrm{d}x \;-\; \frac{1}{2}.
$$
The gradient
$$
\nabla_\theta D_2(\pi \,\|\, q_\theta)
\;=\; -\int \frac{\pi(x)^2}{q_\theta(x)}\,\nabla_\theta\!\log q_\theta(x)\,\mathrm{d}x
$$
is estimated by AIS (Neal, 2001) along the path
$$
\pi_t(x) \;\propto\; q_\theta(x)^{1-2\beta_t}\,\pi(x)^{2\beta_t}, \qquad \beta_t: 0 \to 1,
$$
targeting the variance-optimal proposal $g_\theta \propto \pi^2/q_\theta$.
Self-normalized estimator:
$$
\widehat{\nabla_\theta D_2} \;=\; -\frac{\sum_i w^{(i)}_{\mathrm{AIS}}\,\nabla_\theta\!\log q_\theta(x_i)}{\sum_i w^{(i)}_{\mathrm{AIS}}}.
$$
A **prioritized replay buffer** (Schaul et al., 2015) amortizes AIS cost
across $K_{\mathrm{refresh}}$ gradient steps.

### 2.2 Mode-collapse strategy

Two compounded mechanisms.

**(a) Mass-covering loss.** $D_2(\pi\|q_\theta)$ has $\pi$ in the
numerator and $q_\theta$ in the denominator: missed modes appear as
quadratic divergences in the integrand, strongly penalizing zero-coverage.

**(b) Mode-seeking AIS proposal.** $g_\theta \propto \pi^2/q_\theta$
concentrates exactly on regions where $q_\theta$ underapproximates $\pi$
— singular at lost modes. AIS samples drawn from this proposal land on
the missed regions (when AIS bridges the path successfully), producing
gradient signal that pulls the flow toward the missing mass.

### 2.3 Classification

**Pure energy-driven.** Requires only $\tilde\pi$ and $\nabla U$. No
target samples needed.

### 2.4 Cost

Per gradient step with replay buffer ($T$ AIS temperatures, $K$ MCMC
steps per temperature, $L$ leapfrog steps per HMC, refreshed every
$K_{\mathrm{refresh}}$ gradient steps):
$$
\frac{C_{\mathrm{FAB}}}{C_{\mathrm{KL}}} \;\approx\; \frac{TKL}{K_{\mathrm{refresh}}} + 1.
$$
With Midgley et al.'s typical settings ($T = 8$, $K = 1$, $L = 5$,
$K_{\mathrm{refresh}} = 8$): $\approx 5\text{–}6\times$. Plus
$\sim 1.5\text{–}2\times$ from slower convergence in optimizer steps.
**End-to-end: $\sim 5\text{–}10\times$** wall-clock vs. vanilla KL.

### 2.5 IS at test time

**Yes.** $q_\theta$ is a flow with tractable density. After training,
unbiased IS estimator with weights $\tilde w_i = \tilde\pi(x_i)/q_\theta(x_i)$
is available. Moreover, $\alpha = 2$ training optimizes IS-weight variance
specifically — the trained flow is asymptotically optimal as IS proposal.

### 2.6 Failure regime

- AIS path resolution: $T = O(\sqrt{d})$ (Grosse et al., 2013).
- MCMC mixing on $\pi_t$ inherits $\pi$'s barriers (no $\beta = 0$
  uniform distribution along the path).
- ESS exponential decay in $d$ for self-normalized IS.
- Singular $g_\theta$ at lost modes: helps if AIS reaches them, fatal
  if it doesn't.
- Demonstrated comfort zone: $d \lesssim 60$ (alanine dipeptide). Fails
  on LJ-55 ($d = 165$).

---

## 3. Jeffreys Flow: Symmetric KL Distillation from Parallel Tempering

**Reference**: Lin, Moya, Qi, Ye, 2026 (arXiv:2604.05303).

### 3.1 Loss

The Jeffreys divergence is the symmetrization of KL:
$$
J(\pi \,\|\, q_\theta) \;=\; \mathrm{KL}(\pi \,\|\, q_\theta) + \mathrm{KL}(q_\theta \,\|\, \pi).
$$
With samples $\{x^{\mathrm{PT}}_j\}_{j=1}^N \sim \pi$ from a Parallel
Tempering simulation, the empirical Jeffreys loss is
$$
\widehat J(\theta) \;=\; -\frac{1}{N}\sum_{j=1}^N \log q_\theta(x^{\mathrm{PT}}_j)
\;+\; \mathbb{E}_{x \sim q_\theta}\!\bigl[\log q_\theta(x) + U(x)\bigr]
\;+\; \mathrm{const}.
$$
First term = forward KL = NLL on PT samples (mass-covering).
Second term = reverse KL = energy-based (target-seeking precision).
The paper extends this with Rényi-divergence variants for variance
control.

### 3.2 Mode-collapse strategy

The forward-KL term is the key ingredient. PT samples cover all modes
of $\pi$ (this is what PT is designed for), so the NLL term forces
$q_\theta$ to assign positive density everywhere PT visits. Reverse KL
alone would mode-collapse; reverse KL + forward KL on PT data cannot.
The Jeffreys form balances target-seeking accuracy against mass coverage.

### 3.3 Classification

**Data-driven.** Requires PT trajectories at multiple temperatures —
arguably the most expensive part of the algorithm. Energy-driven only
in the reverse-KL component.

### 3.4 Cost

Two distinct costs.

**Pre-training cost: PT trajectory generation.** Parallel Tempering with
$R$ replicas, each running $N_{\mathrm{MD}}$ MD steps with swap acceptance
rate $\sim 23\%$. Cost scales as $R \cdot N_{\mathrm{MD}}$ MD evaluations.
For molecular systems with classical force fields this is the dominant
cost — typically $10^6\text{–}10^9$ force evaluations to converge PT for
small biomolecules.

**Per-gradient-step training cost.** Forward pass through flow on PT
samples (NLL) + reverse-KL gradient as in baseline. Roughly $2 \times C_{\mathrm{KL}}$
per step.

**End-to-end**: dominated by PT cost. If PT samples are already available
(e.g., from prior simulations), training is cheap. If PT must be run
afresh, total cost can be $\gg 100 \times C_{\mathrm{KL}}$ in
sample-generation alone.

### 3.5 IS at test time

**Yes.** Standard normalizing flow density.

### 3.6 Failure regime

- Quality of trained $q_\theta$ bounded by quality of PT samples. If PT
  is poorly converged (insufficient swap acceptance, inadequate replica
  count), $q_\theta$ inherits the bias.
- Cannot extrapolate to mode-coverage beyond what PT achieved.
- Demonstrated up to $d = 16$ in published benchmarks.

---

## 4. TA-BG: Temperature-Annealed Boltzmann Generators

**Reference**: Schopmans & Friederich, 2025 (arXiv:2501.19077).

### 4.1 Loss

Two-phase training.

**Phase 1: high-temperature reverse KL.** Train $q_\theta$ at temperature
$T_{\mathrm{high}}$ (typically $5\text{–}10\times$ target temperature $T_0$):
$$
\mathcal{L}_{\mathrm{high}}(\theta) \;=\; \mathrm{KL}\!\bigl(q_\theta \,\|\, \pi^{(T_{\mathrm{high}})}\bigr),
\qquad \pi^{(T)}(x) \;\propto\; \exp(-U(x)/T).
$$
At high $T$, the energy landscape is flatter and reverse KL does not
mode-collapse — the high-$T$ distribution itself is unimodal or weakly
multimodal.

**Phase 2: iterative reweighting to lower temperatures.** Anneal the
target via a sequence $T_0 < T_1 < \cdots < T_{\mathrm{high}}$. At each
step, train $q_{\theta_{i+1}}$ to match $\pi^{(T_i)}$ using importance
weighted samples from $q_{\theta_i}$:
$$
\mathcal{L}_i(\theta_{i+1}) \;=\; -\mathbb{E}_{x \sim q_{\theta_i}}\!\left[\frac{\pi^{(T_i)}(x)/q_{\theta_i}(x)}{Z_{\mathrm{IS}}}\,\log q_{\theta_{i+1}}(x)\right],
$$
i.e., an importance-weighted maximum likelihood. Effectively forward-KL
to $\pi^{(T_i)}$ using $q_{\theta_i}$ as proposal.

### 4.2 Mode-collapse strategy

**High-$T$ initialization solves mode discovery.** At high $T$, the
target is easy to sample (single basin or weakly metastable). Reverse-KL
at high $T$ converges to a $q_\theta$ that covers all of $\pi^{(T_{\mathrm{high}})}$.
Annealing then preserves coverage as long as each step does not
re-introduce mode collapse.

The crucial assumption: modes of $\pi$ at $T_0$ are not separated from
$\pi^{(T_{\mathrm{high}})}$'s mass — i.e., they emerge by *concentration*
during cooling, not by *appearance* of new basins. This holds for most
molecular systems under reasonable choice of $T_{\mathrm{high}}$.

### 4.3 Classification

**Pure energy-driven.** No samples required at any stage. The IS proposal
at each annealing step is the previous flow.

### 4.4 Cost

Phase 1: standard reverse-KL training at $T_{\mathrm{high}}$. Cost
$\sim C_{\mathrm{KL}}$.

Phase 2: $K_{\mathrm{anneal}}$ annealing steps, each requiring
- importance-weighted ML training (cheap: $\sim C_{\mathrm{KL}}$ per gradient step);
- $K_{\mathrm{phase}}$ gradient steps per annealing temperature;

Total cost: $\sim (1 + K_{\mathrm{anneal}} \cdot K_{\mathrm{phase}}) \times C_{\mathrm{KL}}$.

In practice: $K_{\mathrm{anneal}} \sim 10\text{–}50$, $K_{\mathrm{phase}} \sim 10^3\text{–}10^4$. Total **$\sim 10\text{–}50\times C_{\mathrm{KL}}$** if annealing is well-tuned.

### 4.5 IS at test time

**Yes.** Trained flow has tractable density.

### 4.6 Failure regime

- **Annealing step size sensitivity.** If $T_{i+1} - T_i$ too large,
  $q_{\theta_i}$ has insufficient overlap with $\pi^{(T_i)}$ and ESS
  collapses; if too small, training is slow. The user explicitly tested
  this and found instability — this is the algorithmic Achilles heel.
- **Modes that appear during cooling are missed.** If two basins of $\pi$
  merge into one at $T_{\mathrm{high}}$ but separate during cooling, the
  high-$T$ initialization places mass at the merger; annealing concentrates
  it on whichever basin happens to be initially favored, losing the other.
- Demonstrated up to alanine dipeptide ($d \approx 60$). Untested at LJ-55
  scale.

---

## 5. AFT and CRAFT: Annealed Flow Transport Monte Carlo

**References**:
- Arbel, Matthews, Doucet, ICML 2021 (arXiv:2102.07501) — AFT.
- Matthews, Arbel, Rezende, Doucet, ICML 2022 (arXiv:2201.13117) — CRAFT.

### 5.1 Loss

Define an annealing path of intermediate distributions
$$
\pi_t(x) \;\propto\; \pi_0(x)^{1-\beta_t}\,\pi(x)^{\beta_t}, \qquad \beta_t: 0 \to 1, \quad t = 1, \ldots, T.
$$
For each interval $[t-1, t]$, train a normalizing flow
$T^{(t)}_{\theta_t}: \mathbb{R}^d \to \mathbb{R}^d$ to transport $\pi_{t-1}$
to $\pi_t$. The flow loss for segment $t$ (CRAFT version):
$$
\mathcal{L}_t(\theta_t) \;=\; \mathrm{KL}\!\bigl((T^{(t)}_{\theta_t})_{\#}\pi_{t-1} \,\|\, \pi_t\bigr),
$$
estimated using particles from the SMC simulation up to time $t-1$.
The final density $q_\theta = (T^{(T)}_{\theta_T} \circ \cdots \circ T^{(1)}_{\theta_1})_{\#}\pi_0$
has tractable density via composition.

CRAFT improves on AFT by training all flows jointly with continual
gradient updates (rather than sequentially fixing each), which is more
stable.

### 5.2 Mode-collapse strategy

**Sequential transport with SMC particle reweighting.** Mode discovery
happens at small $\beta_t$, where $\pi_t$ is close to the easy base $\pi_0$
and resembling its support. As $\beta_t$ increases, particles are reweighted
by SMC and resampled, with MCMC kernels mixing within each $\pi_t$. The
flow only needs to handle one *small* transport step at a time — much
easier than the global transport from $\pi_0$ to $\pi$.

The per-step KL is well-defined because the SMC particles approximate
$\pi_{t-1}$, providing forward-KL training data for each segment without
the risk of mode collapse: each segment's source distribution is
algorithmically guaranteed to cover the next.

### 5.3 Classification

**Pure energy-driven.** SMC particles are generated using only $\tilde\pi$
along the annealing path. No external data required.

### 5.4 Cost

Each gradient step requires running the SMC up to the current time $t$,
which is itself iterative. With $T$ annealing temperatures, $N$ particles,
and $K$ MCMC steps per temperature, one full SMC pass costs $O(TNK)$
target evaluations.

In CRAFT, an SMC pass is run alongside each gradient step, with shared
amortization. Per-step cost:
$$
\frac{C_{\mathrm{CRAFT}}}{C_{\mathrm{KL}}} \;\approx\; T \cdot K + 1.
$$
Typical $T = 10\text{–}50$, $K = 1\text{–}10$. **End-to-end:
$10\text{–}500\times C_{\mathrm{KL}}$**, dominated by SMC cost. Heavier
than FAB but more reliable for hard targets.

### 5.5 IS at test time

**Yes.** The composed flow $q_\theta = (T^{(T)}_{\theta_T} \circ \cdots \circ T^{(1)}_{\theta_1})_{\#}\pi_0$
has tractable density via the change-of-variables formula. Furthermore,
the final SMC ensemble itself provides weighted samples that approximate
$\pi$ directly with controlled bias.

### 5.6 Failure regime

- Cost scales linearly in $T$ (annealing temperatures), expensive for
  high-barrier systems requiring fine annealing schedules.
- Training instability for very large $T$ or aggressive schedules.
- Each segment's flow is small (since the local transport is small),
  but total parameter count $\sim T \cdot |\theta|$ can be large.
- Demonstrated on lattice field theories and high-dimensional
  Bayesian inference; performance on molecular systems competitive with
  FAB on benchmarks where both are run.

---

## 6. SNF: Stochastic Normalizing Flows

**Reference**: Wu, Köhler, Noé, NeurIPS 2020 (arXiv:2002.06707).

### 6.1 Loss

An SNF interleaves $L$ deterministic invertible maps
$\{F^{(\ell)}_{\theta_\ell}\}_{\ell=1}^L$ with $L$ stochastic kernels
$\{\kappa^{(\ell)}\}_{\ell=1}^L$ (e.g., Langevin or HMC steps). The path
distribution is
$$
P_\theta(z_0, z_1, \ldots, z_L) \;=\; \pi_0(z_0)\,\prod_{\ell=1}^L \kappa^{(\ell)}(z_\ell \mid F^{(\ell)}_{\theta_\ell}(z_{\ell-1})).
$$
The training loss is a path-space KL between $P_\theta$ and a backward
reference path $Q$ ending at $\pi$:
$$
\mathcal{L}_{\mathrm{SNF}}(\theta) \;=\; \mathrm{KL}\bigl(P_\theta \,\|\, Q\bigr)
\;=\; \mathbb{E}_{P_\theta}\!\left[-\log\frac{Q(z_0, \ldots, z_L)}{P_\theta(z_0, \ldots, z_L)}\right],
$$
where $Q$ uses the reverse stochastic kernels and terminal $\pi$.
Path-space ELBO: bounds $\mathrm{KL}(\mathrm{marg}_L P_\theta \| \pi)$
from above. Optimization is by reparameterized gradient through the
deterministic blocks; stochastic blocks contribute via importance
weights computed from kernel densities.

### 6.2 Mode-collapse strategy

**Stochastic kernels break invertibility constraints locally.** In a pure
flow, no diffeomorphism can change the topology of the density. SNF inserts
Langevin/HMC steps that allow particles to *cross* between modes — the
stochastic block can take a particle from one basin to another, something
no deterministic invertible map can do efficiently.

Effectively, the stochastic blocks act as the "AIS bridge" inside the
sampler, with the deterministic flow refining within each basin.

### 6.3 Classification

**Pure energy-driven** (requires only $U$ and $\nabla U$ for the Langevin
kernels). Optionally augmentable with data via a forward-KL term on
training samples.

### 6.4 Cost

Each forward pass evaluates $L$ flow blocks plus $L$ MCMC kernels (each
typically requiring $\nabla U$). Per gradient step:
$$
\frac{C_{\mathrm{SNF}}}{C_{\mathrm{KL}}} \;\approx\; L_{\mathrm{stoch}} + 1,
$$
where $L_{\mathrm{stoch}}$ is the number of stochastic kernel calls per
forward pass. Typical $L_{\mathrm{stoch}} = 5\text{–}20$. **End-to-end:
$5\text{–}20\times C_{\mathrm{KL}}$.**

### 6.5 IS at test time

**Yes — but on path-space.** The marginal density $q_\theta(x_L)$ is not
in closed form, but the path-space ratio $P_\theta/Q$ provides a
self-normalized importance weight that gives unbiased estimates of
$\mathbb{E}_\pi[f]$. This is the standard SNF inference procedure (Wu
et al. 2020, §2.3). For physical observables this is fine; for tasks
requiring marginal density evaluation specifically, additional work is
needed.

### 6.6 Failure regime

- Number of stochastic blocks $L_{\mathrm{stoch}}$ scales with the
  difficulty of the target (more barriers ⇒ more Langevin steps needed).
- For very high-barrier systems, path-space ESS collapses similarly to
  the FAB / AIS pathology.
- Less developed in the BG-specific literature than FAB; benchmarks tend
  to be smaller-scale.

---

## 7. CMT: Constrained Mass Transport

**Reference**: Klitzing, Hagemann, et al., 2025 (arXiv:2510.18460).

### 7.1 Loss

CMT replaces the single-shot reverse-KL minimization with a sequence of
constrained subproblems. At iteration $i$, given current flow density
$q_{i-1}$, find $q_i$ as the solution of
$$
\begin{aligned}
q_i \;=\; \arg\min_{q}\;& \mathrm{KL}(q \,\|\, \pi) \\
\text{subject to}\;& \mathrm{KL}(q \,\|\, q_{i-1}) \;\leq\; \epsilon, \\
& H(q) \;\geq\; H(q_{i-1}) - \delta,
\end{aligned}
$$
where $H(q) = -\int q\log q\,\mathrm{d}x$ is the differential entropy.

The first constraint is a **trust region** (Schulman et al. 2015's TRPO,
adapted): $q_i$ must stay close to $q_{i-1}$. The second is an **entropy
floor**: prevents $q_i$ from collapsing too fast.

The Lagrangian is
$$
\mathcal{L}_i(q) \;=\; \mathrm{KL}(q \,\|\, \pi) + \lambda_1\,\bigl[\mathrm{KL}(q \,\|\, q_{i-1}) - \epsilon\bigr] + \lambda_2\,\bigl[H(q_{i-1}) - \delta - H(q)\bigr],
$$
solved per iteration with normalizing flow parameterizations. The
sequence $\{q_i\}$ defines a geometric annealing path with
**automatically tuned** schedule.

### 7.2 Mode-collapse strategy

Two distinct mechanisms.

**(a) Trust region prevents abrupt jumps.** If $q_i$ tries to
mode-collapse, the trust-region constraint $\mathrm{KL}(q_i \| q_{i-1}) \leq \epsilon$
prevents it from doing so in a single step. Mode loss requires
$\mathrm{KL}(q_i \| q_{i-1}) \to \infty$ (since $q_{i-1}$ has support on
the mode but $q_i$ doesn't). Hence the trust region forces gradual
transitions.

**(b) Entropy floor prevents mass teleportation.** "Mass teleportation"
is the phenomenon where probability mass abruptly shifts to a new region,
leaving the old region with negligible density (Klitzing et al. 2025).
The entropy floor $H(q_i) \geq H(q_{i-1}) - \delta$ caps the rate at
which the flow becomes peaky, preventing it from teleporting mass to a
high-density region while abandoning others.

### 7.3 Classification

**Pure energy-driven.** No samples required. Each constrained subproblem
uses only $\tilde\pi$ and the previous flow.

### 7.4 Cost

Each constrained subproblem solves a Lagrangian dual, typically requiring
$K_{\mathrm{inner}}$ gradient steps with adaptive Lagrange multipliers.
Outer loop: $K_{\mathrm{outer}}$ trust-region steps until convergence.
Total:
$$
\frac{C_{\mathrm{CMT}}}{C_{\mathrm{KL}}} \;\approx\; K_{\mathrm{outer}} \cdot K_{\mathrm{inner}}.
$$
Typical $K_{\mathrm{outer}} \sim 50\text{–}200$, $K_{\mathrm{inner}} \sim 100\text{–}1000$.
**End-to-end: $5\text{–}50\times C_{\mathrm{KL}}$**, depending on schedule.

The entropy term $H(q)$ is intractable in closed form for most flows;
estimated by Monte Carlo, adding variance.

### 7.5 IS at test time

**Yes.** Final flow has tractable density.

### 7.6 Failure regime

- Trust-region radius $\epsilon$ and entropy floor $\delta$ require
  tuning. The automatic schedule helps but doesn't eliminate the need.
- Estimating $H(q)$ accurately is difficult in high $d$ — the entropy
  constraint can become noisy at $d \gtrsim 50$.
- Reportedly outperforms FAB on alanine dipeptide; broader benchmarking
  pending (paper is October 2025).

---

## 8. Log-Variance Loss for Continuous Flows

**Reference**: Richter & Berner, ICLR 2024 (arXiv:2307.01198), with
flow-applicable variant in Berner, Richter, Ullrich, TMLR 2024.

### 8.1 Loss

For a continuous normalizing flow parameterized as a controlled SDE
$$
\mathrm{d}X^u_t \;=\; (b(X^u_t, t) + \sigma(X^u_t, t)\,u_\theta(X^u_t, t))\,\mathrm{d}t \;+\; \sigma(X^u_t, t)\,\mathrm{d}W_t,
$$
with control $u_\theta$, the path-space log-variance divergence is
$$
D_{\mathrm{LV}}(P^u, Q) \;=\; \mathrm{Var}_{\tilde P}\!\bigl[\log(\mathrm{d}P^u / \mathrm{d}Q)\bigr],
$$
where $P^u$ is the law of $X^u$, $Q$ is a reference path measure with
terminal $\pi$, and $\tilde P$ is an arbitrary reference (often $P^u$
itself with stop-gradient, or a fixed initial $P^{u_0}$).

### 8.2 Mode-collapse strategy

The log-variance loss has the **sticking-the-landing property** (Roeder
et al. 2017): at the optimum where $P^u = Q$, the integrand
$\log(\mathrm{d}P^u/\mathrm{d}Q) = 0$, hence its variance is zero,
and the gradient estimator has zero variance. Standard reverse-KL gradients
do not have this property: even at the optimum, the score-function
gradient term has non-vanishing variance.

For mode collapse specifically: since the loss is a variance, it is
sensitive to *any* mismatch between $P^u$ and $Q$, not just the mean
discrepancy that reverse KL captures. Modes that $u_\theta$ misses
contribute to the variance even when the mean log-ratio is small.

### 8.3 Classification

**Pure energy-driven.** Reference path measure $Q$ is constructed from
$U$ and the SDE drift; no samples needed.

### 8.4 Cost

Per gradient step, simulate the SDE forward over $N_t$ Euler-Maruyama
steps, evaluate $u_\theta$ and $\log(\mathrm{d}P^u/\mathrm{d}Q)$. Cost:
$$
\frac{C_{\mathrm{LV}}}{C_{\mathrm{KL}}} \;\approx\; N_t,
$$
typically $N_t = 50\text{–}500$. **End-to-end: $50\text{–}500\times C_{\mathrm{KL}}$.**

Higher than discrete-time flow training, but log-variance loss does NOT
require backprop through the SDE solver (unlike KL-based path-space
losses), saving memory.

### 8.5 IS at test time

**Yes — on path-space.** The path-space Radon-Nikodym derivative
$\mathrm{d}P^u/\mathrm{d}Q$ is computable from Girsanov's theorem,
giving unbiased IS estimates for $\mathbb{E}_\pi[f]$. Marginal density
$q_\theta(x_T)$ requires integrating the SDE, expensive.

### 8.6 Failure regime

- Applicable only to continuous-time / SDE-based parameterizations.
  Discrete-time invertible flows are not covered without modification.
- Sensitive to SDE solver step size; coarse discretization biases the
  estimator.
- Recent benchmarks (Sanokowski et al. 2025, *Rethinking Losses for
  Diffusion Bridge Samplers*) indicate that LV loss can be unstable
  during training despite the sticking-the-landing property at the
  optimum, requiring careful hyperparameter tuning.

---

## 9. Comparison Table

| Method | Loss type | Strategy | Data-driven? | Cost ($\times C_{\mathrm{KL}}$) | IS at test |
|--------|-----------|----------|:------------:|:-------------------------------:|:----------:|
| Reverse KL | $\mathrm{KL}(q_\theta\|\pi)$ | None | No | 1 | Yes |
| FAB | $D_2(\pi\|q_\theta)$ + AIS | Mass-covering loss + AIS to spike $\pi^2/q_\theta$ | No | 5–10 | Yes |
| Jeffreys Flow | $\mathrm{KL}(\pi\|q) + \mathrm{KL}(q\|\pi)$ on PT | Forward KL on PT samples enforces coverage | **Yes (PT)** | 2 + PT cost | Yes |
| TA-BG | High-$T$ rev-KL → IS-reweighted ML | High-$T$ trivial coverage + gradual annealing | No | 10–50 | Yes |
| CRAFT | Per-segment $\mathrm{KL}$ on SMC particles | SMC discovers modes, each segment is small | No | 10–500 | Yes |
| SNF | Path-space KL with stochastic blocks | Langevin steps cross barriers inside the sampler | No | 5–20 | Yes (path-space) |
| CMT | Trust-region + entropy-constrained KL | Bounded change per step prevents collapse + mass teleportation | No | 5–50 | Yes |
| Log-Variance | $\mathrm{Var}[\log\mathrm{d}P^u/\mathrm{d}Q]$ | Zero-variance gradient at optimum + path-space sensitivity | No | 50–500 | Yes (path-space) |

---

## 10. Strategic Summary

### 10.1 Pure energy-driven (no samples)

**FAB** and **TA-BG** rely on *bootstrapping*: FAB bootstraps via AIS to
a non-trivial proposal; TA-BG bootstraps via high-temperature initialization.
Both are vulnerable to the bootstrap failing — FAB if AIS doesn't cross
barriers, TA-BG if annealing step sizes are wrong.

**CRAFT** and **AFT** rely on *sequential transport* with SMC. The cost
is high but the algorithm is robust because each segment is small and
SMC particles provide reliable training signal.

**SNF** relies on *embedded MCMC* — Langevin steps cross barriers
within the model itself. Lowest variance among pure energy-driven methods
on simple targets, but cost scales with barrier height.

**CMT** is the newest and most distinctive: explicit trust-region and
entropy constraints prevent mode loss *during* training rather than
trying to recover from it. Conceptually the cleanest approach to the
mode-collapse problem.

**Log-variance** is mainly a tool for SDE-based samplers and is not a
direct replacement for flow training.

### 10.2 Data-driven

**Jeffreys Flow** is the only method here that requires upstream
sample generation (PT trajectories). The cost is significant but the
mode-coverage guarantee is the strongest: forward KL on PT samples
*provably* prevents mode collapse to the extent PT itself doesn't
mode-collapse.

### 10.3 For your specific setting

You have high-temperature samples + minima locations. This puts you
in an *intermediate* regime between pure energy-driven and full PT
data-driven:

- **vs. TA-BG**: your high-T samples give you Phase 1 for free (no
  reverse-KL training needed at $T_{\mathrm{high}}$), and your minima
  give you mode-anchoring information that TA-BG doesn't use. The
  annealing step instability you experienced is intrinsic to TA-BG's
  pure reweighting, and is exactly what trust-region constraints (CMT)
  or SMC particle resampling (CRAFT) are designed to prevent.

- **vs. CMT**: the trust-region + entropy-constraint mechanism is the
  closest in spirit to "stable annealing without mode loss". But CMT
  is designed to discover modes from scratch via the entropy floor;
  with minima already known, you can replace the entropy constraint
  with a more direct *mode-mass-conservation* constraint, which should
  be both cheaper and more reliable.

- **vs. CRAFT**: SMC + per-segment flows is robust but expensive. With
  minima locations as anchors, you can replace SMC's mode-discovery role
  with direct anchoring, potentially saving a factor of $T \cdot K$ in
  cost.

- **vs. Jeffreys Flow** (your prior work): this method needs PT samples
  *at every temperature*. Replacing PT samples at intermediate temperatures
  with minima-anchored constraints would be a direct simplification of
  Jeffreys Flow — same loss family, much less data.

The cleanest framing of your contribution may be: **CMT-style constrained
loss, but with mode-mass-conservation constraints anchored at known
minima instead of generic entropy floors**. This combines the most robust
mode-preservation mechanism in the literature with information you have
that the existing methods don't.

---

## 11. References

- Arbel, M., Matthews, A., & Doucet, A. (2021). *Annealed Flow Transport
  Monte Carlo.* ICML 2021. arXiv:2102.07501.
- Berner, J., Richter, L., & Ullrich, K. (2024). *An optimal control
  perspective on diffusion-based generative modeling.* TMLR 2024.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.*
  Springer.
- Grosse, R. B., Maddison, C. J., & Salakhutdinov, R. R. (2013).
  *Annealing between distributions by averaging moments.* NeurIPS 2013.
- Klitzing, F., Hagemann, P., et al. (2025). *Learning Boltzmann
  Generators via Constrained Mass Transport.* arXiv:2510.18460.
- Lin, G., Moya, C., Qi, D., & Ye, X. (2026). *Jeffreys Flow: Robust
  Boltzmann Generators for Rare Event Sampling via Parallel Tempering
  Distillation.* arXiv:2604.05303.
- Matthews, A. G. D. G., Arbel, M., Rezende, D. J., & Doucet, A. (2022).
  *Continual Repeated Annealed Flow Transport Monte Carlo.* ICML 2022.
  arXiv:2201.13117.
- Midgley, L. I., Stimper, V., Simm, G. N. C., Schölkopf, B., &
  Hernández-Lobato, J. M. (2023). *Flow Annealed Importance Sampling
  Bootstrap.* ICLR 2023. arXiv:2208.01893.
- Minka, T. (2005). *Divergence measures and message passing.* Microsoft
  Research Technical Report MSR-TR-2005-173.
- Neal, R. M. (2001). *Annealed importance sampling.* Statistics and
  Computing 11(2), 125–139.
- Noé, F., Olsson, S., Köhler, J., & Wu, H. (2019). *Boltzmann generators:
  Sampling equilibrium states of many-body systems with deep learning.*
  Science 365(6457), eaaw1147.
- Nüsken, N., & Richter, L. (2021). *Solving high-dimensional
  Hamilton-Jacobi-Bellman PDEs using neural networks: perspectives from
  the theory of controlled diffusions and measures on path space.*
  Partial Differential Equations and Applications 2(48).
- Richter, L., & Berner, J. (2024). *Improved sampling via learned
  diffusions.* ICLR 2024. arXiv:2307.01198.
- Roeder, G., Wu, Y., & Duvenaud, D. K. (2017). *Sticking the landing:
  Simple, lower-variance gradient estimators for variational inference.*
  NeurIPS 2017.
- Sanokowski, S., et al. (2025). *Rethinking Losses for Diffusion Bridge
  Samplers.* arXiv:2506.10982.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). *Prioritized
  experience replay.* ICLR 2016. arXiv:1511.05952.
- Schopmans, H., & Friederich, P. (2025). *Temperature-Annealed Boltzmann
  Generators.* arXiv:2501.19077.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M. I., & Moritz, P.
  (2015). *Trust region policy optimization.* ICML 2015.
- Wu, H., Köhler, J., & Noé, F. (2020). *Stochastic Normalizing Flows.*
  NeurIPS 2020. arXiv:2002.06707.
