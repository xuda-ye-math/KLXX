# Fisher–Rao and JKO analyses of the $\mathrm{X}$-augmented forward KL

## Setup and first variations

Throughout, $\mu$ is a fixed target on $\mathbb R^d$, $\nu$ a probability density to be optimized, and
$$
r(x) \;:=\; \log\frac{\mu(x)}{\nu(x)}, \qquad W(x) \;:=\; \frac{\mu(x)}{\nu(x)} = e^{r(x)}.
$$
The objective is
$$
\mathcal F[\nu] \;=\; \mathrm{KL}(\mu\Vert\nu) \;+\; \mathrm{X}_\mu(\mu\Vert\nu),
$$
with
$$
\mathrm{KL}(\mu\Vert\nu) = \int\mu\log\tfrac{\mu}{\nu}\,dx, \qquad
\mathrm{X}_\mu(\mu\Vert\nu) = \iint \mu(x)\mu(y)\,|r(x)-r(y)|\,dx\,dy.
$$

**First variation of KL.** $\delta\mathrm{KL}/\delta\nu = -\mu/\nu$.

**First variation of $\mathrm{X}_\mu$.** Using $\delta r(x)/\delta\nu(z) = -\delta(x-z)/\nu(x)$, the subdifferential $\partial|s|/\partial s = \operatorname{sgn}(s)$, and antisymmetry of $\operatorname{sgn}$,
$$
\boxed{\;\frac{\delta\,\mathrm{X}_\mu}{\delta\nu}(z) \;=\; -\,\frac{2\,\mu(z)}{\nu(z)}\,S(z),\qquad
S(z) := \mathbb E_{Y\sim\mu}\bigl[\operatorname{sgn}(r(z)-r(Y))\bigr] \in [-1,1].\;}
$$
$S$ is the rank field of the log-ratio under $\mu$: $S(z)>0$ where $\nu$ undercovers a mode of $\mu$, $S(z)<0$ where it overcovers, $S\equiv 0$ at $\nu=\mu$, and $\mathbb E_\mu[S] = 0$ by symmetry.

Combined first variation, with $\lambda=1$:
$$
\frac{\delta\mathcal F}{\delta\nu}(z) \;=\; -\frac{\mu(z)}{\nu(z)}\bigl(1+2S(z)\bigr).
$$
The factor $1+2S \in [-1,3]$ acts as a rank-based modulator. The two routes below — Fisher–Rao and JKO — yield contrasting convergence guarantees from this common variational data.

---

## 1. Fisher–Rao gradient flow

### 1.1 The Fisher–Rao metric

Tangent vectors at $\nu\in\mathcal P(\mathbb R^d)$ are signed measures $\sigma$ with $\int\sigma = 0$. The Fisher–Rao (a.k.a. spherical Hellinger, information-geometric) metric is
$$
\boxed{\;\langle \sigma_1, \sigma_2\rangle^{\mathrm{FR}}_\nu \;=\; \int \frac{\sigma_1(x)\,\sigma_2(x)}{\nu(x)}\,dx.\;}
$$
The change of variable $\nu\mapsto 2\sqrt{\nu}$ sends $\mathcal P$ isometrically onto a piece of the unit sphere in $L^2$; FR geodesics interpolate $\sqrt{\nu}$ linearly (with renormalization), equivalently interpolate $\log\nu$ affinely modulo a constant. The geometry is **purely multiplicative**: mass at $x$ is reweighted, never transported.

### 1.2 Riemannian gradient formula

The FR gradient of $\mathcal F$ at $\nu$ is the tangent vector $G$ with $\langle G,\sigma\rangle^{\mathrm{FR}}_\nu = D\mathcal F[\sigma]$ for all $\sigma\in T_\nu\mathcal P$. Writing $G(x) = \nu(x)h(x)$, the defining identity becomes
$$
\int h(x)\,\sigma(x)\,dx \;=\; \int \tfrac{\delta\mathcal F}{\delta\nu}(x)\,\sigma(x)\,dx \quad \forall\,\sigma\text{ with }\int\sigma=0,
$$
which forces $h = \delta\mathcal F/\delta\nu - c$; the constraint $\int G = \int\nu h = 0$ fixes $c = \langle\delta\mathcal F/\delta\nu\rangle_\nu$. Hence
$$
\boxed{\;\mathrm{grad}^{\mathrm{FR}}\mathcal F(\nu) \;=\; \nu\Big(\tfrac{\delta\mathcal F}{\delta\nu} - \big\langle\tfrac{\delta\mathcal F}{\delta\nu}\big\rangle_\nu\Big),\qquad \langle\,\cdot\,\rangle_\nu := \!\int\!(\cdot)\,\nu\,dx.\;}
$$
The FR gradient flow $\partial_t\nu = -\mathrm{grad}^{\mathrm{FR}}\mathcal F$ automatically conserves total mass and admits the universal dissipation identity
$$
\frac{d}{dt}\mathcal F[\nu_t] \;=\; -\bigl\|\mathrm{grad}^{\mathrm{FR}}\mathcal F\bigr\|^2_{\mathrm{FR},\nu_t}
\;=\; -\!\int\!\bigl(\tfrac{\delta\mathcal F}{\delta\nu} - \langle\tfrac{\delta\mathcal F}{\delta\nu}\rangle_{\nu_t}\bigr)^{\!2}\,\nu_t\,dx.
$$

### 1.3 FR flow of forward KL

For $\mathrm{KL}(\mu\Vert\nu)$, $\delta\mathrm{KL}/\delta\nu = -\mu/\nu$ and $\langle -\mu/\nu\rangle_\nu = -\!\int\mu = -1$. Hence
$$
\mathrm{grad}^{\mathrm{FR}}\mathrm{KL}(\nu) \;=\; \nu(-\mu/\nu + 1) \;=\; \nu - \mu,
$$
and the FR gradient flow is the **linear** ODE on densities
$$
\boxed{\;\partial_t\nu_t \;=\; \mu - \nu_t.\;}
$$
Its closed-form solution is
$$
\boxed{\;\nu_t(x) \;=\; e^{-t}\,\nu_0(x) \;+\; (1-e^{-t})\,\mu(x).\;}
$$

### 1.4 Universal exponential rate

Differentiate KL along the flow:
$$
\frac{d}{dt}\mathrm{KL}(\mu\Vert\nu_t)
\;=\; -\!\int\!\tfrac{\mu}{\nu_t}(\mu-\nu_t)\,dx
\;=\; 1 - \!\int\!\tfrac{\mu^2}{\nu_t}\,dx
\;=\; -\,\chi^2(\mu\Vert\nu_t).
$$
The chi-squared majorizes the KL via $u\log u \le u^2 - u$ applied to $u = \mu/\nu$:
$$
\chi^2(\mu\Vert\nu) \;=\; \int\!\tfrac{\mu}{\nu}(\tfrac{\mu}{\nu}-1)\,\nu\,dx \;\ge\; \int\!\log\tfrac{\mu}{\nu}\cdot\mu\,dx \;=\; \mathrm{KL}(\mu\Vert\nu).
$$
Combining and applying Grönwall,
$$
\boxed{\;\mathrm{KL}(\mu\Vert\nu_t) \;\le\; e^{-t}\,\mathrm{KL}(\mu\Vert\nu_0).\;}
$$

> **Proposition 1.1 (FR exponential decay of forward KL).** The FR gradient flow of $\mathrm{KL}(\mu\Vert\nu)$ exists for all $\nu_0$ with $\mathrm{KL}(\mu\Vert\nu_0)<\infty$ and satisfies $\mathrm{KL}(\mu\Vert\nu_t)\le e^{-t}\mathrm{KL}(\mu\Vert\nu_0)$ for all $t\ge 0$. **No structural assumption on $\mu$ is needed** — neither log-concavity, nor a spectral gap, nor any growth condition.

### 1.5 FR flow of $\mathrm{X}_\mu$

With $\delta\mathrm{X}_\mu/\delta\nu = -2(\mu/\nu)S$, the $\nu$-mean is
$$
\bigl\langle -2(\mu/\nu)S\bigr\rangle_\nu \;=\; -2\!\int\mu\,S\,dz \;=\; -2\,\mathbb E_\mu[S] \;=\; 0,
$$
by symmetry (swapping $X\leftrightarrow Y$ negates the integrand). Therefore
$$
\mathrm{grad}^{\mathrm{FR}}\mathrm{X}_\mu(\nu) \;=\; \nu\bigl(-2(\mu/\nu)S\bigr) \;=\; -2\mu S,
$$
and the FR flow of $\mathrm{X}_\mu$ alone is
$$
\boxed{\;\partial_t\nu_t \;=\; 2\,\mu\,S(\nu_t).\;}
$$

Two observations:

- The flux is a **fixed multiple of $\mu$** modulated by the rank field $S$. Mass is *added* where $r$ exceeds its $\mu$-median (modes that $\nu$ undercovers) and *removed* otherwise.
- Probability is conserved: $\int 2\mu S\,dx = 2\mathbb E_\mu[S] = 0$.

### 1.6 Combined FR flow

For $\mathcal F = \mathrm{KL} + \mathrm{X}_\mu$,
$$
\mathrm{grad}^{\mathrm{FR}}\mathcal F(\nu) \;=\; (\nu-\mu) + (-2\mu S) \;=\; \nu - \mu(1+2S),
$$
so the flow is
$$
\boxed{\;\partial_t\nu_t \;=\; \mu\bigl(1 + 2\,S(\nu_t)\bigr) - \nu_t.\;}
$$

This is a linear ODE in $\nu_t$ with a time-varying source $\mu(1+2S(\nu_t))$ — nonlinear through $S$. Sanity checks:

- *Conservation:* $\int[\mu(1+2S) - \nu] = (1 + 2\mathbb E_\mu[S]) - 1 = 0$. ✓
- *Stationarity at $\nu=\mu$:* $r\equiv 0\Rightarrow S\equiv 0$, so $\partial_t\nu = \mu - \mu = 0$. ✓

### 1.7 Energy dissipation and convergence of the combined flow

The general FR identity gives
$$
\frac{d}{dt}\mathcal F[\nu_t]
\;=\; -\!\int\!\Bigl(\tfrac{\mu}{\nu_t}(1+2S) - 1\Bigr)^{\!2}\nu_t\,dx.
\tag{1.1}
$$
Expanding the square and using $\mathbb E_\mu[S] = 0$,
$$
\frac{d}{dt}\mathcal F[\nu_t]
\;=\; 1 - \!\int\!\tfrac{\mu^2}{\nu_t}(1+2S)^2\,dx
\;=\; -\,\chi^2(\mu\Vert\nu_t) - 4\!\int\!\tfrac{\mu^2}{\nu_t}S\,dx - 4\!\int\!\tfrac{\mu^2}{\nu_t}S^2\,dx.
$$

The sign of the second piece is settled by:

> **Lemma 1.2.** $\displaystyle\int\!\tfrac{\mu^2}{\nu}S\,dz \;=\; \tfrac{1}{2}\,\mathbb E_{X,Y\stackrel{\mathrm{iid}}{\sim}\mu}\bigl[|W(X)-W(Y)|\bigr] \;\ge\; 0.$
>
> *Proof.* $\int\!\tfrac{\mu^2}{\nu}S\,dz = \mathbb E_\mu[W(X)\,S(X)] = \mathbb E_{X,Y}[W(X)\operatorname{sgn}(W(X)-W(Y))]$ since $\operatorname{sgn}(r(X)-r(Y))=\operatorname{sgn}(W(X)-W(Y))$. Symmetrize $X\leftrightarrow Y$ and average to get $\tfrac{1}{2}\mathbb E_{X,Y}[(W(X)-W(Y))\operatorname{sgn}(W(X)-W(Y))] = \tfrac{1}{2}\mathbb E[|W(X)-W(Y)|]$. $\quad\square$

Hence both correction terms in (1.1) are $\le 0$, and the combined-flow dissipation strictly exceeds the pure-KL dissipation whenever $\nu\neq\mu$.

**Consequence A: KL decays at least as fast along the combined flow.**
$$
\frac{d}{dt}\mathrm{KL}(\mu\Vert\nu_t)
\;=\; -\!\int\!\tfrac{\mu}{\nu_t}\bigl[\mu(1+2S)-\nu_t\bigr]dx
\;=\; -\,\chi^2(\mu\Vert\nu_t) - 2\!\int\!\tfrac{\mu^2}{\nu_t}S\,dx
\;\le\; -\,\mathrm{KL}(\mu\Vert\nu_t),
$$
by Lemma 1.2 and $\chi^2\ge\mathrm{KL}$. So
$$
\boxed{\;\mathrm{KL}(\mu\Vert\nu_t) \;\le\; e^{-t}\,\mathrm{KL}(\mu\Vert\nu_0)\;\text{ along the combined FR flow.}\;}
$$

**Consequence B: $\mathrm{X}_\mu$ decreases monotonically.**
$$
\frac{d}{dt}\mathrm{X}_\mu[\nu_t]
\;=\; \int\!\bigl(-2(\mu/\nu_t)S\bigr)\bigl[\mu(1+2S)-\nu_t\bigr]dx
\;=\; -2\!\int\!\tfrac{\mu^2}{\nu_t}S\,dx - 4\!\int\!\tfrac{\mu^2}{\nu_t}S^2\,dx \;\le\; 0.
$$
$\mathrm{X}_\mu$ is non-increasing along the flow, and $\mathrm{X}_\mu[\nu_t]\to 0$ as $\nu_t\to\mu$ by continuity.

**Consequence C: $\mathcal F$ decays.**
$$
\boxed{\;\mathcal F[\nu_t] \;\le\; e^{-t}\,\mathrm{KL}(\mu\Vert\nu_0) + \mathrm{X}_\mu[\nu_0]\;\xrightarrow{t\to\infty}\; 0.\;}
$$
KL part decays exponentially at rate 1; $\mathrm{X}_\mu$ part decays monotonically and is bounded by its initial value. A unified exponential bound on $\mathcal F$ would require the functional inequality
$$
\chi^2(\mu\Vert\nu) + 4\!\int\!\tfrac{\mu^2}{\nu}S\,dx + 4\!\int\!\tfrac{\mu^2}{\nu}S^2\,dx \;\ge\; \mathrm{KL}(\mu\Vert\nu) + \mathrm{X}_\mu(\mu\Vert\nu),
$$
which holds when $r$ is uniformly bounded but not in general. The asymmetric decay reflects the role of each term: KL drives the geometry, $\mathrm{X}_\mu$ regularizes.

### 1.8 Mirror descent / replicator / EM connections

The FR flow of forward KL coincides, in three discrete-time guises, with:

**Mirror descent with negative-entropy mirror map $\Phi(\nu) = \int\nu\log\nu$.** The step
$$
\log\nu_{k+1} \;=\; \log\nu_k - \eta\,\tfrac{\delta\mathcal F}{\delta\nu}(\nu_k) + \mathrm{const}
$$
becomes, for forward KL, $\nu_{k+1}\propto\nu_k\exp(\eta\mu/\nu_k)$. In the $\eta\to 0$ limit,
$$
\partial_t\log\nu_t \;=\; \tfrac{\mu}{\nu_t} - \mathbb E_{\nu_t}\bigl[\tfrac{\mu}{\nu_t}\bigr] \;=\; \tfrac{\mu}{\nu_t} - 1,
$$
which integrates to $\partial_t\nu_t = \mu - \nu_t$, the FR flow.

**Replicator dynamics.** The equation $\partial_t\nu_t(x) = \nu_t(x)(f(x) - \langle f\rangle_{\nu_t})$ with $f = \mu/\nu_t$ is exactly the FR flow of forward KL; $\mu$ plays the role of a frequency-dependent fitness landscape.

**EM iterations.** For latent-variable mixture models with target $\mu$, an EM step at $\nu_k$ is the $\tau\to\infty$ limit of the FR proximal scheme, which is why EM converges monotonically in KL without further conditions.

---

## 2. JKO scheme

### 2.1 Definition

For step size $\tau > 0$, the **Jordan–Kinderlehrer–Otto** scheme is the proximal iteration
$$
\boxed{\;\nu_{k+1} \;=\; \arg\min_{\nu\in\mathcal P_2(\mathbb R^d)}\Bigl\{\mathcal F[\nu] + \tfrac{1}{2\tau}W_2^2(\nu,\nu_k)\Bigr\}.\;}
$$
Existence of $\nu_{k+1}$ requires lower semicontinuity and coercivity of $\mathcal F$ on $\mathcal P_2$ — both hold for $\mathrm{KL}(\mu\Vert\cdot)$ when $\mu$ has a density with $-\log\mu$ of polynomial growth, and for $\mathrm{X}_\mu(\mu\Vert\cdot)$ under similar integrability conditions.

### 2.2 Optimality conditions

Stationarity of the JKO problem yields the Euler–Lagrange equation
$$
\tfrac{\delta\mathcal F}{\delta\nu}\bigl(\nu_{k+1}\bigr)(x) \;+\; \tfrac{1}{\tau}\,\phi_{k+1}(x) \;=\; \mathrm{const},
$$
where $\phi_{k+1}$ is the Kantorovich potential for the optimal transport from $\nu_{k+1}$ to $\nu_k$: the optimal map $T_{k+1}(x) = x - \nabla\phi_{k+1}(x)$ pushes $\nu_{k+1}$ to $\nu_k$. For forward KL,
$$
\tfrac{\mu(x)}{\nu_{k+1}(x)} \;=\; \tfrac{\phi_{k+1}(x)}{\tau} + c,
$$
implicit because $\phi_{k+1}$ depends on the unknown $\nu_{k+1}$ via the OT problem.

For the combined functional $\mathcal F = \mathrm{KL} + \mathrm{X}_\mu$,
$$
\tfrac{\mu(x)}{\nu_{k+1}(x)}\bigl(1 + 2\,S_{k+1}(x)\bigr) \;=\; \tfrac{\phi_{k+1}(x)}{\tau} + c,
$$
doubly implicit — both $\phi_{k+1}$ and the rank field $S_{k+1}$ depend on $\nu_{k+1}$.

### 2.3 Explicit-Euler particle form

For small $\tau$, the implicit step linearizes to the explicit Euler update
$$
\boxed{\;X^{(i)}_{k+1} \;=\; X^{(i)}_k + \tau\,\nabla\!\Bigl[\tfrac{\mu}{\nu_k}\bigl(1+2S_k\bigr)\Bigr]\!(X^{(i)}_k) \;+\; o(\tau),\qquad X^{(i)}_k\stackrel{\mathrm{iid}}{\sim}\nu_k.\;}
$$
To compute the velocity, one needs:

- $\nu_k(x)$ and $\nabla\nu_k(x)$ — provided by the normalizing-flow architecture in closed form (this is the practical reason the paper uses normalizing flows rather than score networks);
- $S_k(x)$ — estimated from a batch of $\mu$-samples by replacing $\mathbb E_{Y\sim\mu}$ with an empirical mean of $\operatorname{sgn}(r(x) - r(Y_j))$.

### 2.4 Consistency

> **Proposition 2.1 (JKO consistency for forward KL).** Suppose $\mu$ has a density bounded above and below on every compact set, with $\int|x|^2\mu\,dx <\infty$. Let $\nu_0\in\mathcal P_2$ with $\mathrm{KL}(\mu\Vert\nu_0)<\infty$. Define the JKO iterates $\{\nu_k^\tau\}$ for step size $\tau>0$ and the piecewise-constant interpolant $\bar\nu^\tau_t := \nu^\tau_{\lfloor t/\tau\rfloor}$. Then as $\tau\downarrow 0$, $\bar\nu^\tau_t$ converges narrowly to a curve $\nu_t$ uniformly on compact $t$-intervals; $\nu_t$ is the unique curve of maximal slope of $\mathrm{KL}(\mu\Vert\cdot)$ on $(\mathcal P_2, W_2)$ starting at $\nu_0$.

The argument follows the original Jordan–Kinderlehrer–Otto proof, with two adaptations:

1. The dissipation inequality $\mathcal F[\nu_{k+1}] + \tfrac{1}{2\tau}W_2^2(\nu_{k+1},\nu_k)\le\mathcal F[\nu_k]$ — an immediate consequence of $\nu_{k+1}$ being a minimizer with $\nu_k$ as a feasible competitor — gives a uniform bound on $\sum_k W_2^2(\nu_{k+1},\nu_k)$, hence equicontinuity in $t$ of the interpolant.
2. Lower-semicontinuity and coercivity of $\mathrm{KL}(\mu\Vert\cdot)$ in the narrow topology give compactness; the limit is identified as a curve of maximal slope via the energy-dissipation balance.

The same theorem applies to $\mathcal F = \mathrm{KL} + \mathrm{X}_\mu$ once one verifies lsc and coercivity for $\mathrm{X}_\mu$, both of which follow from the absolute-value-of-log-ratio structure under mild integrability.

### 2.5 Functional descent inequality

The JKO scheme satisfies a discrete energy-dissipation inequality **without any convexity assumption**:
$$
\mathcal F[\nu_{k+1}] + \tfrac{1}{2\tau}W_2^2(\nu_{k+1},\nu_k) \;\le\; \mathcal F[\nu_k]
\quad\Longleftrightarrow\quad
\mathcal F[\nu_{k+1}] - \mathcal F[\nu_k] \;\le\; -\tfrac{1}{2\tau}W_2^2(\nu_{k+1},\nu_k).
$$
Summing telescopes:
$$
\boxed{\;\sum_{k=0}^{N-1} W_2^2(\nu_{k+1},\nu_k) \;\le\; 2\tau\bigl(\mathcal F[\nu_0] - \mathcal F[\nu_N]\bigr) \;\le\; 2\tau\,\mathcal F[\nu_0].\;}
$$
This is the only rate-relevant inequality JKO gives unconditionally. To convert it into a rate on $\mathcal F$ itself, one needs additional convexity.

### 2.6 Rate for forward KL: $O(1/k)$ from L²-convexity

$\mathrm{KL}(\mu\Vert\nu)$ is **L²-convex** in $\nu$ — direct check: $\nu\mapsto -\mu\log\nu$ is convex by concavity of $\log$. It is **not** displacement-convex along W₂ geodesics for generic $\mu$ (e.g. take $\mu$ a two-Gaussian mixture and $\nu$ the W₂-geodesic interpolant between two narrow Gaussians).

Without displacement convexity, the exponential-rate machinery of Ambrosio–Gigli–Savaré fails. With L²-convexity alone, JKO yields the standard proximal-method rate:

> **Proposition 2.2 (sublinear JKO rate for forward KL).** Let $\nu^* = \mu$ be the minimizer of $\mathcal F = \mathrm{KL}(\mu\Vert\cdot)$. Assume the JKO trajectory $\{\nu_k^\tau\}$ stays in a W₂-ball of radius $D$ around $\mu$. Then
> $$
> \mathcal F[\nu_k^\tau] - \mathcal F[\nu^*] \;\le\; \frac{D^2}{2\tau\,k}.
> $$

*Sketch.* L²-convexity gives, for any $\nu^*$,
$$
\mathcal F[\nu_{k+1}] - \mathcal F[\nu^*] \;\le\; \bigl\langle \tfrac{\delta\mathcal F}{\delta\nu}(\nu_{k+1}),\,\nu_{k+1} - \nu^*\bigr\rangle.
$$
The optimality condition (section 2.2) substitutes $\delta\mathcal F/\delta\nu = -\phi_{k+1}/\tau + \mathrm{const}$, and a Brenier-style identity converts $\langle\phi_{k+1},\nu_{k+1}-\nu^*\rangle$ into half a W₂-difference. Telescoping with $W_2(\nu_0,\nu^*)\le D$ yields the rate. $\quad\square$

The $O(1/k)$ rate is **sharp** in the absence of displacement convexity (Ambrosio–Gigli–Savaré, Chapter 4) and **much weaker** than the Fisher–Rao exponential rate of section 1.4 — this is the central tradeoff: W₂ buys you implementable transport at the cost of a slower analytic rate.

### 2.7 Rate for $\mathrm{X}_\mu$: stationarity only

$\mathrm{X}_\mu(\mu\Vert\nu)$ is **convex in $\log\nu$** but **not convex in $\nu$**. The identity (from the first-variation derivation in the preamble)
$$
\bigl|r(x) - r(y)\bigr|
\;=\; \bigl|[\log\mu(x) - \log\mu(y)] - [\log\nu(x) - \log\nu(y)]\bigr|
$$
expresses $\mathrm{X}_\mu$ as the composition of an affine map $\log\nu\mapsto \log\mu - \log\nu$ followed by an integral of $|\cdot|$, giving convexity in $\log\nu$. Composing with the concave change of variable $\nu\mapsto\log\nu$ reverses the sign, so convexity in $\nu$ fails.

Implications for JKO:

- The descent inequality (section 2.5) still holds, so $\mathcal F[\nu_k^\tau]\downarrow\inf\mathcal F$, and any narrow cluster point is a stationary point.
- No L²-convexity is available to convert summability of $W_2^2(\nu_{k+1},\nu_k)$ into an $O(1/k)$ rate on $\mathcal F[\nu_k]-\mathcal F[\nu^*]$.
- The natural geometry for $\mathrm{X}_\mu$ is **Fisher–Rao** (since FR geodesics affinely interpolate $\log\nu$), where the functional **is** geodesically convex. This is the analytic counterpart of the section-1 results: the FR analysis is intrinsically better suited to $\mathrm{X}_\mu$.

> **Proposition 2.3 (JKO converges to a stationary point of $\mathrm{X}_\mu$, no rate).** Under the lsc/coercivity hypotheses of Proposition 2.1, the JKO iterates of $\mathrm{X}_\mu$ converge narrowly along subsequences to stationary points. No explicit rate is known without additional assumptions (e.g. uniform log-concavity of $\mu$, which restores displacement convexity for the KL part but not for $\mathrm{X}_\mu$).

### 2.8 Numerical realizations

JKO is the analytic backbone of several practical algorithms. We sketch three relevant to the paper's normalizing-flow setting.

**Entropic-OT JKO.** Replace $W_2^2(\nu,\nu_k)$ with the Sinkhorn divergence $S_\varepsilon^2(\nu,\nu_k)$. Each step becomes a strictly convex problem solvable by Sinkhorn iterations. When $\nu$ is parametrized as a particle cloud $\{X^{(i)}\}$, the JKO step takes $\nu_k$ to a new cloud $\{X^{(i)}_{k+1}\}$ by joint optimization of positions and the transport plan. Numerically stable and autograd-friendly, but entropic regularization biases the gradient flow at $O(\varepsilon)$.

**Neural JKO / JKO-Flow.** Parametrize the transport $T_\theta$ between $\nu_k$ and $\nu_{k+1}$ by an invertible neural network. Each JKO step trains $\theta$ by gradient descent on
$$
\theta_{k+1} \;=\; \arg\min_\theta\Bigl\{\mathcal F\bigl[T_{\theta\#}\nu_k\bigr] \;+\; \tfrac{1}{2\tau}\!\int|x - T_\theta(x)|^2\,d\nu_k(x)\Bigr\}.
$$
After $K$ JKO steps, the composition $T_{\theta_K}\circ\cdots\circ T_{\theta_1}$ is a normalizing flow that approximates the W₂ trajectory. This is the closest analytic abstraction of what the paper does: training a single flow $G_\theta$ over many gradient steps on $\mathrm{KL} + \mathrm{X}_\mu + \mathrm{X}_{\hat\mu}$ is, in this lens, a *single-block* (rather than nested) JKO scheme with adaptive step size.

**Mirror-descent JKO (Fisher–Rao replacement).** Replace $W_2^2$ by $\mathrm{KL}(\nu\Vert\nu_k)$ as the proximal term. The Bregman-proximal scheme
$$
\nu_{k+1} \;=\; \arg\min_\nu\Bigl\{\mathcal F[\nu] + \tfrac{1}{\tau}\mathrm{KL}(\nu\Vert\nu_k)\Bigr\}
$$
admits, for $\mathcal F = \mathrm{KL}(\mu\Vert\cdot)$, the closed form $\nu_{k+1}\propto\nu_k\exp(\tau\mu/\nu_k)$. This is **exactly the FR discretization** of section 1.4 and inherits the exponential rate $e^{-t}$ as $\tau\downarrow 0$. Mirror-descent JKO is the algorithmic embodiment of the route-1/route-2 tradeoff: the FR rate is unconditional, the W₂ rate requires L²-convexity, and one chooses the geometry to match the implementation constraints.

> **Practical takeaway.** The paper's training loop is, in continuous-time idealization, a W₂-JKO scheme on $\mathrm{KL} + \mathrm{X}_\mu + \mathrm{X}_{\hat\mu}$. JKO gives only an $O(1/k)$ rate on the KL part and stationary-point convergence on the $\mathrm{X}$ parts. The Fisher–Rao analysis of section 1 — applicable because importance weights $\mu/\nu$ are already computed for ESS evaluation — recovers an exponential rate without any structural assumption on $\mu$ and matches the empirical observation that adding $\mathrm{X}_\mu$ accelerates convergence without extra tuning.

---

## Summary

| route | geometry | rate | implementability | suitable for |
|---|---|---|---|---|
| Fisher–Rao | reweight-only | $e^{-t}$ on KL, $\mathrm{X}_\mu$ monotone $\downarrow 0$, **unconditional** | needs density access | rate analysis, EM/replicator algorithms |
| JKO | transport-only | $O(1/k)$ on KL, no rate on $\mathrm{X}_\mu$ | yes (Sinkhorn / neural JKO) | numerical backbone of practical training |

Fisher–Rao is the *cleanest* analysis; JKO is the *most numerically familiar*. Both converge to $\nu=\mu$; only Fisher–Rao gives an unconditional exponential rate, and it does so without any structural assumption on $\mu$.
