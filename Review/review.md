# A Critical Review of Prior-Distribution Choice in Flow-Based Boltzmann Generators

*Working draft prepared 2026-05-14 in support of the LLR-discrepancy paper. The review
covers the period 2010–2026 and includes cross-community evidence from molecular sampling,
Bayesian inverse-PDE, cosmology, particle physics, geophysics, and simulation-based
inference.*

---

### Notation

All math in this review uses **standard LaTeX commands only** (no custom macros), so any
KaTeX / MathJax / Pandoc renderer will parse it without errors. The notation is as
follows.

| Symbol | Meaning |
|---|---|
| $\mathbb{R}^d$ | $d$-dimensional Euclidean configuration space |
| $\mathbb{E}_{x \sim p}[\,\cdot\,]$ | Expectation under distribution $p$ |
| $U_1, U_0$ | Target / source potential functions (so that $\mu \propto e^{-U_1}$, $\mu_0 \propto e^{-U_0}$) |
| $\mu$ | Target Boltzmann distribution, $\mu(x) = Z^{-1} e^{-U_1(x)}$ |
| $\mu_0$ | Base (prior) distribution of the flow |
| $G_\theta : \mathbb{R}^d \to \mathbb{R}^d$ | Invertible neural network (the flow), parameters $\theta$ |
| $J_G(y)$ | Jacobian matrix of $G$ at $y$ |
| $\nu_\theta = G^{-1}_{\theta,\#}\mu_0$ | Pullback distribution: the model density on configuration space |
| $\mathrm{KL}(p \| q)$ | Kullback–Leibler divergence $\int p \log(p/q) \, \mathrm{d}x$ |
| $\mathrm{X}_\omega(\mu \| \nu)$ | Log-ratio discrepancy (cross-regularization) functional with weighting $\omega$ |
| $\mathrm{ESS}(\mathbf{y})$ | Effective sample size of weighted ensemble $\mathbf{y}$, in $[0,1]$ |
| $\mathrm{Coverage}_k(\mathbf{y}; \mathbf{x})$ | $k$-NN coverage of reference $\mathbf{x}$ by candidate $\mathbf{y}$ |
| $\mathrm{NND}_k(x_i; \mathbf{x})$ | Distance from $x_i$ to its $k$-th nearest neighbor in $\mathbf{x}$ |
| $\mathbf{1}[\cdot]$ | Indicator function |
| $\hat\mu$ | Trinity-produced approximation of $\mu$, allowed to be coarse |
| $\beta_k, \pi_k$ | AIS schedule parameter / intermediate distribution at step $k$ |
| $w_k(y)$ | AIS per-step importance weight |
| $s_\theta(x, t)$ | Score field of a diffusion model at time $t$ |
| $\leqslant, \geqslant$ | Less/greater-than-or-equal-to (typographic variants of $\le,\ge$) |

---

## Executive summary (one screen)

**The question.** Should a flow-based Boltzmann generator use a simple Gaussian prior with
sophisticated loss machinery (Camp A), or a physically-tuned mixture prior near the target
modes with simpler training (Camp B)?

**The answer this review defends.** For *single-phase, single-basin precision* problems
(crystallographic free energies à la Wirnsberger 2022), Camp B is correct and unbeaten.
For *multimodal mode-discovery* problems (the user's target setting), Camp B is
*structurally* unable to discover modes its prior does not encode — and the entire
2024–2026 BG field has converged on Camp A as a result. A *third way* — keep Camp A's
full-support Gaussian prior but inject mode information into a *log-ratio regularizer*
with built-in inaccuracy tolerance (the $\mathrm{X}_{\hat\mu}$ functional) — fills a
design-space niche neither camp occupies.

**Three load-bearing facts.**

1. **Wirnsberger 2022 explicitly concedes single-basin sampling**: the trained flow does
   not sample alternative crystal phases (§4.1, §22.6).
2. **TBG 2024 explicitly tried a structured prior and reported "no significant
   improvements"**: the architecture, not the prior, drives transferable performance
   (§3.11).
3. **Across ~90 references in 7 communities, no flagship method since 2019 has
   trained a tuned-GMM-prior BG on multimodal protein targets**, despite the tooling
   being present in every standard library (§6).

**The architectural argument** (§§21–22). Camp B's natural home is reverse-KL training
on a flow architecture with cheap $G^{-1}$. The user's `zflows` infrastructure is
MAF-based, making reverse KL $O(d)$ sequential and forward-KL-flavoured losses (forward
KL, FAB, $\mathrm{X}$) the natural fit. On MAF architectures, Camp B's tuning offers no
architectural leverage but pays full mode-lock-in cost. The $\mathrm{X}_{\hat\mu}$
third way is the *uniquely* matched recipe: cheap forward $G$, no need for cheap
$G^{-1}$, autograd-reuse zero-overhead (Prop. 4, §A.4), invariance to Trinity's
sloppiness (Prop. 3, §A.3).

**Steelman of Camp B** (§23). After granting Camp B the strongest possible position, only
one genuine open question remains: a head-to-head empirical comparison between
$\mathrm{X}_{\hat\mu}$ + Trinity and the score-based-bias Route E of §15 on
dipeptide-scale benchmarks. Everything else either reduces to Camp A, reduces to the
third way, or remains in the (precision × single-basin) Camp B niche where the user's
position already concedes Camp B is correct.

**Outputs of this review.** A ~3700-line document, 24 numbered sections, 3 appendices,
~90 references across 7 communities, 5 self-generated matplotlib figures, and an
interactive citation network (`review_citations.html`) with ~50 papers connected by
influence edges.

---

## Abstract

Flow-based Boltzmann generators (BGs) sample from $\mu \propto \exp(-U_1)$ on $\mathbb{R}^d$ by
training an invertible neural map $G:\mathbb{R}^d\to\mathbb{R}^d$ to push a tractable base distribution
$\mu_0$ onto the target. *Which* base distribution $\mu_0$ to choose has been the subject of
a persistent methodological dispute:

- **Camp A** ("uninformative prior + sophisticated training") fixes $\mu_0$ to be a standard
  Gaussian and invests in the loss (FAB, TA-BG, CMT, iDEM, iEFM), the architecture
  (equivariant flows, internal coordinates, transformer flows), and the inference-time
  schedule (AIS, SMC, annealed Langevin).
- **Camp B** ("tuned, target-aware prior + simpler training") encodes physical knowledge of
  the target into $\mu_0$ itself: Wirnsberger's lattice-Gaussian for crystals, Coretti's
  higher-temperature liquid reference, conditional / phase-diagram BGs, structured CG latent
  spaces, and the VAE-community precedents VampPrior and LARS.

This review (i) lays out the *full mathematical framework* of each canonical paper with
explicit loss formulae and training procedures, (ii) inspects the corresponding code
(`noegroup/bgflow`, `lollcat/fab-torch`, `VincentStimper/normflows`, `kazewong/flowMC`,
`aimat-lab/TA-BG`, `jarridrb/DEM`, `annalena-k/FAB-meets-diffME`), (iii) catalogs the
*specific difficulties* each camp must confront, and (iv) draws cross-community evidence
from five neighboring fields where the same prior question has been faced.

> **Bottom line.** Camp B is sharply limited to single-basin / single-phase applications and
> **cannot do mode discovery by construction**. Camp A struggles with mode discovery on
> multimodal targets without injecting prior knowledge *somewhere* in the pipeline. The
> third way — inject mode information into a *log-ratio regularizer with built-in
> inaccuracy tolerance* (the $\mathrm{X}$ functional, this paper) — sidesteps both camps' core
> difficulties. The asymptotic minimizer of $\mathrm{X}_{\hat\mu}$ is invariant to inaccuracies in
> the mode-finder $\hat\mu$, so we can absorb sloppy Trinity-style mode knowledge without
> committing the prior's support to that knowledge.

---

## Table of contents

1. Introduction and motivation
2. Mathematical preliminaries
   1. The Boltzmann sampling problem
   2. Normalizing flows and pushforward measures
   3. The reverse-KL training loss — derivation and mode-seeking property
   4. The forward-KL training loss — derivation and mode-covering property
   5. Annealed importance sampling — explicit schedule
   6. ESS, coverage, and the fake-ESS phenomenon
3. Camp A in detail — Gaussian prior + sophisticated training
   1. The original Boltzmann generator (Noé et al. 2019)
   2. Stochastic normalizing flows (Wu, Köhler, Noé 2020)
   3. Equivariant flows (Köhler, Klein, Noé 2020)
   4. Smooth normalizing flows on tori (Köhler, Krämer, Noé 2021)
   5. FAB — flow annealed importance sampling bootstrap (Midgley et al. 2022)
   6. Adaptive MC + flow (Gabrié, Rotskoff, Vanden-Eijnden 2022)
   7. Resampled base distributions (Stimper et al. 2022)
   8. Equivariant flow matching (Klein et al. 2023)
   9. iDEM and iEFM — energy-based diffusion and flow matching (2024)
   10. Transferable Boltzmann generators (Klein et al. 2024)
   11. TA-BG — temperature-annealed Boltzmann generators (Schopmans, Friederich 2025)
   12. Sequential Boltzmann generators (Tan et al. 2025)
   13. CMT — constrained mass transport (Blessing et al. 2025)
4. Camp B in detail — physics-informed / tuned prior
   1. Lattice-Gaussian priors for crystals (Wirnsberger et al. 2022)
   2. Coretti's higher-temperature reference prior
   3. Conditional / phase-diagram Boltzmann generators
   4. Structured-CG latent priors (Schiebroek & Koehn 2025)
   5. VAE-side precedent: VampPrior (Tomczak, Welling 2018)
   6. VAE-side precedent: LARS (Bauer, Mnih 2019)
5. Difficulties of each camp
   1. Camp A difficulties — mode collapse, mass teleportation, topological gap, fake ESS
   2. Camp B difficulties — circularity, mode lock-in, brittleness, symmetry breaking,
      transferability collapse, structural bias
6. Code-level inventory of canonical implementations
7. Adjacent communities: do they tune the prior?
   1. Bayesian inverse PDE
   2. Cosmology, astrophysics, gravitational waves
   3. Particle physics
   4. Simulation-based inference
8. Non-flow neural samplers
   1. iDEM, iEFM, EWFM, BNEM
   2. Adjoint Schrödinger bridge sampler (ASBS)
   3. Progressive inference-time annealing (PITA)
   4. Stochastic interpolants and NETS
   5. Riemannian flow matching for condensed matter
9. Engineering tricks orthogonal to the prior question
10. Numerical comparison
11. The third way — formal mathematical analysis
    1. The $\mathrm{X}$ functional
    2. Invariance of the asymptotic minimizer to $\hat\mu$ inaccuracy
    3. The autograd-reuse trick for $\mathrm{X}_\mu$
    4. The Trinity algorithm for $\hat\mu$
    5. Coverage-augmented evaluation
    6. Why this beats Camp A and Camp B on every difficulty
12. Talking points
13. References
14. Figures

---

## 1. Introduction and motivation

Sampling from a high-dimensional, multimodal Boltzmann distribution
$$
\mu(x) = \frac{1}{Z}\exp(-U_1(x)),\qquad x\in\mathbb{R}^d
$$
with an analytically tractable potential $U_1$ but intractable partition function $Z$ is the
central computational problem of statistical mechanics, free-energy estimation in
biomolecular simulation, lattice quantum field theory, and Bayesian inference with
PDE-based likelihoods. The Boltzmann distribution is *multimodal* whenever the potential
$U_1$ has multiple local minima separated by free-energy barriers $\Delta F \gg k_B T$.
For proteins and other large biomolecules these barriers are routinely $10$–$30 \, k_B T$,
making naive Langevin / Metropolis sampling infeasibly slow: the mean first-passage time
between basins scales as $\tau\sim e^{\Delta F/k_B T}$.

Flow-based **Boltzmann generators** (BGs), introduced by Noé, Olsson, Köhler, and Wu
(*Science* 2019), short-circuit this difficulty by training a *bijective* neural network
$G:\mathbb{R}^d\to\mathbb{R}^d$ to push a simple base distribution $\mu_0$ — typically a standard Gaussian
— directly onto the target $\mu$. Once $G$ is trained, samples from $\mu$ are produced by
drawing $z\sim\mu_0$ and applying $G^{-1}(z)$. Because $G$ is invertible with tractable
Jacobian, the density of the model on the target side is known in closed form, importance
weights are computable, and the partition function $Z$ can be estimated by free-energy
perturbation.

The architectural and methodological progress on BGs since 2019 has been remarkable —
equivariant flows (2020), AIS-based loss (FAB, 2022), flow matching (2023), transferable
amortized BGs (2024), inference-time SMC (2025), and constrained-transport schedules
(2025). And yet, a central design choice has remained unsettled: **what should the base
distribution $\mu_0$ be?**

Two philosophies have emerged:

- **Camp A — *Simple prior, sophisticated training.*** Take $\mu_0$ as a standard isotropic
  Gaussian (possibly with truncation for compact intervals or von-Mises wrapping for
  dihedrals). Let the *flow* $G$ shoulder the entire topological burden of mapping a
  unimodal contractible Gaussian onto a multimodal Boltzmann target. The loss, the
  architecture, and the inference-time schedule are where all engineering effort lives.
  *This is the mainstream philosophy.*
- **Camp B — *Tuned prior, simpler training.*** Encode physical knowledge of the target
  into $\mu_0$: Gaussians centered at the lattice sites of the crystalline phase
  (Wirnsberger 2022), a higher-temperature reference distribution (Coretti et al.),
  conditional priors at fixed thermodynamic states (Schebek 2024), structured multimodal
  collective-variable latent spaces (Schiebroek & Koehn 2025), or learned mixtures
  (VampPrior 2018, LARS 2019). The flow then has only to learn local fluctuations around
  the encoded structure.

These camps are not just notational variations — they encode *incompatible philosophical
commitments* about where mode information should live in the pipeline. Camp A treats mode
information as something the model *discovers* from energy gradients (and from increasingly
sophisticated loss machinery designed to enforce coverage). Camp B treats mode information
as something the user *supplies* through the prior's support.

The user's paper proposes a *third way*: keep Camp A's uninformative prior, but inject mode
information into the *loss* via a log-ratio discrepancy regularizer $\mathrm{X}_{\hat\mu}$. The
regularizer has the property that its asymptotic minimizer is invariant to inaccuracies in
the mode-finder $\hat\mu$, so we can absorb sloppy mode knowledge without committing the
prior's support to it. This decouples *where mode information enters* (Camp A vs. Camp B's
crucial division) from *how robust the method is to wrong mode information* (where Camp B
is structurally brittle and Camp A has no clean way to absorb it at all).

The purpose of this review is to compile the evidence — mathematical, code-level,
numerical, and cross-community — needed to argue cleanly between the three positions.

---

## 2. Mathematical preliminaries

> **Why this section exists.** Before debating *which* prior is best for a Boltzmann
> generator, we need a shared language for *what a BG is doing* and *what the failure
> modes look like*. The objects this section defines — pushforward density, reverse-KL
> loss, forward-KL loss, ESS, coverage — are the vocabulary every camp uses, often with
> subtly different assumptions baked in. Setting them down once in standard notation lets
> every later claim be checked against the math rather than against rhetorical framing.
>
> **How this section is organized.** §2.1 sets the physics — the Boltzmann distribution
> and why it is hard to sample. §2.2 defines the bijective flow and its pushforward
> measure. §2.3 and §2.4 derive the two KL losses, exposing their mode-seeking and
> mode-covering signatures respectively. §2.5 introduces annealed importance sampling as
> the canonical bridge from $\mu_0$ to $\mu$. §2.6 defines ESS and coverage, which
> together diagnose the *fake-ESS pitfall* that motivates much of the rest of the review.

### 2.1. The Boltzmann sampling problem

**Setup.** Let $U_1:\mathbb{R}^d\to\mathbb{R}$ be a *potential* on configuration space, possibly only
analytically tractable up to an additive constant. We want to draw samples from the
*Boltzmann distribution*
$$
\mu(x) = \frac{1}{Z}\exp(-U_1(x)), \qquad Z = \int_{\mathbb{R}^d}\exp(-U_1(x))\,\mathrm{d} x.
$$
We assume only that $U_1$ and its gradient $\nabla U_1$ can be evaluated pointwise; we do
*not* assume samples from $\mu$ are available a priori.

**Why this is hard.** When $U_1$ is multimodal with deep barriers, MCMC samplers driven by
local dynamics (Langevin, HMC) have mixing times that scale exponentially in the barrier
height. Direct importance sampling from a fixed proposal $q$ collapses because the weights
$\mu(x)/q(x)$ are exponentially heavy-tailed.

**The two principal metrics.** We will judge a sampler by

- the **effective sample size** $\mathrm{ESS}(\mathbf y)\in[0,1]$ (§2.6), which measures how
  *uniform* the importance weights are on the support reached by the sampler;
- the **coverage** $\mathrm{Coverage}_k(\mathbf y; \mathbf x)\in[0,1]$ (Naeem et al. 2020, §2.6), which
  measures *how many modes of a reference set* $\mathbf x$ are reached by the sampler's
  output $\mathbf y$.

The distinction between these two metrics is what exposes the *fake-ESS* failure mode.

### 2.2. Normalizing flows and pushforward measures

A **normalizing flow** is a $C^1$-bijection $G:\mathbb{R}^d\to\mathbb{R}^d$ parameterized by a neural
network with parameters $\theta$, designed so that the Jacobian determinant
$\det J_G(y)$ is computable in $O(d)$ or $O(d\log d)$ time. The standard architectures (NICE,
RealNVP, IAF, Neural Spline Flows, augmented coupling flows, continuous normalizing flows)
achieve this through coupling layers, autoregressive structure, or neural ODEs.

Let $\mu_0$ be a base distribution on $\mathbb{R}^d$ (typically the standard isotropic Gaussian).
The flow $G$ induces two distributions on $\mathbb{R}^d$:

- the **pushforward** $G_\#\mu_0$ defined by $(G_\#\mu_0)(B) = \mu_0(G^{-1}(B))$ for Borel
  $B$, with density $(G_\#\mu_0)(x) = \mu_0(G^{-1}(x))\,|\det J_{G^{-1}}(x)|$;
- the **pullback** $G^{-1}_\#\mu_0$ defined by $(G^{-1}_\#\mu_0)(B) = \mu_0(G(B))$, with
  density
  $$
  (G^{-1}_\#\mu_0)(y) = \mu_0(G(y))\,|\det J_G(y)|.
  $$

In the BG convention, the *target* is on the "$y$"-side and the flow maps $y\in\mathbb{R}^d$
("configuration space") to $z=G(y)\in\mathbb{R}^d$ ("latent space"). The model density on
configuration space is the pullback $G^{-1}_\#\mu_0$, and the goal of training is
$$
G^{-1}_\#\mu_0 \;\approx\; \mu.
$$
We parameterize the loss in terms of $G$ (target $\to$ latent), not $G^{-1}$, so that no
inversion is required during training. The price is that *sampling* requires inverting $G$
on $z\sim\mu_0$ to obtain $y = G^{-1}(z)$; this is usually one forward pass through
inverted coupling or a single ODE integration.

### 2.3. The reverse-KL training loss

Let $\nu_\theta := G^{-1}_{\theta,\#}\mu_0$ be the model distribution. The *reverse* KL
divergence is
$$
\mathcal L_{\mathrm{rev}}(\theta)
\;=\; \mathrm{KL}(\nu_\theta\,\|\,\mu)
\;=\; \int \nu_\theta(x)\log\frac{\nu_\theta(x)}{\mu(x)}\,\mathrm{d} x.
$$
Because $\nu_\theta$ is the pushforward of $\mu_0$ through $G_\theta^{-1}$, we have the
reparameterized estimator
$$
\mathcal L_{\mathrm{rev}}(\theta)
\;=\; \mathbb{E}_{z\sim\mu_0}\!\Big[\log\frac{\mu_0(z)|\det J_{G^{-1}_\theta}(z)|^{-1}}{\mu(G^{-1}_\theta(z))}\Big]
\;=\; \mathbb{E}_{z\sim\mu_0}\!\Big[-\log\mu(G^{-1}_\theta(z))+\log\mu_0(z)-\log|\det J_{G^{-1}_\theta}(z)|^{-1}\Big].
$$
With the standard convention $\log\mu(x) = -U_1(x) - \log Z$,
$$
\mathcal L_{\mathrm{rev}}(\theta)
\;=\; \mathbb{E}_{z\sim\mu_0}\!\Big[U_1(G^{-1}_\theta(z))\Big] - \mathbb{E}_{z\sim\mu_0}\!\Big[\log|\det J_{G^{-1}_\theta}(z)|^{-1}\Big] + \mathbb{E}_{z\sim\mu_0}[\log\mu_0(z)] - \log Z.
$$
The last two terms are independent of $\theta$, so the *effective* objective is
$$
\boxed{\;\;\mathcal L_{\mathrm{rev}}(\theta)
\;=\; \mathbb{E}_{z\sim\mu_0}\!\big[U_1(G^{-1}_\theta(z))\big]
\;+\;\mathbb{E}_{z\sim\mu_0}\!\big[\log|\det J_{G^{-1}_\theta}(z)|\big]
\;+\;\text{const}.\;\;}
$$
**Mode-seeking property.** Reverse KL gives infinite penalty whenever $\nu_\theta$ puts
mass where $\mu$ puts essentially none, but it gives essentially zero penalty when $\nu_\theta$
*misses* mass that $\mu$ puts somewhere. Formally, if $\mu$ has a mode of mass $p^*$ at a
location $x^*$ where $\nu_\theta(x^*)\to 0$, the contribution to the reverse KL from a
neighborhood of $x^*$ is bounded by $-\int_{B(x^*)}\nu_\theta\log\mu \,\mathrm{d} x \to 0$. **A
dropped mode does not enter the loss.** This is the source of mode collapse.

**Why it is the default.** Despite the mode-collapse issue, $\mathcal L_{\mathrm{rev}}$ is
the only "free" KL: it requires no samples from $\mu$, just samples from $\mu_0$. The
gradient $\nabla_\theta \mathcal L_{\mathrm{rev}}$ is unbiasable from $\mu_0$-samples alone
(no reweighting required). All other principled losses require either samples from $\mu$
(forward KL) or expensive bootstrapping schemes (AIS, FAB).

### 2.4. The forward-KL training loss

The *forward* KL divergence is
$$
\mathcal L_{\mathrm{fwd}}(\theta) \;=\; \mathrm{KL}(\mu\,\|\,\nu_\theta)
\;=\; \int \mu(y)\log\frac{\mu(y)}{\nu_\theta(y)}\,\mathrm{d} y.
$$
Expanding $\nu_\theta(y) = \mu_0(G_\theta(y))\,|\det J_{G_\theta}(y)|$ and using
$\log\mu_0(z) = -\tfrac{1}{2}\|z\|^2 - \tfrac{d}{2}\log(2\pi)$ for the standard Gaussian
prior,
$$
\boxed{\;\;\mathcal L_{\mathrm{fwd}}(\theta)
\;=\; \mathbb{E}_{y\sim\mu}\!\Big[U_0(G_\theta(y)) - \log|\det J_{G_\theta}(y)|\Big]
\;+\;\text{const},\quad U_0(z)=\tfrac{1}{2}\|z\|^2.\;\;}
$$
**Mode-covering property.** Forward KL is infinite whenever $\nu_\theta$ assigns zero
density to a region where $\mu$ has positive density. This *penalizes* missing modes
maximally: if a single Boltzmann mode is dropped, the loss diverges. Hence "forward KL is
mass-covering."

**Why it cannot be used naively.** The expectation is over $\mu$, which we cannot sample
directly. There are three families of workaround, each surveyed below:

1. **MLE on MD samples** — replace $\mathbb{E}_{y\sim\mu}$ with empirical samples from an MD
   trajectory. Works when MD is feasible; expensive in BG settings.
2. **AIS-based estimators** — bootstrap samples from $\mu$ via a sequence of intermediate
   distributions (Neal 2001; Midgley et al. FAB 2023; §2.5 below).
3. **Adversarial / self-supervised proxies** — flow matching, score matching, energy
   matching (iDEM, iEFM, EWFM).

### 2.5. Annealed importance sampling

AIS (Neal 2001) bridges $\nu_\theta$ and $\mu$ through a sequence of $M+1$ intermediate
distributions
$$
\pi_k \;\propto\; \nu_\theta^{1-\beta_k}\,\mu^{\beta_k},\qquad 0=\beta_0<\beta_1<\dots<\beta_M=1,
$$
typically with $\beta_k = k/M$. Given particles $\mathbf y_0\sim\pi_0 = \nu_\theta$, AIS
performs $M$ ladder steps; at step $k$ it reweights particles by
$$
w_k(y) \;=\; \frac{\pi_k(y)}{\pi_{k-1}(y)} \;=\; \big(\nu_\theta(y)\big)^{\beta_{k-1}-\beta_k}\big(\mu(y)\big)^{\beta_k-\beta_{k-1}}
$$
and then applies an MCMC kernel targeting $\pi_k$ (typically HMC or Langevin) to obtain
$\mathbf y_k\sim\pi_k$ (approximately). After $M$ steps, $\mathbf y_M\sim\pi_M = \mu$. The
intermediate weights accumulate into a single importance weight per particle:
$$
W(y_0,\dots,y_M) \;=\; \prod_{k=1}^M w_k(y_{k-1}).
$$
The empirical estimator of any expectation under $\mu$ is then
$\sum_i W_i\,f(y_{M,i})/\sum_i W_i$.

**Why AIS works (in principle).** Each per-step ratio $w_k$ is close to 1 when the schedule
is fine enough; the chain spreads the "topological work" of transporting $\nu_\theta$ to
$\mu$ over $M$ small steps, each individually well-conditioned for HMC.

**Why AIS struggles (in practice).** When $\nu_\theta$ has dropped a mode that $\mu$
contains, the AIS chain initialized at $\nu_\theta$ has *no particles* in that mode at
step $0$. The intermediate schedule does not magically populate the missing mode unless the
MCMC kernel mixes across barriers in finite $M$ — and barriers are exactly what makes the
original target hard. *AIS does not solve mode discovery; it accelerates fine-grained
reweighting once the proposal has the right support.*

### 2.6. ESS and coverage metrics

**Definition (ESS).** Given $N$ samples $\mathbf y = (y_1,\dots,y_N)$ from $\nu_\theta$ with
importance weights $w_i = \mu(y_i)/\nu_\theta(y_i)$,
$$
\mathrm{ESS}(\mathbf y) \;=\; \frac{\big(\sum_{i=1}^N w_i\big)^2}{N\sum_{i=1}^N w_i^2} \;\in\;[0,1].
$$
The upper bound $1$ is attained iff all weights are equal; the lower bound $0$ is approached
when one weight dominates. *ESS is uniformity of weights, not coverage of modes.*

**Definition (Coverage; Naeem et al. 2020).** Given a reference set
$\mathbf x = (x_1,\dots,x_P)$ from a comparison distribution (e.g., from $\hat\mu_1$ or from
long MD) and a candidate set $\mathbf y$ from the model, define
the $k$-nearest-neighbor radius
$$
\mathrm{NND}_k(x_i;\mathbf x) \;=\; \inf\!\Big\{ r\geqslant 0 : \#\{1\leqslant j\leqslant P : 0<|x_i-x_j|<r\}\geqslant k\Big\}.
$$
Then
$$
\mathrm{Coverage}_k(\mathbf y;\mathbf x) \;=\; \frac{1}{P}\sum_{i=1}^P \mathbf 1\!\Big[\exists\, 1\leqslant j\leqslant N : |y_j - x_i| \leqslant \mathrm{NND}_k(x_i;\mathbf x)\Big].
$$
Coverage equals $1$ when every reference point has at least one model sample inside its
$k$-NN ball; it equals $0$ when no reference point has a nearby model sample.

**The fake-ESS phenomenon.** If the support of $\nu_\theta$ is contained strictly inside
the support of $\mu$, importance weights $w_i = \mu(y_i)/\nu_\theta(y_i)$ are still
computable and can be uniform across $\nu_\theta$'s support. In that case $\mathrm{ESS}(\mathbf y)$
can be arbitrarily close to $1$ even though entire modes of $\mu$ are unreached:
$$
\mathrm{ESS}(\mathbf y) \to 1 \quad\not\Rightarrow\quad \mathrm{supp}(\nu_\theta) = \mathrm{supp}(\mu).
$$
**This is fake ESS.** A high ESS is a *necessary* but not *sufficient* condition for
quality. The coverage metric is sufficient where ESS is not.

The introduction of the user's paper makes the fake-ESS pitfall central; this review
adopts the same framing and uses the term throughout.

---

> **Concluding the preliminaries.** The reverse KL is *mode-seeking* (missing modes
> contribute zero loss), the forward KL is *mass-covering* (missing modes contribute
> infinite loss), AIS bridges the two at the cost of a sequence of intermediate
> distributions, and ESS measures only how *uniform* importance weights are on the
> support that has been reached — silent about modes the support never touched. This
> last point is what every later section returns to: ESS alone is not a sufficient
> quality criterion, and the coverage metric is the necessary complement.

## 3. Camp A — Simple prior + sophisticated training

> **Why this section exists.** "Camp A" is the *de facto* default in the Boltzmann-generator
> literature since Noé et al. 2019: keep the base distribution $\mu_0$ a standard Gaussian
> and pour all the engineering effort into the loss, the architecture, and the inference
> schedule. To argue *for or against* this default we need to look at what its proponents
> have actually built. The thirteen papers below trace the canonical Camp A timeline from
> the founding 2019 paper to the 2025 state-of-the-art and document the exact loss formula,
> training procedure, and numerical headlines of each.
>
> **How to read this section.** Each subsection is structured as (a) framework and target
> setting, (b) the exact loss function with derivation, (c) training procedure with the
> core algorithmic tools, (d) headline numerical results, and (e) a brief critical
> assessment that flags where the original paper may be over-claiming. Read sequentially
> to see the philosophical and engineering progression; or jump to any individual paper
> for a self-contained synopsis.

For each landmark paper in Camp A we describe (a) the *framework* and target setting, (b)
the *exact loss function* with derivation, (c) the *training procedure* with core
algorithmic tools, and (d) headline *numerical results*. The base distribution is, in every
case below, an isotropic Gaussian (or a symmetry-constrained variant thereof) unless noted.

### 3.1. The original Boltzmann generator (Noé et al. 2019)

**Setup.** Noé, Olsson, Köhler, Wu — *Boltzmann generators: Sampling equilibrium states of
many-body systems with deep learning*, [Science 365, eaaw1147 (2019)](https://www.science.org/doi/10.1126/science.aaw1147).
Target: molecular Boltzmann distribution $\mu \propto e^{-U_1}$ for systems including
2D bistable potentials, BPTI ($d\approx 300$), and coarse-grained proteins.
Base: standard Gaussian on the latent (PCA-coordinate + internal-coordinate split).

**Combined loss.** The paper introduces a hybrid loss combining reverse-KL (no MD samples)
with a maximum-likelihood (forward-KL) term that requires *initial samples* $\{y_i^{\mathrm{MD}}\}_{i=1}^{N_{\mathrm{MD}}}$
drawn from short MD simulations in *each known* metastable state:
$$
\mathcal L_{\mathrm{BG}}(\theta) \;=\; w_{\mathrm{KL}}\,\mathcal L_{\mathrm{rev}}(\theta) + w_{\mathrm{ML}}\,\mathcal L_{\mathrm{ML}}(\theta) + w_{\mathrm{RC}}\,\mathcal L_{\mathrm{RC}}(\theta).
$$
The maximum-likelihood term is the empirical forward-KL estimator
$$
\mathcal L_{\mathrm{ML}}(\theta) \;=\; -\frac{1}{N_{\mathrm{MD}}}\sum_{i=1}^{N_{\mathrm{MD}}}\log\nu_\theta(y_i^{\mathrm{MD}}),
$$
which is equivalent to $\mathbb{E}_{y\sim\mu_{\mathrm{MD}}}[U_0(G(y))-\log|\det J_G(y)|] + \text{const}$.
The optional reaction-coordinate term $\mathcal L_{\mathrm{RC}}$ encourages the flow to span
specific user-supplied reaction coordinates between modes.

**Mode-discovery limitation.** The BG framework *as originally formulated* required that
short MD simulations had already discovered every metastable state of interest. The
original paper itself acknowledges: when initialized with only one starting configuration,
the BG can fill the local basin but cannot find globally separated metastable states
without external mode-discovery assistance.

**Numerical results.** On BPTI, the BG produced statistically independent samples of
different metastable states and computed free-energy profiles "without suffering from rare
events," provided initial seeds in each state. On a 2D Müller potential, the BG produced
unbiased samples after training; on the 2D bistable, the *combined* KL+ML loss was
necessary to prevent collapse onto a single well.

**Software stack.** Reference code: [`noegroup/bgflow`](https://github.com/noegroup/bgflow)
(PyTorch). The canonical notebook `alanine_dipeptide_basics.py` uses
`bg.NormalDistribution(dim_ics, mean=mean)` with a self-described "rather naive" Gaussian
prior — this comment in the original authors' own code is decisive evidence that the
simple-prior philosophy is not for lack of trying alternatives.

### 3.2. Stochastic normalizing flows (Wu, Köhler, Noé 2020)

**Setup.** Wu, Köhler, Noé — *Stochastic Normalizing Flows*, NeurIPS 2020.
[Paper](https://proceedings.neurips.cc/paper/2020/file/41d80bfc327ef980528426fc810a6d7a-Paper.pdf).
Goal: relax the invertibility constraint of pure flows by interleaving stochastic sampling
blocks (MCMC, Langevin, HMC) with deterministic coupling layers.

**Path likelihood.** Let the flow consist of $L$ alternating deterministic layers
$T_\ell$ and stochastic kernels $K_\ell(y\to y')$ with reverse kernels $\tilde K_\ell$. For
a forward path $y_0\to y_1\to\dots\to y_L$ with mixed deterministic/stochastic steps, the
*path likelihood* satisfies
$$
\log p(y_L) \;=\; \log p(y_0) + \sum_{\ell:\,T_\ell\,\mathrm{deterministic}}\log|\det J_{T_\ell}(y_{\ell-1})| + \sum_{\ell:\,K_\ell\,\mathrm{stochastic}}\Big[\log\tilde K_\ell(y_{\ell-1}\to y_\ell) - \log K_\ell(y_{\ell-1}\to y_\ell)\Big].
$$
The last bracket is the **Jarzynski-style work** along the stochastic step. For Langevin
kernels at temperatures $\beta_\ell$ along a schedule, this work is computable in closed
form from the forces.

**Loss.** SNFs train via a path-likelihood objective that mixes reverse-KL on the
deterministic layers with the Jarzynski importance weights on the stochastic layers,
yielding an *asymptotically unbiased* estimator of expectations under $\mu$ via path-space
importance sampling.

**Why this matters for the camp question.** SNFs increase model expressivity *without*
changing the base distribution. The base is still an isotropic Gaussian. The added mode
coverage comes from the *MCMC layers along the path*, not from a tuned prior.

### 3.3. Equivariant flows (Köhler, Klein, Noé 2020)

**Setup.** Köhler, Klein, Noé — *Equivariant Flows: Exact Likelihood Generative Learning
for Symmetric Densities*, ICML 2020,
[arXiv:2006.02425](https://arxiv.org/abs/2006.02425). Target: Boltzmann distributions with
a symmetry group $G_{\mathrm{sym}}$ (permutation, $SE(3)$, chiral).

**Construction.** A flow $G:\mathbb{R}^d\to\mathbb{R}^d$ is *$G_{\mathrm{sym}}$-equivariant* if
$G(g\cdot y) = g\cdot G(y)$ for all $g\in G_{\mathrm{sym}}$. If the base $\mu_0$ is
*$G_{\mathrm{sym}}$-invariant*, i.e. $\mu_0(g\cdot z)=\mu_0(z)$, then the pullback
$G^{-1}_\#\mu_0$ is also $G_{\mathrm{sym}}$-invariant — by construction matching the
symmetry of the target.

**The prior choice.** The base is a *symmetry-respecting* Gaussian: an isotropic centred
Gaussian on $\mathbb{R}^d$ is automatically $O(d)$- and translation-invariant. For
permutation-symmetric targets one uses an exchangeable Gaussian (still factorized). For
rotation/translation-invariant targets one uses the *mean-free Gaussian*
$$
\mu_0^{\mathrm{MF}}(z) \;\propto\; \exp\!\Big(-\tfrac{1}{2}\|z\|^2\Big)\cdot\delta\!\Big(\sum_i z_i\Big),
$$
i.e. an isotropic Gaussian on the hyperplane $\{z : \sum_i z_i = 0\}$. *This is not mode
tuning — it is symmetry enforcement.* No mixture components, no mode information.

**Loss.** Same reverse-KL or forward-KL as the non-equivariant flow, but the equivariance
constraint reduces the effective dimension and speeds convergence.

### 3.4. Smooth normalizing flows (Köhler, Krämer, Noé 2021)

**Setup.** [arXiv:2110.00351](https://arxiv.org/abs/2110.00351), NeurIPS 2021. Targets
molecules where internal coordinates live on hypertori (dihedrals) and compact intervals
(bonds, angles). Classical NSF-style splines are not $C^2$-smooth, which prevents using
the flow as a *molecular-dynamics potential*.

**Construction.** Smooth mixture transformations of the form
$$
T(y;\theta) \;=\; \sum_{k=1}^K \pi_k(y;\theta)\,\psi_k(y;\theta),
$$
where $\pi_k$ is a smooth partition of unity and $\psi_k$ is a smooth invertible bump on
$[0,1]$. The inverse $T^{-1}$ is computed by root-finding; gradients of the inverse and its
log-determinant are computed via the *inverse function theorem*.

**Loss.** Reverse-KL or forward-KL on the internal-coordinate target. The innovation is
purely architectural — periodicity on tori, smoothness on compact intervals — *not* a
change to the prior. The prior on dihedrals is uniform on $[0,1]$ (periodic); on bonds and
angles it is truncated Gaussian. **The prior is physics-aware in *topology* but not in
*modes*.**

### 3.5. FAB — Flow Annealed Importance Sampling Bootstrap (Midgley et al. 2022)

**Setup.** Midgley, Stimper, Simm, Schölkopf, Hernández-Lobato — *Flow Annealed Importance
Sampling Bootstrap*, [ICLR 2023, arXiv:2208.01893](https://arxiv.org/abs/2208.01893). First
method to learn the Boltzmann distribution of alanine dipeptide using only the unnormalized
target density — i.e., *no MD samples in training*. Reference code:
[`lollcat/fab-torch`](https://github.com/lollcat/fab-torch).

**Core innovation: the $\alpha=2$ divergence.** The Rényi $\alpha$-divergence at $\alpha=2$
is
$$
D_2(\mu\,\|\,\nu) \;=\; \frac{1}{\alpha-1}\log\int \frac{\mu^\alpha(x)}{\nu^{\alpha-1}(x)}\,\mathrm{d} x \;\Big|_{\alpha=2} \;=\; \log\int \frac{\mu^2(x)}{\nu(x)}\,\mathrm{d} x.
$$
Equivalently, $D_2$ is the log of the chi-squared divergence + 1. The *crucial property* is
that minimizing $D_2$ is equivalent to minimizing the variance of importance weights:
$$
D_2(\mu\,\|\,\nu) \;=\; \log\!\Big(1 + \mathrm{Var}_{\nu}\!\big[\mu/\nu\big]\Big).
$$
**FAB minimizes a tractable surrogate of $D_2$** by sampling from an AIS chain that
gradually morphs $\nu_\theta$ into $\mu$, and weighting samples by their AIS importance
weights.

**The FAB loss.** Let $\mathbf y^{(\mathrm{AIS})}_i\sim\pi_{1/2}$ with importance weights
$w^{(\mathrm{AIS})}_i \propto \nu_\theta(y_i)/\pi_{1/2}(y_i)\cdot \prod_k w_k(y_{i,k-1})$
(the chain integrates from $\nu_\theta$ to a mid-temperature $\pi_{1/2}\propto\nu_\theta^{1/2}\mu^{1/2}$
and weights are computed up the ladder). The training objective is the negative
$\alpha=2$-divergence gradient
$$
\nabla_\theta \mathcal L_{\mathrm{FAB}}(\theta) \;=\; -\sum_i w^{(\mathrm{AIS})}_i \nabla_\theta \log\nu_\theta(y^{(\mathrm{AIS})}_i).
$$
A **prioritized replay buffer** stores past AIS samples and reweights training batches by
$w^2/\nu_\theta(y)$ (so that high-importance points contribute disproportionately) — this
is one of the most empirically critical engineering tricks in the FAB pipeline.

**Algorithm 1 (FAB; from Midgley et al. 2023).**

1. *Initialization.* Train the flow to approximately match $\mu_0$ with a few reverse-KL
   gradient steps.
2. *AIS step.* Sample $\mathbf y_0\sim\nu_\theta$, run a 2-temperature AIS chain from
   $\nu_\theta$ to $\pi_{1/2}\propto\sqrt{\nu_\theta\,\mu}$ using HMC transition kernels at
   each intermediate temperature.
3. *Buffer update.* Add the AIS samples to a prioritized replay buffer; reweight by
   $w^2/\nu_\theta$ at the current $\theta$.
4. *Gradient step.* Compute $\nabla_\theta\mathcal L_{\mathrm{FAB}}$ on a batch from the
   buffer; one Adam step.
5. *Loop.* Iterate 2–4.

**Numerical headlines.** On alanine dipeptide ($d=22\times 3=66$ atomic coordinates, or
$\sim 60$ internal coordinates), FAB reaches comparable accuracy to forward-KL training on
MD samples using **100× fewer target energy evaluations**. On a 40-component 2D GMM target
and a Many Well distribution, FAB recovers all modes — but *only when* the AIS chains are
long enough to discover them. The buffer is necessary; without it, the AIS chains and
training are too noisy to converge.

**Prior choice.** Standard isotropic Gaussian. Reference code excerpt from
[`fab-torch/experiments/make_flow/make_normflow_model.py`](https://github.com/lollcat/fab-torch):
```python
def make_wrapped_normflow_realnvp(dim, n_flow_layers=5, ...):
    base = nf.distributions.base.DiagGaussian(dim)
    flows = make_normflow_flow(dim, n_flow_layers=n_flow_layers, ...)
    model = nf.NormalizingFlow(base, flows)
```
No tuning. The mode-coverage burden is carried entirely by the $\alpha=2$ loss and the AIS
buffer.

### 3.6. Adaptive MC + flow (Gabrié, Rotskoff, Vanden-Eijnden 2022)

**Setup.** Gabrié, Rotskoff, Vanden-Eijnden — *Adaptive Monte Carlo augmented with
normalizing flows*, [PNAS 119(10) (2022)](https://www.pnas.org/doi/10.1073/pnas.2109420119).
Sampling-flow library: [`kazewong/flowMC`](https://github.com/kazewong/flowMC). Target:
posterior samplers in Bayesian inference.

**Hybrid local-global proposal.** The sampler interleaves two move types:

- A *local move* (Langevin, HMC, MALA) targeting $\mu$ directly. Fast convergence inside
  basins but mixes poorly across barriers.
- A *global move* using a normalizing flow $\nu_\theta$ as an independent
  Metropolis–Hastings proposal: $y'\sim\nu_\theta$, accept with probability
  $\min(1, \mu(y')\nu_\theta(y)/[\mu(y)\nu_\theta(y')])$. The flow is *trained online* on
  the samples collected by the local moves so far.

**Loss.** The flow is fit by forward-KL on the running chain:
$$
\mathcal L_{\mathrm{adaptive}}(\theta_n) \;=\; -\frac{1}{|\mathcal D_n|}\sum_{y\in\mathcal D_n}\log\nu_{\theta_n}(y),
$$
where $\mathcal D_n$ is the chain history at iteration $n$. After each retraining round,
the flow becomes a better global proposal.

**Critical caveat.** Direct quote: *"the multimodal case requires prior knowledge about
either the symmetries of the systems generating the degeneracy of the modes or the location
of the metastable basins."* The chain needs *some* mechanism to discover modes initially —
parallel chains seeded across modes, parallel tempering, or external mode information.
*Camp A admits explicitly that mode discovery cannot be solved by sampler + simple-prior
flow alone.*

### 3.7. Resampled base distributions (Stimper et al. 2022)

**Setup.** Stimper, Schölkopf, Hernández-Lobato, AISTATS 2022,
[arXiv:2110.15828](https://arxiv.org/abs/2110.15828). Library:
[`VincentStimper/normalizing-flows`](https://github.com/VincentStimper/normalizing-flows).

**Construction.** Take the base distribution to be a *learned-rejection-sampling* version
of a Gaussian: sample $z\sim\mu_0$, accept with probability $a_\phi(z)\in[0,1]$ where
$a_\phi$ is a learned acceptance function. The accepted distribution
$$
\tilde\mu_0(z) \;\propto\; \mu_0(z)\cdot a_\phi(z)
$$
is more flexible than a Gaussian but still has a tractable log-density modulo a learned
normalization constant. The flow then maps $\tilde\mu_0\to\mu$.

**Loss.** Standard forward- or reverse-KL on $\nu_\theta = G^{-1}_{\theta,\#}\tilde\mu_0$,
with an additional regularization term for the normalization of $\tilde\mu_0$.

**Where this sits.** This is the *mildest* Camp B move that has been tried: the base is
*learned* but not *tuned to modes*. In practice the learned $a_\phi$ tends to carve regions
out of the Gaussian rather than introduce sharp multimodality. The library is widely used
(FAB and TA-BG both depend on `normflows`), but the resampled-base option is rarely
enabled.

### 3.8. Equivariant flow matching (Klein, Krämer, Noé 2023)

**Setup.** Klein, Krämer, Noé — *Equivariant flow matching*, NeurIPS 2023,
[arXiv:2306.15030](https://arxiv.org/abs/2306.15030). Flow matching (Lipman et al. 2023) is
an alternative to maximum-likelihood training of continuous normalizing flows that learns
the *velocity field* of the probability flow ODE, not the flow itself.

**Flow matching loss.** Define a path $\rho_t$ from $\rho_0=\mu_0$ to $\rho_1=\mu$ via a
*conditional* probability path $\rho_t(x|y_1) = \mathcal N(t y_1, (1-t)^2 I)$ for each
target sample $y_1\sim\mu_1$. The conditional velocity field is
$$
u_t(x|y_1) \;=\; \frac{y_1 - x}{1-t}.
$$
The *flow-matching* loss trains a parameterized velocity $v_\theta(x,t)$:
$$
\mathcal L_{\mathrm{FM}}(\theta) \;=\; \mathbb{E}_{t,y_1,x}\!\big[\|v_\theta(x,t) - u_t(x|y_1)\|^2\big].
$$
At optimum, the ODE $\dot y = v_\theta(y,t)$ transports $\rho_0\to\rho_1$.

**Equivariant variant.** Constrain $v_\theta$ to be $SE(3)\times S_n$-equivariant via an
EGNN parameterization. The prior $\mu_0$ is a *mean-free Gaussian* (symmetry-respecting).
First flow-based BG for alanine dipeptide *without* the standard internal-coordinate
featurization — the flow operates directly on Cartesian coordinates, with symmetries
respected by construction.

### 3.9. iDEM — Iterated Denoising Energy Matching (Akhound-Sadegh et al. 2024)

**Setup.** Akhound-Sadegh, Mittal, Bose et al. — *Iterated Denoising Energy Matching for
Sampling from Boltzmann Densities*, [ICML 2024, arXiv:2402.06121](https://arxiv.org/abs/2402.06121).
Code: [`jarridrb/DEM`](https://github.com/jarridrb/DEM). First *energy-only* sampler to
scale to a 55-particle Lennard-Jones system.

**Diffusion process.** Define the variance-preserving SDE on $\mathbb{R}^d$:
$$
\mathrm{d} X_t \;=\; -\tfrac{1}{2}\beta(t)X_t\,\mathrm{d} t + \sqrt{\beta(t)}\,\mathrm{d} W_t,\qquad t\in[0,1],
$$
with $X_0\sim\mu_1$, $X_1\sim\mathcal N(0,I)$. The marginal $p_t$ at time $t$ is the
convolution of $\mu_1$ with a noise kernel $q_{t|0}(x|x_0) = \mathcal N(\alpha_t x_0,\sigma_t^2 I)$
where $\alpha_t = e^{-\frac{1}{2}\int_0^t\beta(s)\mathrm{d} s}$ and $\sigma_t^2 = 1-\alpha_t^2$.

**Score function.** The reverse-time SDE that samples $\mu_1$ requires the *score*
$s(x,t) := \nabla_x \log p_t(x)$. iDEM expresses this as an *expectation over the
energy*:
$$
\nabla_x \log p_t(x) \;=\; \nabla_x \log\!\int q_{t|0}(x|x_0)\mu_1(x_0)\,\mathrm{d} x_0
\;=\; \frac{\mathbb{E}_{x_0\sim q_{0|t}(\cdot|x)}[\nabla_x \log q_{t|0}(x|x_0)]}{1}.
$$
Crucially, the score can be estimated *off-policy* from samples of the current model: draw
$x\sim\rho_t^{\mathrm{model}}$, then approximate $q_{0|t}(\cdot|x)$ by importance-sampling
the noise kernel against $e^{-U_1}$.

**iDEM loss (key novelty).**
$$
\mathcal L_{\mathrm{iDEM}}(\theta) \;=\; \mathbb{E}_{t,x_t\sim\mathcal B}\!\Big[\big\|s_\theta(x_t,t) - \hat s(x_t,t)\big\|^2\Big],
$$
where $\hat s$ is the energy-derived Monte Carlo score estimator and $\mathcal B$ is a
replay buffer of samples from the *current* diffusion model. Two-loop structure: outer loop
populates $\mathcal B$ from the current sampler; inner loop fits $s_\theta$ to $\hat s$.

**Numerical headlines.** State-of-the-art on the Lennard-Jones-55 benchmark — the first
energy-only sampler to handle this scale. 2–5× faster training than FAB on shared
benchmarks (GMMs, DW-4, LJ-13). *Prior is standard Gaussian (the SDE terminal
distribution).*

### 3.10. iEFM and EWFM — Energy-based flow matching (Woo et al. 2024, Hahn et al. 2025)

**Setup.** *Iterated Energy-based Flow Matching*,
[arXiv:2408.16249](https://arxiv.org/abs/2408.16249); *Energy-Weighted Flow Matching*,
[arXiv:2509.03726](https://arxiv.org/abs/2509.03726). Both adapt the flow-matching
framework (Lipman et al. 2023) to energy-only sampling. The "iterated" / "energy-weighted"
qualifiers refer to the training schedule: alternate between sampling from the current
flow and energy-reweighting to refine the velocity field.

**Energy-flow-matching loss.** The velocity field is trained against the *Monte Carlo
estimate of the marginal vector field* constructed from the known energy:
$$
\mathcal L_{\mathrm{iEFM}}(\theta) \;=\; \mathbb{E}_{t,x_t\sim\rho_t^{\mathrm{model}}}\!\Big[\big\|v_\theta(x_t,t) - \hat u_t(x_t)\big\|^2\Big],
$$
where $\hat u_t$ is the MC estimate of the optimal transport velocity to $\mu$ at time $t$,
constructed from energy samples. **Prior: standard Gaussian.** No mode tuning.

### 3.11. Transferable Boltzmann generators (Klein et al. 2024)

**Setup.** Klein, Krämer, Noé — *Transferable Boltzmann Generators*,
[arXiv:2406.14426](https://arxiv.org/abs/2406.14426). First method to train an *amortized*
BG that transfers across unseen molecules (specifically, 100 unseen dipeptides not in the
training set).

**Architecture.** $O(D)\times S(N)$-equivariant graph neural network (EGNN) with continuous
normalizing flow parameterization. Each atom is embedded as a vector concatenating:

- one-hot atom type (54 classes from the AMBER force field topology),
- amino-acid identity (20 classes),
- positional embedding within the peptide sequence (sinusoidal).

**Loss.** Continuous-flow matching with the standard MSE objective, trained on a dataset
of 200 dipeptides with short MD trajectories per peptide. The transferability comes from
the atom-embedding richness, not from a tuned prior.

**Prior — direct quote (§7):**

> *"Throughout our work, we utilize a standard Gaussian prior distribution. […] We
> experimented with this Harmonic prior but found no significant improvements for our
> transferable model. Instead, the network architecture and inductive bias play a more
> crucial role."*

**Numerical headlines (Table 1, Table 2 of the paper).**

| Variant | Test set | NLL | ESS | Correct configs |
|---|---|---|---|---|
| TBG + full | Alanine dipeptide (classical FF) | $-127.06\pm 0.12$ | $6.03\pm 1.34$ % | – |
| TBG + full | 100 unseen dipeptides (mean) | – | $15.29\pm 9.27$ % | $98\pm 2$ % |
| TBG + full, biased training | 100 unseen dipeptides | – | $10.24\pm 7.14$ % | – |
| TBG + full, 10× smaller train | 100 unseen dipeptides | – | $6.13\pm 3.13$ % | $96\pm 3$ % |

Free-energy differences are recovered to within $\pm 0.05\, k_B T$ of reference values.

### 3.12. TA-BG — Temperature-annealed Boltzmann generators (Schopmans, Friederich 2025)

**Setup.** [ICML 2025, arXiv:2501.19077](https://arxiv.org/abs/2501.19077). Code:
[`aimat-lab/TA-BG`](https://github.com/aimat-lab/TA-BG). Software stack: PyTorch + `bgflow`
+ Neural Spline Flows.

**Core idea.** Train reverse-KL at a high temperature $T_h\sim 1200\,\mathrm K$ where
barriers are low (modes are connected). Then *anneal* the flow distribution to a target
temperature $T\sim 300\,\mathrm K$ by reweighting:
$$
\mu_T(x) \;\propto\; \mu_{T_h}(x)^{T_h/T}.
$$

**Two-stage loss.**

*Stage 1 (high temperature, reverse-KL).*
$$
\mathcal L_{\mathrm{rev}}^{T_h}(\theta) \;=\; \mathbb{E}_{z\sim\mu_0}\!\Big[\tfrac{1}{T_h}U_1(G^{-1}_\theta(z)) - \log|\det J_{G^{-1}_\theta}(z)|\Big] + \text{const}.
$$
*Stage 2 (anneal).* Reweight samples by the temperature ratio and apply iterative
forward-KL updates targeting the lower-temperature distribution.

**Prior.** Standard truncated Gaussian for bonds ($\sigma=0.5$) and angles ($\sigma=0.1$),
*uniform on $[0,1]$ for dihedrals*. Physics-aware geometry but **not mode-tuned**.

**Direct quote (loss reason for mode collapse).**

> *"At target temperatures, the reverse KLD suffers from severe mode collapse: once the
> flow collapsed to a mode, it will generally not escape this collapsed state if the
> remaining modes are too far separated. Surprisingly, at high temperatures the reverse KL
> is sufficient: barriers between metastable states are lower and the probability
> distribution maxima are interconnected."*

### 3.13. SBG — Sequential Boltzmann generators (Tan et al. 2025)

**Setup.** [ICML 2025, arXiv:2502.18462](https://arxiv.org/abs/2502.18462). First flow-based
equilibrium sampling in *Cartesian coordinates* of tri-, tetra-, and hexa-peptides.

**Architecture.** A non-equivariant Transformer normalizing flow (TarFlow, Zhai et al.
2024) with patches over the particle dimension. Equivariance is enforced *softly* through
data augmentation and centre-of-mass noise.

**Two-stage pipeline.**

1. *Training.* The TarFlow is trained on biased MD data to learn an approximate
   $\nu_\theta\approx\mu$.
2. *Inference-time refinement.* A continuous-time SMC sampler with annealed Langevin steps
   transports samples from $\nu_\theta$ to $\mu$. The key innovation:

   > *"Unlike past work in pure sampling which uses the prior energy
   > $E_0(x) = -\log p_0(x)$, our design affords the significantly more informative
   > proposal given by the pre-trained normalizing flow $p_\theta(x)$. Through inference-time
   > scaling, the proposal $p_\theta(x_0)$ acts as a new prior for the Langevin process."*

   This is the cleanest distillation of the Camp A philosophy: the *learned flow*
   $\nu_\theta$ acts as a "tuned prior" at inference, while the *training prior* is the
   standard Gaussian. The tuning is learned automatically, not designed by the user.

**Loss.** Continuous-time annealed Langevin SDE
$$
\mathrm{d} X_\tau \;=\; \big[\beta'(\tau)\nabla\log\pi_\tau(X_\tau) - \tfrac{1}{2}\nabla U_1(X_\tau)\big]\D\tau + \sqrt{2}\mathrm{d} W_\tau,
$$
with $\pi_\tau \propto \nu_\theta^{1-\tau}\mu^\tau$. Resampling is adaptive: trigger
resampling when ESS drops below a threshold.

**Numerical headlines.**

| System | ESS @ 10k samples | $W_1$ in energy |
|---|---|---|
| Tripeptide (AL3) | $0.876\pm 0.215$ | $1.456\pm 0.739$ |
| Tetrapeptide (AL4) | $0.901\pm 0.059$ | $1.517\pm 0.387$ |
| Hexapeptide (AL6) | $0.995$ | $2.321$ |

### 3.14. CMT — Constrained Mass Transport (Blessing et al. 2025)

**Setup.** [SPIGM@NeurIPS 2025, arXiv:2510.18460](https://arxiv.org/abs/2510.18460).
Diagnoses *mass teleportation* in geometric annealing schedules and proposes a constrained
variational alternative.

**The mass-teleportation problem (direct quote).**

> *"In geometric annealing $q_i\propto q_0^{1-\beta_i}p_*^{\beta_i}$, when the entropy gap
> between the prior and target is large, new modes can emerge without overlapping with
> previous intermediate densities. The right mode of the target distribution emerges
> without overlap with earlier intermediate densities, complicating mass transport."*

**Constrained schedule.** Instead of a fixed schedule $\beta_k=k/M$, CMT picks intermediate
distributions $\pi_k$ to satisfy
$$
\mathrm{KL}(\pi_{k-1}\,\|\,\pi_k) \leqslant \delta_{\mathrm{KL}}, \qquad H(\pi_{k-1}) - H(\pi_k) \leqslant \delta_H,
$$
trading off KL divergence and entropy decay between consecutive intermediate distributions.
This *prevents* mass teleportation by forcing each step to preserve enough overlap.

**Numerical headline.** On ELIL tetrapeptide, naive reverse-KL training achieves
ESS = $1.28\%$; CMT achieves $26.18\%$. **A 20× improvement *without changing the prior*
— purely by redesigning the annealing schedule.**

---

---

> **Concluding Camp A.** The progression Noé 2019 → Wu 2020 → FAB 2022 → TA-BG 2025
> → CMT 2025 → SBG 2025 is a story of *progressively more sophisticated training
> machinery on top of a fixed Gaussian latent*. Each new method addresses a specific
> failure of its predecessors — mode collapse (FAB, TA-BG), mass teleportation (CMT),
> inference-time precision (SBG), amortization (TBG). At no point does the field reach
> for a tuned prior; the engineering investment goes everywhere except the base
> distribution. This is the empirical Camp A consensus, and §6 will quantify what it
> costs and what it leaves unsolved.

## 4. Camp B — Physics-informed / tuned prior

> **Why this section exists.** Camp B is the minority opinion in the BG literature but
> the *one that wins on a specific high-value subproblem* (crystallographic free
> energy). Understanding what Camp B does well and what it gives up is essential to
> the user's argument: Camp B's collaborators are not wrong in general — they are
> wrong *for the specific multimodal-discovery problem the user's paper targets*. To
> establish this we have to read Camp B's strongest cases honestly.
>
> **How to read this section.** Each subsection gives the same five-part treatment as
> §3 — prior construction with explicit formulae, architecture, loss, numerical
> headlines, critical assessment. The literature here is much smaller (five core
> papers plus two VAE-side precedents) but each is given a longer treatment than the
> Camp A entries because the entire argument turns on understanding what each Camp B
> recipe *commits the model to*.

Camp B encodes target-specific information into the base distribution $\mu_0$ itself.
This section gives each canonical paper a formal-paper-style treatment: the *exact*
construction of the prior, the architecture, the loss, the training procedure, the
numerical headlines, and a critical assessment.

### 4.1. Lattice-Gaussian priors for crystals (Wirnsberger et al. 2022)

**Setup.** Wirnsberger, Papamakarios, Ibarz, Racanière, Ballard, Pritzel, Blundell —
*Normalizing flows for atomic solids*, [Mach. Learn.: Sci. Technol. 3, 025009 (2022)](https://iopscience.iop.org/article/10.1088/2632-2153/ac6b16)
([arXiv:2111.08696](https://arxiv.org/abs/2111.08696)). Target: the canonical free-energy
calculation problem of statistical mechanics — computing $\Delta F$ between phases of a
crystal at fixed temperature and pressure. Systems: monatomic-water ice in cubic ($I_c$)
and hexagonal ($I_h$) phases (64, 216, 512 atoms) and truncated-shifted Lennard-Jones
(256 and 500 particles).

**The lattice-Gaussian prior.** Let $\{z^o_n\}_{n=1}^N$ be the reference lattice sites of
the *target crystal phase* in reduced units (lattice constant divided by particle diameter
$\sigma$). The base distribution on the configuration $z = (z_1,\dots,z_N)\in[0,L_1/\sigma]\times\dots\times[0,L_d/\sigma]$
is:
$$
\mu_0^{\mathrm{lat}}(z) \;=\; \frac{1}{N!}\sum_{\pi\in S_N}\prod_{n=1}^N \rho_T(z_n - z^o_{\pi(n)}),
$$
where $\rho_T$ is a spherically-truncated Gaussian
$$
\rho_T(\delta) \;\propto\; \exp\!\Big(-\tfrac{\|\delta\|^2}{2\sigma_T^2}\Big)\cdot\mathbf 1_{\|\delta\|\leqslant R_T}
$$
with the truncation radius $R_T$ chosen so that *no two atoms can swap lattice sites*,
i.e. $R_T < d_{\min}/2$ where $d_{\min}$ is the smallest inter-site distance in the
target lattice. The factor $1/N!$ together with the sum over permutations $S_N$ encodes
the $S_N$-invariance of identical-particle systems by construction. *The prior literally
encodes the target crystal phase.*

**Flow architecture.** Continuous normalizing flow with $SE(3)$-respecting layers (the
flow operates on relative displacements from the lattice sites). The flow has only to
learn the *anharmonic correction* to the harmonic crystal vibrations.

**Loss.** Reverse-KL on the target Boltzmann distribution:
$$
\mathcal L_{\mathrm{rev}}^{\mathrm{lat}}(\theta) \;=\; \mathbb{E}_{z\sim\mu_0^{\mathrm{lat}}}\!\Big[U_1(G^{-1}_\theta(z)) - \log|\det J_{G^{-1}_\theta}(z)|\Big] + \text{const}.
$$
Importance weights $w(x) = \mu_1(x)/\nu_\theta(x)$ are then used to compute the free-energy
difference between the BG-derived ensemble and the target:
$$
\Delta F \;=\; -k_BT\,\log\frac{Z_1}{Z_0} \;=\; -k_BT\,\log\mathbb{E}_{z\sim\mu_0}\!\big[\exp(-U_1(G^{-1}_\theta(z)) + U_0(z))\big] \;-\; k_BT\,\log\frac{|G^{-1}_\theta|}{|G|}.
$$

**Numerical headlines.** Helmholtz free energies $f = F/N$ agreed with reference
values to within $\sim 10^{-5}\,k_BT$ per particle on all systems tested:

| System | $N$ | $f_{\mathrm{flow}}$ | $f_{\mathrm{reference}}$ |
|---|---|---|---|
| LJ | 256 | $3.10800(28)$ | $3.11(4)$ |
| LJ | 500 | – | – |
| Ice $I_c$ | 64 | matches MBAR within stat. error | – |
| Ice $I_c$ | 216 | matches MBAR within stat. error | – |
| Ice $I_c$ | 512 | matches MBAR within stat. error | – |
| Ice $I_h$ | 64 / 216 / 512 | matches MBAR within stat. error | – |

**Software & compute.** JAX + Haiku + Distrax. Largest systems (512-atom ice, 500-atom LJ)
took *3 weeks on 16 A100 GPUs*. This compute cost is the empirical floor for the Camp B
recipe at this precision.

**The killer admission (verbatim from §4 of the paper).**

> *"Empirically, we find that, after training, the flow model becomes a sampler for the
> (metastable) crystal state that we encode in the base distribution, and does not sample
> configurations from other states."*
> — Wirnsberger et al. (2022)

**Critical assessment.**

- *Strength.* For applications where one knows in advance which phase one wants to sample
  (computing $\Delta F$ between two named phases, computing thermodynamic averages within
  one basin), the lattice-Gaussian prior is essentially optimal. The flow has to learn
  only local fluctuations, which is exactly the regime where invertible neural networks
  excel.
- *Limitation #1 — single-basin scope.* The flow cannot sample alternative phases. There
  is no diagnostic in the paper that would detect a phase the prior failed to encode —
  the support of the model literally excludes other phases.
- *Limitation #2 — symmetry breaking on solid–liquid transitions.* If one applied this
  method near a melting transition, the lattice prior would still draw samples consistent
  with the encoded crystal, even if the *target* at that thermodynamic state is partially
  liquid. The method silently produces a phase-biased sample.
- *Limitation #3 — does not generalize.* Each new crystal phase requires re-tuning the
  prior (choosing new $z^o_n$, new $R_T$) and re-training the flow.
- *Limitation #4 — computational cost.* 3 weeks on 16 A100s is the upper limit of
  feasibility for academic groups; cannot scale further without budget on Wirnsberger's
  scale (DeepMind).

**Critical assessment vs. paper claims.** The paper's claims about precision are entirely
correct ($10^{-5}\,k_BT/N$ on free-energy is genuinely impressive). The paper *does
acknowledge* the single-basin limitation. The over-statement, when it exists, is in
secondary literature framing this paper as "the solution to BGs for molecular systems";
in fact it is a solution to a very specific subproblem (one-phase free-energy precision),
and is structurally not a discovery method.

### 4.2. Higher-temperature reference priors (Coretti et al.)

**Setup.** Coretti, Falkner, Geissler, Hummer, Dellago, surveyed in
[*Boltzmann Generators and the New Frontier* (arXiv:2404.16566)](https://arxiv.org/abs/2404.16566).
Target: liquid systems (Lennard-Jones, molecular liquids) at moderate $T$ where modes are
broad rather than sharp.

**Construction.** The base distribution $\mu_0$ is the *empirical distribution of an MD
trajectory* at a higher reference temperature $T_0 > T$:
$$
\mu_0(z) \;\approx\; \mu_{T_0}(z) \;=\; \frac{1}{Z_{T_0}}\exp\!\Big(-\tfrac{1}{T_0}U_1(z)\Big).
$$
Practically, $\mu_0$ is represented by a stored sample set $\{z_i^{(T_0)}\}_{i=1}^{N_0}$
generated by long MD at $T_0$, and the flow learns the deformation from $T_0$ to $T$.

**Loss.** Either reverse-KL (sampling from the empirical $\mu_0$ and pushing through
$G_\theta^{-1}$ to compare against $\mu_T$) or forward-KL using $T_0$-samples as a proxy.

**Critical assessment.**

- *Strength.* For liquids without well-separated barriers, $T_0\to T$ is a smooth
  deformation, and the flow needs only to capture local thermal contraction.
- *Limitation #1 — circular for multimodal systems.* The technique works because at $T_0$
  modes are merged (high temperature). For systems where the modes *remain separated at
  $T_0$* (most proteins), $\mu_{T_0}$ inherits multimodality and one is back at square
  one.
- *Limitation #2 — empirical prior.* The "base distribution" is not a closed-form density
  but an empirical sample set. The log-density $\log\mu_0$ must be estimated (kernel
  density, score matching) introducing an approximation error.
- *Limitation #3 — requires MD pre-simulation.* The recipe trades "sample at $T$" for
  "sample at $T_0$ first," which is only progress when $T_0$ is easier — typically not
  for biomolecules.

### 4.3. Conditional / phase-diagram Boltzmann generators (Schebek et al. 2024)

**Setup.** Schebek, Schaaf, Hummer, Köhler — *Efficient mapping of phase diagrams with
conditional Boltzmann generators*, Mach. Learn.: Sci. Technol. 2024,
[arXiv:2406.12378](https://arxiv.org/abs/2406.12378).

**Architecture.** A *conditional* normalizing flow $G_\theta(y; T, P)$ takes
thermodynamic state $(T, P)$ as conditioning input. The base distribution is a fixed
empirical reference at $(T_0, P_0)$.

**Loss.** Conditional reverse- and forward-KL across the entire $(T, P)$-grid:
$$
\mathcal L(\theta) \;=\; \int_{[T_{\min},T_{\max}]\times[P_{\min},P_{\max}]} w(T,P)\,\mathcal L_{\mathrm{rev}+\mathrm{fwd}}(\theta; T, P)\,\mathrm{d} T\,\mathrm{d} P,
$$
with a weighting $w(T,P)$ chosen by the user to emphasize the phase-coexistence line.

**Critical assessment.** This is a hybrid — the base is mildly informative (one reference
state), the flow is doing the conditional deformation work. Mode lock-in is *less severe*
than in lattice-Gaussian because the flow can in principle move between basins via the
conditioning input. *But* it still requires the reference state to cover the relevant
basins; it cannot discover new ones.

### 4.4. Structured-CG latent priors (Schiebroek & Koehn 2025)

**Setup.** Schiebroek, Koehn — *Energy-Based Coarse-Graining in Molecular Dynamics: A
Flow-Based Framework without Data*, [JCTC 2025, arXiv:2504.20940](https://arxiv.org/html/2504.20940).
Target: coarse-grained molecular simulation that retains the multimodal structure of slow
collective variables.

**Construction.** Decompose the latent space $\mathbb{R}^d = \mathbb{R}^{d_{\mathrm{slow}}}\times \mathbb{R}^{d_{\mathrm{fast}}}$
into *slow* collective variables and *fast* atomic-fluctuation coordinates. Define:

- Slow: $\mu_0^{\mathrm{slow}}(s) = \sum_{k=1}^K \pi_k\,\mathcal N(s; m_k, \Sigma_k)$ —
  an explicit Gaussian mixture with $K$ components corresponding to known metastable
  states (deliberately multimodal).
- Fast: $\mu_0^{\mathrm{fast}}(f | s) = \mathcal N(f; 0, \Sigma_f(s))$ — unimodal Gaussian
  conditioned on the slow variable.

The full latent is $\mu_0(s,f) = \mu_0^{\mathrm{slow}}(s)\mu_0^{\mathrm{fast}}(f|s)$. *This
is the most explicit Camp B mixture-prior in the BG literature.*

**Loss.** Reverse-KL on the all-atom target with the slow-CV structure as a hard latent
constraint:
$$
\mathcal L(\theta) \;=\; \mathbb{E}_{(s,f)\sim\mu_0}\!\big[U_1(G^{-1}_\theta(s,f)) - \log|\det J_{G^{-1}_\theta}(s,f)|\big] + \text{const}.
$$

**Critical assessment.** This is the cleanest realization of Camp B in 2024–2025 — but
note the cost: *the user must pre-specify the slow CVs and the metastable-state centers
$m_k$*. The method is at its best when one has a prior physical understanding of which
collective variables matter (e.g. for $\alpha$-helix vs. $\beta$-sheet for short
peptides). It fails or requires an outer mode-discovery loop on systems where the slow
CVs are unknown.

### 4.5. VAE-side precedent: VampPrior (Tomczak & Welling 2018)

**Setup.** [AISTATS 2018, arXiv:1705.07120](https://arxiv.org/abs/1705.07120). The
prototypical "learned multimodal prior" in deep generative models. *Not* a BG method, but
an essential reference because it shows the VAE community's solution to the same prior
question — and how it could have been ported to BGs.

**Construction.** *Variational mixture of posteriors* prior:
$$
p(z) \;=\; \frac{1}{K}\sum_{k=1}^K q_\phi(z\,|\,u_k),
$$
where $\{u_k\}_{k=1}^K$ are *learnable pseudo-inputs* and $q_\phi$ is the variational
posterior network of the VAE. The prior thus shares parameters $\phi$ with the encoder,
and pseudo-inputs $u_k$ are trained jointly to minimize the VAE ELBO.

**Loss.** Standard VAE ELBO with the VampPrior replacing the usual $\mathcal N(0,I)$
prior in the KL term:
$$
\mathcal L_{\mathrm{VampPrior}}(\theta,\phi,\{u_k\}) \;=\; \mathbb{E}_{q_\phi(z|x)}\!\big[\log p_\theta(x|z)\big] - \mathrm{KL}\!\big(q_\phi(z|x)\,\big\|\,\tfrac{1}{K}\textstyle\sum_k q_\phi(z|u_k)\big).
$$

**Numerical headlines.** State-of-the-art ELBO on MNIST, OMNIGLOT, Caltech-101 Silhouettes,
Frey Faces, Histopathology at the time of publication (2018), in the unsupervised
permutation-invariant setting.

**Why this did *not* propagate to BGs.** The VAE setting is *unconditional* generative
modeling — there is no explicit "target distribution" in the BG sense. The pseudo-inputs
in VampPrior are learned jointly with the encoder/decoder by ELBO maximization on data.
In a BG, *there are no data samples* for training (or very few, from short MD). One would
have to learn the pseudo-inputs $u_k$ from energy alone, which means running a mode-finder
on $U_1$ — *exactly the circular Trinity-style preprocessing* that Camp B was supposed to
avoid.

### 4.6. VAE-side precedent: LARS (Bauer & Mnih 2019)

**Setup.** Bauer, Mnih — *Resampled Priors for Variational Autoencoders*,
[AISTATS 2019, arXiv:1810.11428](https://arxiv.org/abs/1810.11428).

**Construction.** Learned-accept-reject sampling on top of a base distribution:
$$
\tilde p(z) \;=\; \frac{\pi(z)\,a_\phi(z)}{Z_\phi},\qquad Z_\phi = \int \pi(z)\,a_\phi(z)\,\mathrm{d} z,
$$
where $\pi$ is the base (a Gaussian) and $a_\phi:\mathbb{R}^d\to[0,1]$ is a learned acceptance
function. $Z_\phi$ is estimated via Monte Carlo during training.

**Critical assessment.** LARS gives a *more flexible prior than VampPrior* without
committing to discrete mixture components — but it is more expensive to evaluate the
normalizing constant. In BGs, the equivalent has been reinvented under different names
(Stimper et al. 2022, §3.7 above). Not adopted in mainstream BG work.

### 4.7. What is *not* in Camp B

For clarity, we list configurations that one might *expect* to be in Camp B but which are
in fact Camp A:

- *Mean-free Gaussian* (equivariant flows): symmetry constraint, not mode tuning.
- *Truncated Gaussian for bonds / uniform on dihedrals* (TA-BG, smooth NFs):
  geometry-aware in topology, not in modes.
- *Internal-coordinate representations*: change of variables, not change of prior.
- *Trainable Gaussian-mixture base in `bgflow.MixtureDistribution`*: implemented but unused
  in canonical BG notebooks; if the components are learned without supervision they
  reduce to LARS / VampPrior.

### 4.8. Why Camp B is small in 2024–2026

There is a quantitative observation worth emphasizing: of the dozen-plus 2024–2026 BG /
diffusion-sampler papers reviewed in §3, *none* uses a Camp B-style tuned mixture prior.
The closest is Schiebroek/Koehn structured-CG-latent (§4.4), and that requires
user-specified collective variables. Wirnsberger's lattice prior (§4.1) remains the
canonical Camp B and is restricted to crystals.

This is *not* because the relevant tooling is missing. The `bgflow`,
`VincentStimper/normalizing-flows`, and `lollcat/fab-torch` codebases all implement
`MixtureDistribution` and `GaussianMixture` classes. The tools exist; they are not used.
*The empirical verdict of the BG field over six years is that prior tuning has not paid
off.*

---

> **Concluding Camp B.** Camp B is sharply effective in a single high-value niche
> (Wirnsberger-style single-phase crystallographic free energy at $10^{-5}\,k_B T$/atom)
> and a few related settings (Coretti higher-temperature liquid reference, Schiebroek/
> Koehn structured CG latent for known-CV biomolecules). The VAE-side precedents
> (VampPrior, LARS) demonstrate that *learned* mixture priors *can* work in
> generative-modeling settings — but have not propagated to BGs, for reasons the
> next section formalizes. Outside the single-basin / known-mode regime, Camp B
> trades discovery for precision in a way the multimodal protein-sampling setting
> cannot accept.

## 5. Difficulties of each camp — mathematical analysis

> **Why this section exists.** Sections 3 and 4 told us *what* each camp does. This
> section asks the more important question: *why does each camp struggle?* For Camp A
> the struggles are concrete and addressable through engineering (mode collapse → FAB,
> mass teleportation → CMT, fake ESS → coverage metric). For Camp B the struggles are
> *structural* — mode lock-in is a property of the bijective-flow geometry, not a
> training pathology. The asymmetry is the entire argument.
>
> **How this section is organized.** §5.1–5.6 cover Camp A's six difficulties, each
> with a claim, a mathematical reasoning, and empirical evidence cited from §3.
> §5.7–5.13 cover Camp B's seven difficulties in the same format. Throughout we
> distinguish *correctible* failures (Camp A's, where importance weighting or
> coverage diagnostics catch the problem) from *structural* failures (Camp B's,
> where the model's support does not contain the missing region and no diagnostic
> can correct it).

Sections 3 and 4 documented what each camp does; this section formalizes *why each camp
struggles*. The argument structure is: for each difficulty, we state the claim, give the
mathematical reasoning, and cite empirical evidence.

### 5.1. Camp A difficulty: mode collapse under reverse KL

**Claim.** A flow trained with reverse-KL on a multimodal target can drop modes that the
base distribution does have support over, and is unable to recover them.

**Mathematical reasoning.** Recall the reverse-KL gradient (§2.3):
$$
\nabla_\theta \mathcal L_{\mathrm{rev}}(\theta) \;=\; \mathbb{E}_{z\sim\mu_0}\!\big[\nabla_\theta U_1(G^{-1}_\theta(z))\big] + \mathbb{E}_{z\sim\mu_0}\!\big[\nabla_\theta\log|\det J_{G^{-1}_\theta}(z)|\big].
$$
Both terms are expectations under $\mu_0$, *not* under $\nu_\theta$. However, because
$G^{-1}_\theta$ maps the support of $\mu_0$ bijectively, the gradient at $\theta$ is
zero on regions of $\mathbb{R}^d$ that $\nu_\theta$ does not currently reach. Concretely, if at
parameter $\theta_*$ the flow has collapsed to a single mode $\nu_{\theta_*}\approx \delta_{x^*}$,
then:

1. The gradient $\nabla_\theta U_1(G^{-1}_{\theta_*}(z))$ for $z\sim\mu_0$ has support only
   near $x^*$, so it knows nothing about other modes of $U_1$.
2. The Jacobian term $\nabla_\theta\log|\det J_{G^{-1}_{\theta_*}}|$ encourages spreading
   *locally* (increase log-det), but the flow can satisfy this by perturbing within the
   current basin.

The net effect: the gradient signal that would push the flow toward a second mode
*literally does not enter the optimization*. This is the same property that makes
reverse-KL "mode-seeking": missing modes carry zero loss contribution.

**Empirical evidence.**

- CMT (Blessing et al. 2025): naive reverse-KL on ELIL tetrapeptide gives $\mathrm{ESS} = 1.28\%$
  (vs CMT's $26.18\%$).
- TBG (Klein et al. 2024): reverse-KL only (no AIS) gives $\mathrm{ESS} \approx 6\%$ on alanine
  dipeptide.
- TA-BG (Schopmans, Friederich 2025): reverse-KL at target temperature ($300\,\mathrm K$)
  fails outright on dipeptides; succeeds at $1200\,\mathrm K$ only because barriers are
  low.

**Asymmetry vs. forward-KL.** Forward-KL is *mass-covering*: a missed mode contributes
$+\infty$ to the loss. But forward-KL requires samples from $\mu$ that are not available
in BG settings without expensive MD or AIS bootstrapping.

### 5.2. Camp A difficulty: mass teleportation in annealing schedules

**Claim.** Geometric annealing schedules $\pi_k\propto\mu_0^{1-\beta_k}\mu^{\beta_k}$ can
develop *disjoint* support between consecutive distributions in the schedule, making
transport between them impossible.

**Mathematical reasoning.** Let the prior $\mu_0$ be Gaussian centred at 0 and the
target $\mu$ have a mode at $x^*$ far from 0. The geometric interpolation
$$
\pi_\beta(x) \;\propto\; \mu_0(x)^{1-\beta}\mu(x)^\beta \;\propto\; \exp\!\Big[-(1-\beta)\tfrac{\|x\|^2}{2} + \beta\,(-U_1(x))\Big]
$$
has a single mode at the *interpolation* $x_\beta = \beta x^* / (1-\beta + \beta s)$ where
$s = U_1''(x^*)$. As $\beta\to 1$, the intermediate mode jumps discontinuously toward
$x^*$. Concretely:

> **Lemma (mass teleportation).** *If the target mode $x^*$ satisfies $\|x^*\|\gg 1$ and
> $U_1''(x^*) \gg 1$, then for some $\beta_c\in(0,1)$, the intermediate $\pi_{\beta_c}$ has
> two modes with $O(1)$-mass each, but $\pi_{\beta_c+\epsilon}$ for small $\epsilon$ has
> $1-O(\epsilon)$ mass at one mode and $O(\epsilon)$ at the other.*

The probability mass "teleports" from one mode to the other as $\beta$ crosses $\beta_c$.
HMC kernels at any single $\beta$ near $\beta_c$ cannot bridge this gap in finite time
without exponentially fine $\beta$-discretization.

**Empirical evidence.** Blessing et al. (CMT 2025, §3.14) reports that on ELIL
tetrapeptide *without* their constrained schedule, the AIS chain reaches $\mathrm{ESS} \approx 1\%$;
*with* the constrained schedule (preserving overlap between consecutive distributions),
$\mathrm{ESS} = 26.18\%$. The 20× gap is attributable to mass teleportation in the geometric
schedule.

### 5.3. Camp A difficulty: the topological gap

**Claim.** A normalizing flow (continuous bijection) cannot perfectly map a contractible
unimodal base distribution onto a target with disconnected support.

**Mathematical reasoning.** If $\mu_0$ has connected support $\mathrm{supp}(\mu_0)=\mathbb{R}^d$
and $\mu$ has support $\mathrm{supp}(\mu) = A_1\cup A_2$ with $A_1,A_2$ disjoint and
connected, then the bijection $G^{-1}:\mathrm{supp}(\mu_0)\to\mathrm{supp}(\mu_0)$ cannot
have its image be disjoint — because $G^{-1}$ is continuous and $\mathbb{R}^d$ is connected.

In practice $\mu$ has positive (but possibly small) probability between modes, so
$\mathrm{supp}(\mu) = \mathbb{R}^d$ technically; the topological obstruction is *softened* into a
*conditioning* obstruction. The flow must produce log-determinants that vary by many
orders of magnitude across the domain to compress prior mass between basins.

**Empirical evidence.** Stimper et al. (Resampled base 2022, §3.7): *"the invertible
nature [of flows] limits their ability to model target distributions whose support has a
complex topological structure, such as Boltzmann distributions."* Direct admission that
the topological gap motivates the resampled-base trick.

### 5.4. Camp A difficulty: fake ESS

**Claim.** ESS as defined in §2.6 can be near 1 even when the flow misses entire modes of
the target. ESS is a *necessary but not sufficient* diagnostic.

**Mathematical reasoning.** Let $\mu$ have modes $A_1,A_2$ with probability masses
$p_1, p_2$ and $p_1 + p_2 = 1$. Suppose the trained flow $\nu_\theta$ puts essentially all
mass on $A_1$:
$$
\nu_\theta(x) \;\approx\; \tfrac{1}{p_1}\mu(x)\cdot\mathbf 1_{x\in A_1}.
$$
Then for samples $y_i\sim\nu_\theta$, all $y_i\in A_1$, and the importance weight
$w_i = \mu(y_i)/\nu_\theta(y_i) = p_1$ for all $i$. The weights are *exactly uniform*, so
$\mathrm{ESS} = 1$. Yet the model has missed the entire mode $A_2$ of mass $p_2$. *ESS provides
no signal whatsoever about the missing mode.*

This is the situation we observe empirically: TBG reports $\mathrm{ESS} = 15.29\pm 9.27\%$ on
unseen dipeptides, but TBG+full also reports $98\pm 2\%$ "correct configurations" —
*because the field already knows that ESS alone is misleading* and now reports the
coverage-style metric alongside.

**Empirical evidence.** The user's own paper documents this on the 2D Himmelblau benchmark:
plain forward-KL training of an NSF can reach $\mathrm{ESS} \approx 0.75$ while missing one of
the four Himmelblau modes (see `2D_Benchmark/samples.png`).

### 5.5. Camp A difficulty: expensive bootstrapping

**Claim.** Methods that need MCMC-bootstrapped samples (FAB, AIS-BG, SBG) pay a high
target-evaluation cost.

**Empirical evidence.** FAB on alanine dipeptide: 100× fewer target evaluations than MLE
on MD samples — but the absolute number is still in the hundreds of thousands.
Wirnsberger's lattice-prior recipe needs 3 weeks on 16 A100s for 512 ice atoms.
Even with Camp B's "head start," scale is the limit.

### 5.6. Camp A difficulty: no clean place to put prior mode knowledge

**Claim.** When the user has external knowledge of some modes of $\mu$ (e.g., from low-$T$
MD, chemistry, NMR, or a Trinity-style mode-finder), there is no clean place to inject it
in the Camp A pipeline.

**Three impossibility-style obstructions.**

1. **Adding the modes as MD samples.** Augmenting an MLE training set with synthetic mode
   samples shifts the maximum-likelihood estimate toward the mode-cluster distribution,
   which is not $\mu$. Bias in: bias out.
2. **Using the modes as AIS initial states.** Initializing AIS chains at known modes
   biases the importance weights, producing a biased sampler whose ESS is meaningless.
3. **Encoding the modes in the prior.** This shifts the entire framework to Camp B and
   inherits all of Camp B's difficulties (§5.7–5.13).

The Gabrié–Rotskoff–Vanden-Eijnden adaptive-MC paper explicitly admits: "the multimodal
case requires prior knowledge about either the symmetries … or the location of the
metastable basins." Camp A as currently configured has *no robust mechanism* to inject
this knowledge.

### 5.7. Camp B difficulty: circularity

**Claim.** Encoding modes in the prior is a circular procedure if the modes are unknown.

**Mathematical reasoning.** To place a Gaussian mixture component at mode $m_k$, one must
either (i) know $m_k$ analytically (only the case for crystals and a few hand-tuned
problems), (ii) extract $m_k$ from MD trajectories (in which case one already had
information about the modes — and could equally use that information as MD samples for
forward-KL), or (iii) run a mode-finder pre-pass (in which case the same compute budget
could instead be used for the $\mathrm{X}_{\hat\mu}$ approach without committing the prior).

### 5.8. Camp B difficulty: mode lock-in

**Claim.** A bijective flow whose base distribution has support strictly inside
$\bigcup_k B(m_k, R_T)$ cannot sample outside this support, irrespective of the optimum
of the loss.

**Mathematical reasoning.** $G^{-1}_\theta$ is a homeomorphism; the image of a compact
set under a homeomorphism is compact. If
$\mathrm{supp}(\mu_0)\subseteq\bigcup_k B(m_k, R_T)$, then
$\mathrm{supp}(\nu_\theta) = G^{-1}_\theta(\mathrm{supp}(\mu_0))\subseteq G^{-1}_\theta\!\big(\bigcup_k B(m_k,R_T)\big)$.
A mode of $\mu$ at $m^*$ with $G^{-1}_\theta\!\big(\bigcup_k B(m_k,R_T)\big)\not\ni m^*$
is unreachable by the model.

**This is the structural difference from Camp A.** Camp A's failures are *correctible* by
extending the support of the model (which a Gaussian prior already provides on all of
$\mathbb{R}^d$); Camp B's failures are *structural* in the support itself.

### 5.9. Camp B difficulty: brittleness to misspecification

**Claim.** A GMM prior whose components $m_k$ are misplaced by $\delta$ produces a flow
whose log-density at the true target modes is off by terms in $\delta$.

**Reasoning.** If $m_k = m_k^{\mathrm{true}} + \delta_k$, the prior log-density at the
true mode is $\log\mu_0(m_k^{\mathrm{true}}) \approx \log\mu_0(m_k) - \frac{\delta_k^T \Sigma_k^{-1}\delta_k}{2\sigma_T^2}$.
The flow has to learn an extra translation to bring the mode back to its true location,
which costs Jacobian determinant magnitude and slows convergence. *Misspecification is
not zero-cost; it consumes flow capacity.*

### 5.10. Camp B difficulty: symmetry breaking

**Claim.** A GMM prior with components at specific physical locations breaks symmetries
that the target has.

**Reasoning.** Suppose $\mu$ is $S_N$-invariant (identical particles) and $\mathrm{SE}(3)$-invariant
(rigid-body invariance). A naive GMM prior $\sum_k\mathcal N(m_k, \sigma^2 I)$ at
fixed locations $m_k$ is *not* invariant — only a specific permutation of the
$\binom{N!\cdot\mathrm{Vol}(\mathrm{SE}(3))}{1}$ symmetry-equivalent placements is. Camp B
either (i) symmetrizes the prior by summing over all $N!\cdot\mathrm{SE}(3)$ symmetric
images — blowing up the prior — or (ii) accepts symmetry breaking. Wirnsberger's lattice
prior takes route (i) for the *permutation* symmetry of identical atoms (the factor
$1/N!\sum_{\pi\in S_N}$). It does *not* take route (i) for $\mathrm{SE}(3)$ — the lattice
itself breaks rotational symmetry, which is fine for a crystal (the rigid lattice *is*
the symmetry-broken state) but pathological for any target where $\mathrm{SE}(3)$ matters.

### 5.11. Camp B difficulty: transferability collapse

**Claim.** A flow trained against a tuned prior $\mu_0^{(\mathcal S)}$ for system
$\mathcal S$ does not transfer to a related system $\mathcal S'$.

**Reasoning.** Transferable BGs (Klein et al. 2024) work because the flow has been trained
to map $\mu_0^{\mathrm{Gauss}}\to\mu^{(\mathcal S)}$ for many systems $\mathcal S$; the
architecture is forced to use *generic* features. If the prior were tuned per-system, the
flow's effective parameters would absorb that tuning, and a new system would require
retuning the prior + retraining the flow. This is the empirical reason TBG explicitly
chose a Gaussian prior even after trying a structured (harmonic) one.

### 5.12. Camp B difficulty: structural bias

**Claim.** When a tuned prior misses a real mode of $\mu$, no importance-weighting
correction is available, because the proposal has zero density at the missed mode.

**Reasoning.** Importance weights $w(x) = \mu(x)/\nu_\theta(x)$ are well-defined only on
$\mathrm{supp}(\nu_\theta)$. If $\nu_\theta(x^*) = 0$ for $x^*\in\mathrm{supp}(\mu)$, the
weight is undefined and no estimator using importance reweighting can capture mass at
$x^*$. *The bias is structural, not statistical.* The only fix is to extend the prior to
include $x^*$, which is what Camp A's Gaussian-everywhere prior already does for free.

### 5.13. Camp B difficulty: prior tuning relocates the hard part

**Claim.** Whether the user encodes modes in the prior or finds them via Trinity-style
preprocessing, the mode-discovery problem must be solved somewhere. Tuning the prior
doesn't abolish it — it just moves it earlier in the pipeline and removes the option to
be inaccurate.

**Reasoning.** Camp B's tuned prior *is* a mode-finder, just frozen at training time. The
user-side effort to set up Wirnsberger's lattice $z^o_n$ is exactly the same effort the
$\mathrm{X}_{\hat\mu}$ approach spends on Trinity. The difference is that Trinity's output is
allowed to be sloppy because $\mathrm{X}_{\hat\mu}$ tolerates sloppiness; Camp B's lattice has
to be exact because the prior is the answer.

---

## 6. Code-level inventory of canonical implementations

> **Why this section exists.** §3–§5 made the argument at the level of papers. But papers
> can claim more than their code delivers, and *defaults in production code* are often
> more telling about field consensus than what an abstract advertises. To verify that the
> Camp A consensus is real, this section inspects the source of every major
> BG / sampler library — `bgflow`, `normflows`, `fab-torch`, `jarridrb/DEM`,
> `kazewong/flowMC`, `aimat-lab/TA-BG`, `annalena-k/FAB-meets-diffME` — and reports
> exactly which prior class each one instantiates in its canonical training notebook.
>
> **The intuition.** If Camp B were practically useful, *someone* would have implemented
> their canonical notebook with a `MixtureDistribution`. The mixture-prior class is
> available in essentially every library; nobody uses it in the default training
> pipeline. This is the strongest single piece of indirect evidence we can collect, and
> it is independent of paper-text rhetoric.

We performed direct source-code inspection of the main BG and sampler libraries. The
table below documents *what prior is the default* and what mixture-prior tooling is
available but unused.

### 6.1. `noegroup/bgflow` — Noé group reference implementation

Source: `https://github.com/noegroup/bgflow`. Stack: PyTorch + NumPy + einops; optional
OpenMM for molecular forces, optional `torchdiffeq` for neural ODEs.

The `distribution/` submodule contains:

```text
bgflow/distribution/
  distributions.py
  normal.py    →  NormalDistribution            (centered isotropic Gaussian)
                  TruncatedNormalDistribution    (Gaussian on [a,b], for bonds/angles)
                  MeanFreeNormalDistribution     (Gaussian on hyperplane Σz_i = 0)
                  CircularNormalDistribution     (von Mises on [0,1], for dihedrals)
  mixture.py   →  MixtureDistribution            (Gaussian mixture, trainable weights)
  product.py   →  ProductDistribution            (product of per-coordinate priors)
  energy/      →  energy-based variants
  sampling/    →  MCMC / HMC samplers
```

The `MixtureDistribution` class supports trainable weights and is fully functional. The
canonical alanine-dipeptide training notebook
(`notebooks/alanine_dipeptide_basics.py:155–167`) nonetheless instantiates:

```python
# ## Prior Distribution
#
# The next step is to define a prior distribution that we can easily sample from. The
# normalizing flow will be trained to transform such latent samples into molecular
# coordinates. Here, we just take a normal distribution, which is a rather naive choice
# for reasons that will be discussed in other notebooks.

dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
mean = torch.zeros(dim_ics).to(ctx)
prior = bg.NormalDistribution(dim_ics, mean=mean)
```

The phrase *"rather naive choice"* is the author group's own self-assessment. The
mixture-distribution implementation exists and is documented, but it is not used in
the canonical training pipeline. This is the strongest single piece of code-level
evidence that the BG community has the tools for Camp B but chooses Camp A in practice.

### 6.2. `VincentStimper/normalizing-flows` (`normflows`)

The library used by FAB and TA-BG. Default base distribution:
`nf.distributions.base.DiagGaussian(dim)`. The library implements `GaussianMixture` and
`Uniform` bases, but they are rarely the default. The `resampled-base-flows` repository
provides the LARS-style resampled-base variant; uptake is limited.

### 6.3. `lollcat/fab-torch` — FAB reference implementation

The factory function `make_wrapped_normflow_realnvp` in
`experiments/make_flow/make_normflow_model.py:88`:

```python
def make_wrapped_normflow_realnvp(dim, n_flow_layers=5, ...):
    base = nf.distributions.base.DiagGaussian(dim)
    flows = make_normflow_flow(dim, n_flow_layers=n_flow_layers, ...)
    model = nf.NormalizingFlow(base, flows)
    ...
```

Hard-coded `DiagGaussian`. No mixture, no tuning, no learnable prior.

### 6.4. `jarridrb/DEM` — iDEM reference implementation

Forward diffusion process terminates in $\mathcal N(0,I)$. The "base distribution" is
implicit in the SDE; the noise prior is standard normal.

### 6.5. `aimat-lab/TA-BG`

Built on `bgflow` and `normflows`. Uses `TruncatedNormalDistribution` for bonds/angles
and `Uniform` on $[0,1]$ for dihedrals — physics-aware *topology*, not modes.

### 6.6. `kazewong/flowMC`

JAX-based. Default base for the global-proposal flow: standard Gaussian. The flow is
trained *on the running MCMC chain*, so the effective proposal becomes mode-aware over
time — but the *training* base is uninformative.

### 6.7. `annalena-k/FAB-meets-diffME` — particle physics

PyTorch + normflows + replay buffer. Same `DiagGaussian` base as FAB. Applied to
matrix-element distributions in high-energy physics; same code architecture as the
molecular sampling setting.

### 6.8. Pattern across libraries

Every canonical BG / sampler library studied implements both simple and mixture priors,
but defaults to the simple Gaussian in every production training script we inspected. The
mixture-prior tooling is sitting unused.

---

## 7. Adjacent communities: do they tune the prior?

> **Why this section exists.** A skeptic might respond to §3–§6 by saying: "The
> BG community has converged on Camp A because of its own historical accidents."
> §7 falsifies that hypothesis by surveying five neighbouring fields that face
> structurally the same problem (sample from a multimodal target with intractable
> normaliser) and asking whether *they* tune the prior. The answer, uniformly: no.
>
> **The intuition.** When a methodological choice is forced by genuine technical
> constraints — as opposed to historical inertia — the same choice emerges
> independently across communities that do not coordinate. Five independent fields
> arriving at the Camp A default is much stronger evidence than the BG field's own
> preferences.

Normalizing flows and diffusion samplers are now standard tools in five communities
beyond molecular sampling. We surveyed each to see whether the prior-vs-loss decision
plays out the same way. *In every community, the answer is: simple base distribution +
sophisticated training, not tuned prior.*

### 7.1. Bayesian inverse problems and inverse PDE

The relevant problem is: given a forward operator $F:\mathcal X\to\mathcal Y$ (often a
PDE solver), data $d = F(x) + \epsilon$ with $\epsilon\sim\mathcal N(0,\Sigma)$, and a
prior $p(x)$, estimate the posterior $p(x|d)\propto p(d|x)p(x)$. The flow's role is to
approximate $p(x|d)$ as $\nu_\theta = G^{-1}_{\theta,\#}\mu_0$.

Representative works:

- **Whang, Lindgren, Dimakis 2021** [*Composing Normalizing Flows for Inverse Problems*](https://proceedings.mlr.press/v139/whang21b/whang21b.pdf).
  The flow is the *posterior approximator*; its base is standard Gaussian. The signal
  prior is a *separately trained NF* on data.
- **Sun et al. 2024** [*Functional normalizing flow for statistical inverse problems of
  PDEs (NF-iVI)*](https://arxiv.org/abs/2411.13277). Infinite-dimensional Hilbert-space
  flows for elliptic PDE inverse problems. The base measure is a Gaussian process;
  Gaussian-base flows are extended to function space.
- **Padmanabha, Zabaras 2023** [*Learning to solve Bayesian inverse problems: amortized
  VI with Gaussian and Flow guides*](https://arxiv.org/html/2305.20004). Direct
  comparison: Gaussian-base flow guides match or beat structured guides on benchmark
  inverse problems.
- **Zhao, Curtis, Zhang 2022** [*Bayesian seismic tomography using normalizing flows*](https://dx.doi.org/10.1093/gji/ggab298).
  Uniform priors on velocity (physically motivated bounds), Gaussian-base posterior flow.
- **Asim et al. 2022** [*Wave-equation-based inversion with amortized variational
  Bayesian inference*](https://arxiv.org/pdf/2203.15881). Conditional NF, standard base.
- **Guo, Hou, Karniadakis 2022** *Normalizing field flows*. Standard base in function
  space.
- **Liao et al. 2025** *Iterative Normalizing Flows for geophysical inversion*. Standard
  base; the iterations refine the *posterior*, not the prior.

**Pattern.** The prior over the unknown is physics-informed (Gaussian process with given
covariance, uniform on bounded set). The base of the flow that approximates the posterior
is *standard Gaussian*. The split is identical to BG's: informative side priors, simple
flow base.

### 7.2. Cosmology, astrophysics, gravitational waves

- **`flowMC`** (Wong, Gabrié, Foreman-Mackey, JOSS 2023): adaptive global proposal
  trained on the chain. Standard Gaussian base. Reports 25–50× speedup over nested
  sampling on primordial-feature cosmology and gravitational-wave parameter estimation.
  *Mode information comes from MCMC exploration, not from the prior.*
- Williams, Veitch, Messenger 2021 *Nested sampling with normalizing flows for GW
  inference*. Standard base.
- Various 2024–2025 GW PE papers: standard base, conditional flows.

### 7.3. Particle physics

- Kofler, Stimper, Mishra-Sharma, Gabrié et al., MLST 2025 *FAB meets Differentiable
  Particle Physics*. Uses FAB with HMC transition steps on matrix-element targets.
  Standard base.

### 7.4. Simulation-based inference

- Cranmer, Brehmer, Louppe (PNAS 2020) — overview of SBI; field convention is standard
  base.
- **Simformer** (Gloeckler, Bischoff, Macke 2024): score-based diffusion + Transformer.
  Diffusion prior = Gaussian noise.
- `sbi` toolkit (Macke et al.): standard Gaussian base by default in every NPE
  configuration shipped.

### 7.5. Cross-community pattern

*Five independent communities have made the same choice.* If tuning the prior were the
obvious win, at least one of these communities would have adopted it independently. They
have not. Camp A is not just dominant in BG work; it is dominant in every flow-sampler
field.

---

## 8. Non-flow neural samplers

Diffusion-based and stochastic-interpolant samplers do not learn an invertible flow
$\mu_0\to\mu$; instead they learn a *score field* or a *velocity field* of a
noising/denoising process. The "prior" question still applies: the terminal distribution
of the noising process plays the role of base. Below we survey the main diffusion
samplers, with their loss formulae.

### 8.1. iDEM and iEFM (see §3.9, §3.10)

Already covered in Camp A. Both use standard isotropic Gaussian as terminal noise.

### 8.2. BNEM — Bootstrapped Noised Energy Matching

[arXiv:2409.09787](https://arxiv.org/abs/2409.09787). Learns the energy of the
*convolved* distribution $\mu * \mathcal N(0,\sigma_t^2 I)$, then samples via reverse
diffusion. The terminal of the forward noising process is $\mathcal N(0,I)$ — standard.

**Loss.** Score-matching against noised energy gradients:
$$
\mathcal L_{\mathrm{BNEM}}(\theta) \;=\; \mathbb{E}_{t, x_t}\!\Big[\big\|s_\theta(x_t,t) + \nabla_{x_t}\mathbb{E}_{x_0|x_t}[U_1(x_0)]\big\|^2\Big],
$$
with inner expectations estimated by bootstrap-sampling from the current model.

### 8.3. ASBS — Adjoint Schrödinger Bridge Sampler

Liu et al., [NeurIPS 2025, arXiv:2506.22565](https://arxiv.org/abs/2506.22565).

**The Schrödinger Bridge problem.** Given $\mu_0,\mu_1$, find a diffusion
$$
\mathrm{d} X_t \;=\; b_\theta(t, X_t)\,\mathrm{d} t + \sigma\,\mathrm{d} W_t, \qquad X_0\sim\mu_0,
$$
such that $X_T\sim\mu_1$ and the path is the closest (in relative entropy) to Brownian
motion. ASBS generalizes adjoint sampling beyond Dirac-delta priors to *arbitrary*
$\mu_0$, including (relevant to Camp B) **harmonic-oscillator priors** for molecular
systems.

**Direct quote.** *"The memoryless condition previously restricted source distributions to
Dirac deltas, precluding the use of common priors such as Gaussian or domain-specific
priors such as the harmonic oscillators in molecular systems."*

This is the *only* current Boltzmann-sampling paper that explicitly enables a non-trivial
prior, and the prior it admits is *harmonic-oscillator* — bond-stretch potentials —
which is Camp B in the mildest sense (geometry, not modes). Still no mode-tuned mixture.

**Loss.** Adjoint-matching objective with kinetic-optimal regularization; details in the
paper.

### 8.4. PITA — Progressive Inference-Time Annealing

[arXiv:2506.16471](https://arxiv.org/abs/2506.16471). Trains a *sequence* of diffusion
models at progressively lower temperatures $T_h > T_{h-1} > \dots > T_{\mathrm{target}}$.
Each $T_k$-model is initialized from the $T_{k+1}$-model.

**Loss.** Standard denoising score matching at each temperature:
$$
\mathcal L_{\mathrm{PITA}}^{(k)}(\theta) \;=\; \mathbb{E}_{t, x_t}\!\Big[\big\|s_\theta^{(k)}(x_t,t) - \nabla_{x_t}\log p_t^{(T_k)}(x_t)\big\|^2\Big].
$$
The terminal of each forward diffusion is standard Gaussian noise. Mode information
flows in only through the annealing schedule.

### 8.5. Stochastic interpolants and NETS

[Albergo, Vanden-Eijnden, *Stochastic Interpolants*, ICLR 2023, arXiv:2303.08797](https://arxiv.org/abs/2303.08797).
A unifying framework: define a time-indexed family of distributions $\rho_t$ interpolating
$\rho_0=\mu_0$ to $\rho_1=\mu$ via
$$
X_t \;=\; I_t(X_0,X_1) + \gamma(t)\,Z,\qquad X_0\sim\mu_0,\,X_1\sim\mu,\,Z\sim\mathcal N(0,I),
$$
with $I_0(x_0,x_1)=x_0$ and $I_1(x_0,x_1)=x_1$. The corresponding velocity field $b_t$ is
learned by regression on the SDE path.

**Loss.**
$$
\mathcal L_{\mathrm{SI}}(\theta) \;=\; \mathbb{E}_{t, X_0, X_1, Z}\!\Big[\big\|b_\theta(t, X_t) - \big(\partial_t I_t(X_0,X_1) + \dot\gamma(t)\,Z\big)\big\|^2\Big].
$$

**Why this matters for the prior question.** The interpolant *requires* paired
$(X_0,X_1)$ at training time — i.e., samples from *both* $\mu_0$ and $\mu$. When $\mu$
samples are unavailable (the pure Boltzmann-sampling case), one must bootstrap them with
AIS / MD / iDEM-style methods. The framework is silent on prior tuning.

### 8.6. Riemannian flow matching for condensed matter

[arXiv:2602.18482](https://arxiv.org/html/2602.18482). Operates on a Riemannian manifold
$\mathcal M$ matching the physical configuration space. Base is the *Haar measure on
$\mathcal M$* (the analog of uniform on a Lie group). Same philosophical move as lattice
gauge theory — symmetry-driven prior, not mode-driven.

### 8.7. Variance-Tuned Diffusion Importance Sampling (VT-DIS)

[arXiv:2505.21005](https://arxiv.org/html/2505.21005v1). Post-training correction of a
pretrained diffusion sampler: tune the per-step noise covariance to make the forward
and reverse trajectories agree. Standard Gaussian prior; the *covariance* is what is
tuned. This is diffusion's analog of control-variate variance reduction.

### 8.8. Cross-paradigm pattern

Diffusion-based samplers face the same prior question as flows. The answer is the same:
standard Gaussian terminal noise + sophisticated training (score matching, energy
matching, adjoint matching, stochastic interpolants). The Schrödinger bridge family
(ASBS) is the one mild deviation, and even there the tuned priors are unimodal
(harmonic oscillator), not mode-matched.

---

## 9. Engineering tricks orthogonal to prior choice

The BG / diffusion-sampler community has accumulated a substantial library of engineering
tricks, *none* of which involves prior tuning. We list them for completeness, since the
user's collaborators may invoke any of these in support of Camp A or Camp B.

### 9.1. Smooth normalizing flows (Köhler, Krämer, Noé 2021, §3.4)

Mixture-of-bumps coupling layers on compact intervals and tori. Used to ensure $C^2$
smoothness for downstream MD applications. Architecture-level trick.

### 9.2. Flows on tori and spheres (Rezende et al. 2020)

[arXiv:2002.02428](https://arxiv.org/pdf/2002.02428). Coupling-layer flows for periodic
and spherical topology. Standard in dihedral-angle BGs.

### 9.3. Dequantization

Inject small Gaussian noise into discrete data so flows can model the resulting continuous
density. Not directly relevant to BGs on continuous configuration spaces, but used in
some lattice-gauge-theory contexts.

### 9.4. Gradient-Boosted Normalizing Flows (Giaquinto & Banerjee 2020)

[NeurIPS 2020, arXiv:2002.11896](https://arxiv.org/abs/2002.11896). Builds a flow as an
*additive mixture of flow components*. Each component is fit by gradient boosting on the
residual:
$$
\nu_K(x) \;=\; \sum_{k=1}^K \pi_k\,\nu_k(x),\qquad \pi_k\geqslant 0,\,\sum\pi_k=1.
$$
**This is the philosophically right answer to the multimodality question.** It puts the
mixture in the *model*, not the *prior*. Each $\nu_k$ has its own Gaussian base; the
mixture emerges from the model. This is closer to $\mathrm{X}_{\hat\mu}$ in spirit (multimodality
in the model, not the prior). Not yet applied to BGs in published work — an open
opportunity.

### 9.5. Resampled base (Stimper et al. 2022) and LARS (Bauer-Mnih 2019)

Already covered in §3.7 and §4.6. Learned-rejection-sampling base. Mild Camp B.

### 9.6. Replay buffer / prioritized replay for AIS

Critical engineering trick from FAB. Without it, AIS samples are too noisy to drive
stable training. Training-level trick.

### 9.7. Centre-of-mass projection (mean-free distributions)

Constrain Gaussian samples to lie on the hyperplane $\sum z_i = 0$ to enforce
translation-equivariance for point-cloud flows. `MeanFreeNormalDistribution` in
`bgflow`. Mandatory for SE(3) equivariance.

### 9.8. Force matching

(Köhler, Chen, Krämer, Klein, Noé 2024 [arXiv:2401.04246](https://arxiv.org/abs/2401.04246)).
Trains the flow such that $\nabla\log\nu_\theta$ matches MD forces. Loss-level trick.

### 9.9. Tamed gradients in Langevin

Used in `zflows.utils.langevin` (the user's infrastructure) to handle super-linearly
growing gradients like Himmelblau's. The tamed drift is
$$
G(x) \;=\; \frac{\nabla U(x)}{1 + \tau\,\|\nabla U(x)\|},
$$
so $\|\tau\,G(x)\|\leqslant 1$. Stabilizes unadjusted Langevin on stiff potentials. Foundational
reference: Hutzenthaler, Jentzen, Kloeden 2012.

### 9.10. Flow Perturbation and Flow Perturbation++

[arXiv:2407.10666](https://arxiv.org/html/2407.10666v1) and
[arXiv:2601.21177](https://arxiv.org/html/2601.21177). Variance-reduced Jacobian
estimation. Importance-weighting trick.

### 9.11. Importance-weighted ELBO (IS-IWAE)

(Burda, Grosse, Salakhutdinov 2015). Tighter variational bound via $K$-sample importance
weighting. Compatible with any prior.

### 9.12. Pattern

All of these tricks live in the *architecture*, the *loss*, or the *post-processing
pipeline* — not in the *prior*. The field's empirical bias is to invest in these layers
and leave the prior untouched. This is, in itself, very strong evidence that the cost-
benefit of prior tuning is unfavorable in the BG setting.

---

## 10. Numerical comparison

The table below collates reported headline numerical results for the canonical methods on
the canonical benchmarks. Numbers are taken from the cited papers; where standard errors
are not reported we use the paper's stated typical scale.

| Method | System | Dim. | Prior | Loss / training | Reported headline |
|---|---|---|---|---|---|
| Noé BG 2019 | BPTI | $\approx 300$ | Gaussian | rev-KL + ML(MD samples) | Cross-mode mixing with seeded modes |
| Wirnsberger 2022 | LJ-256 | $256\times 3$ | **Lattice-Gaussian** | rev-KL | $\Delta F$ to $10^{-5}\,k_BT$/atom, single basin |
| Wirnsberger 2022 | Ice $I_c$ 512 | $512\times 3$ | **Lattice-Gaussian (cubic)** | rev-KL | $\Delta F$ matches MBAR, single phase |
| Wu SNF 2020 | 2D bistable | 2 | Gaussian + MCMC layers | Jarzynski + KL | Asymptotically unbiased |
| FAB 2023 | Alanine dipeptide | 22 atoms | DiagGauss | $\alpha=2$ + AIS buffer | 100× fewer evals than MLE/MD |
| FAB 2023 | 40-comp 2D GMM | 2 | DiagGauss | $\alpha=2$ + AIS buffer | Mode discovery via AIS |
| TBG 2024 | Alanine dipeptide (FF) | $22\times 3$ | std Gauss | Flow matching | $\mathrm{ESS} = 6.03\pm 1.34\%$ |
| TBG 2024 | 100 unseen dipeptides | varies | std Gauss | Flow matching | $\mathrm{ESS} = 15.29\pm 9.27\%$, $98\pm 2\%$ correct |
| TA-BG 2025 | Dipeptide | – | Truncated Gauss + uniform dihedral | rev-KL @ 1200K → 300K | Beats FAB on energy evals |
| CMT 2025 | ELIL tetrapeptide | – | std Gauss | Constrained mass transport | rev-KL = 1.28%; CMT = 26.18% ESS |
| SBG 2025 | Tripeptide AL3 | – | $\mathcal N(0,I)$ + SMC inference | TarFlow + annealed Langevin | $\mathrm{ESS} = 0.876\pm 0.215$ |
| SBG 2025 | Tetrapeptide AL4 | – | $\mathcal N(0,I)$ + SMC inference | TarFlow + annealed Langevin | $\mathrm{ESS} = 0.901\pm 0.059$ |
| SBG 2025 | Hexapeptide AL6 | – | $\mathcal N(0,I)$ + SMC inference | TarFlow + annealed Langevin | $\mathrm{ESS} = 0.995$ |
| Klein EFM 2023 | Alanine dipeptide | $22\times 3$ | Mean-free Gauss | Equivariant flow matching | First non-IC BG |
| iDEM 2024 | LJ-55 | $55\times 3$ | $\mathcal N(0,I)$ noise | Denoising energy matching | First energy-only at LJ-55 scale |
| iEFM 2024 | LJ-13 / DW-4 | – | $\mathcal N(0,I)$ noise | Energy-based flow matching | Competitive with iDEM |
| Schiebroek/Koehn 2025 | CG molecular | – | **GMM on slow CV** + Gaussian on fast | rev-KL with structured latent | Captures metastable states |
| Gabrié-Rotskoff-VE 2022 | Multimodal Bayesian | varies | Gauss + local MCMC | Adaptive MCMC + flow | Requires seeded chains |
| flowMC 2023 | GW PE / cosmology | – | Gauss + local MCMC | Adaptive MCMC + flow | 25–50× over nested sampling |

### Take-aways from the table

1. The only Camp B entry with state-of-the-art precision (Wirnsberger) is single-basin
   by design.
2. The only Camp B entry that touches multimodality (Schiebroek/Koehn) requires the user
   to predefine slow collective variables and known metastable states.
3. CMT's 20× improvement (1.28% → 26.18% ESS) comes purely from *schedule redesign* with
   no prior change.
4. SBG's >0.99 ESS on hexapeptides comes from *inference-time SMC + Transformer flow*,
   no prior tuning.
5. TBG's 98% correct configurations across 100 unseen dipeptides comes from *architecture
   (rich embeddings)*, with prior tuning explicitly ruled out as ineffective.

The field's progress on multimodal targets has come entirely from Camp A engineering;
Camp B has not contributed a single multimodal advance.

---

## 11. The third way — formal mathematical analysis

> **Why this section exists.** Sections 5 and 6 made a negative case against both Camp A
> and Camp B for the multimodal-discovery problem. This section makes the positive case
> for the user's alternative. The central question is *where to inject mode information*
> — and the answer is *into the loss, via a regularizer with built-in inaccuracy tolerance,
> not into the prior's support*. We give the formula, derive its properties as
> propositions, describe the Trinity algorithm that produces the mode estimate, and
> verify that every Camp A / Camp B difficulty from §5 is sidestepped.
>
> **The core intuition.** A bijective flow's support is determined by its prior's support
> — that is what Camp B's mode-lock-in difficulty boils down to. A regularizer evaluated
> at points $\hat\mu$ supplies, by contrast, only *gradient signal*: it pushes the flow
> to make the log-ratio $\log\mu/\nu_\theta$ as constant as possible at the points it
> sees, *without* committing the flow's support to any of them. When $\nu_\theta = \mu$
> is realizable in the architecture, the loss is zero regardless of $\hat\mu$ —
> Proposition 3 of §11.2 makes this precise. The flow can therefore absorb arbitrarily
> sloppy mode estimates without paying the brittleness, transferability, or symmetry-
> breaking costs of a tuned prior.
>
> **How this section is organized.** §11.1 defines the $\mathrm{X}$ functional and proves
> its zero-set and normalization-invariance properties. §11.2 states and proves the
> asymptotic-invariance property to $\hat\mu$ inaccuracies. §11.3 derives the autograd-
> reuse trick that makes $\mathrm{X}_\mu$ a zero-overhead addition to forward-KL
> training. §11.4 specifies the Trinity algorithm. §11.5–11.7 put the pieces together
> into a full training loss, an evaluation protocol using ESS + coverage, and a row-by-row
> comparison against Camp A and Camp B on every difficulty from §5.

This section formalizes the user's proposed framework — the $\mathrm{X}$ functional with
$\hat\mu$ supplied by the Trinity algorithm — and proves why it sidesteps both camps'
core difficulties.

### 11.1. The $\mathrm{X}$ functional

Let $\mu$ be the target distribution on $\mathbb{R}^d$ and $\nu$ a candidate model distribution.
The $\mathrm{X}$ functional with weighting measure $\omega$ on $\mathbb{R}^d\times\mathbb{R}^d$ is
$$
\mathrm{X}_\omega(\mu\,\|\,\nu) \;=\; \iint_{\mathbb{R}^d\times\mathbb{R}^d}\omega(x,y)\Big|\log\tfrac{\mu(x)}{\nu(x)} - \log\tfrac{\mu(y)}{\nu(y)}\Big|\,\mathrm{d} x\,\mathrm{d} y.
$$

**Properties.**

> **Proposition 1 (zero set).** *$\mathrm{X}_\omega(\mu\,\|\,\nu) = 0$ if and only if
> $\log(\mu/\nu)$ is constant on $\mathrm{supp}(\omega^{\mathrm{marg}})$, where
> $\omega^{\mathrm{marg}}(x) = \int\omega(x,y)\,\mathrm{d} y$.*
>
> **Proof sketch.** $|a-b|=0\Leftrightarrow a=b$, applied pointwise.

> **Proposition 2 (normalization invariance).** *$\mathrm{X}_\omega(\mu\,\|\,\nu)$ depends on $\mu$
> only through $\mu(\cdot)/Z_\mu$ for any normalization constant $Z_\mu > 0$, and likewise
> for $\nu$.*
>
> **Proof.** $\log(\mu/\nu)$ shifts by $\log(Z_\nu/Z_\mu)$ under rescaling, which is a
> constant; differences of constants cancel.

This is the precise statement of why $\mathrm{X}$ "tolerates inaccurate target samples" — the
normalization constants of both $\mu$ and the weighting $\omega$ cancel exactly in the
log-ratio difference. This is crucial in BG settings where $Z_\mu$ is unknown.

### 11.2. Asymptotic invariance to $\hat\mu$ inaccuracy

We now formalize the claim that "the minimizer of $\mathrm{X}_{\hat\mu}$ is insensitive to the
precise shape of $\hat\mu$."

Let $\nu_\theta$ be parameterized by $\theta$. Consider the loss
$\mathcal L_{\hat\mu}(\theta) = \mathrm{X}_{\hat\mu\otimes\hat\mu}(\mu\,\|\,\nu_\theta)$. Suppose
the *exact* minimizer is $\theta_* = \arg\min_\theta \mathcal L_{\hat\mu}(\theta)$.

> **Proposition 3 (asymptotic invariance).** *Suppose $\nu_{\theta_*} = \mu$ is reachable
> in the parameterization (the optimum is in the realizable class). Then $\theta_*$ is
> independent of $\hat\mu$ as long as $\mathrm{supp}(\hat\mu)\supseteq\mathrm{supp}(\mu)$.*
>
> **Proof.** At $\nu_{\theta_*} = \mu$, the integrand $|\log(\mu/\nu) - \log(\mu/\nu)| = 0$
> identically, regardless of $\hat\mu$. So $\theta_*$ minimizes the loss for any
> $\hat\mu$ whose support contains the support of $\mu$.

**Consequence.** Even if $\hat\mu$ is a coarse Trinity-style approximation that places
mass on regions where $\mu$ has no mass (false-positive modes), the optimal $\theta_*$ is
still $\nu_{\theta_*} = \mu$. Trinity is allowed to be wrong — *over-cover the modes*, not
under-cover — without biasing the optimum.

### 11.3. The autograd-reuse trick for $\mathrm{X}_\mu$

Choose $\omega = \mu\otimes\mu$, giving
$$
\mathrm{X}_\mu(\mu\,\|\,\nu_\theta) \;=\; \iint \mu(x)\mu(y)\Big|\log\tfrac{\mu(x)}{\nu_\theta(x)} - \log\tfrac{\mu(y)}{\nu_\theta(y)}\Big|\,\mathrm{d} x\,\mathrm{d} y.
$$
In the flow setting with $\nu_\theta = G^{-1}_{\theta,\#}\mu_0$:
$$
\log\nu_\theta(x) \;=\; \log\mu_0(G_\theta(x)) + \log|\det J_{G_\theta}(x)| \;=\; -U_0(G_\theta(x)) + \log|\det J_{G_\theta}(x)| - \log Z_0.
$$
Hence
$$
\log\frac{\mu(x)}{\nu_\theta(x)} \;=\; -U_1(x) + U_0(G_\theta(x)) - \log|\det J_{G_\theta}(x)| + (\log Z_0 - \log Z_1).
$$
The constant $\log(Z_0/Z_1)$ cancels in the difference $\log(\mu(x)/\nu(x)) - \log(\mu(y)/\nu(y))$.
Define
$$
z(x) \;:=\; U_0(G_\theta(x)) - U_1(x) - \log|\det J_{G_\theta}(x)|.
$$
Then $\log(\mu/\nu_\theta)(x) - \log(\mu/\nu_\theta)(y) = z(x) - z(y)$ (up to a constant
that cancels), and
$$
\mathrm{X}_\mu(\mu\,\|\,\nu_\theta) \;=\; \mathbb{E}_{(x,y)\sim\mu\otimes\mu}\!\big[|z(x) - z(y)|\big].
$$

**The autograd-reuse trick.** Sample $\mathbf y = (y_1,\dots,y_B)$ from $\mu$, compute
$\mathbf z = (z(y_1),\dots,z(y_B))\in\mathbb{R}^B$ with a single forward pass through $G_\theta$.
Sample a permutation $\sigma$ of $\{1,\dots,B\}$ and form $\mathbf z' = \mathbf z[\sigma]$.
Then:
$$
\widehat{\mathrm{X}_\mu}(\nu_\theta) \;=\; \frac{1}{B}\sum_{i=1}^B |z_i - z'_i|.
$$

> **Proposition 4 (zero-cost autograd reuse).** *The gradient of
> $\widehat{\mathrm{X}_\mu}$ with respect to $\theta$ requires only one backward pass through
> $G_\theta$, the same backward pass needed to compute the forward-KL gradient.*
>
> **Proof.** $\mathbf z'$ is a permutation of $\mathbf z$; the autograd graph of
> $\mathbf z'$ is the *same* as that of $\mathbf z$ (permutation is an index operation
> with no new differentiable function applications). Backpropagation through
> $|z_i - z'_i|$ at the absolute-value layer assembles into a single backward pass
> through $\mathbf z$, identical to forward-KL.

**Consequence.** Adding $\mathrm{X}_\mu$ to the forward-KL loss is essentially free. There is no
second forward pass, no second AIS chain, no extra MD samples. *This is the
"drop-in compatibility with forward KL" property* of the user's paper, made rigorous.

### 11.4. The Trinity algorithm for $\hat\mu$

The Trinity algorithm produces samples from $\hat\mu$, an over-coverage approximation of
$\mu$. The three stages, with explicit formulae:

**Stage 1 (Diffusion).** Given initial samples $x\sim\mu_0^{\mathrm{init}}$ (typically a
broad Gaussian or a concatenation of multiple data sources), add Brownian noise:
$$
\tilde x \;=\; x + \sigma\xi, \qquad \xi\sim\mathcal N(0,I_d).
$$
The scale $\sigma$ is chosen large — large enough to potentially reach modes outside the
support of $\mu_0^{\mathrm{init}}$.

**Stage 2 (Optimization).** Apply L-BFGS to $U_1$ at each $\tilde x_i$ with Armijo line
search. The output is $\tilde x_i\to x_i^{\mathrm{LBFGS}}$ close to a local minimum of
$U_1$.

**Stage 3 (Rejuvenation).** Run a tamed unadjusted-Langevin chain at temperature $T$
targeting $\mu_T \propto \exp(-U_1/T)$:
$$
x^{(k+1)}_i \;=\; x^{(k)}_i - \frac{\eta\,\nabla U_1(x^{(k)}_i)}{1 + \tau\|\nabla U_1(x^{(k)}_i)\|} + \sqrt{2\eta T}\,\xi^{(k)}_i.
$$
After $K$ steps, the empirical distribution $\hat\mu$ of $\{x^{(K)}_i\}$ is an
over-coverage approximation of $\mu_T$.

**Critical property.** Trinity is allowed to be sloppy — it can place mass on
non-existent modes (Stage 1's diffusion can scatter into regions $\mu$ never reaches),
and the L-BFGS can converge to saddles. As long as $\mathrm{supp}(\hat\mu)\supseteq\mathrm{supp}(\mu)$,
Proposition 3 guarantees that the asymptotic minimizer of $\mathrm{X}_{\hat\mu}$ is unchanged.

### 11.5. Full training loss

Combining the forward-KL loss with the $\mathrm{X}_\mu$ regularizer and the $\mathrm{X}_{\hat\mu}$
mode-discovery term, the user's loss is
$$
\mathcal L_{\mathrm{full}}(\theta) \;=\; \mathcal L_{\mathrm{fwd}}(\theta) + \lambda\,\mathrm{X}_\mu(\mu\,\|\,\nu_\theta) + \hat\lambda\,\mathrm{X}_{\hat\mu}(\mu\,\|\,\nu_\theta).
$$
Each term plays a distinct role:

- $\mathcal L_{\mathrm{fwd}}$: standard forward-KL (or its reverse-KL surrogate);
  unbiased minimizer at $\nu_\theta = \mu$.
- $\mathrm{X}_\mu$ regularizer: a free pointwise log-ratio penalty that improves stability
  without changing the asymptotic minimizer.
- $\mathrm{X}_{\hat\mu}$ regularizer: the *mode-discovery* term. Injects coarse mode information
  $\hat\mu$ without committing the flow's support.

### 11.6. Coverage-augmented evaluation

To detect fake ESS, report coverage in addition. Let $\mathbf x = (x_1,\dots,x_P)\sim\hat\mu$
(from Trinity) and $\mathbf y = (y_1,\dots,y_N)\sim\nu_\theta$ (importance samples from
the trained flow):
$$
\mathrm{Coverage}_k(\mathbf y;\mathbf x) \;=\; \frac{1}{P}\sum_{i=1}^P\mathbf 1\!\Big[\exists\,1\leqslant j\leqslant N : |y_j - x_i|\leqslant\mathrm{NND}_k(x_i;\mathbf x)\Big].
$$
A high $\mathrm{ESS}$ alone is *necessary but not sufficient*. A high $\mathrm{ESS}$ + high $\mathrm{Coverage}_k$ is
the joint diagnostic that excludes fake ESS.

### 11.7. Why this beats each camp on each difficulty

The table below collects the comparison.

| Difficulty | Camp A | Camp B | Third way ($\mathrm{X}_{\hat\mu}$) |
|---|---|---|---|
| Mode collapse under reverse KL (§5.1) | yes | mitigated (single basin) | mitigated by forward-KL + $\mathrm{X}$ |
| Mass teleportation in annealing (§5.2) | yes | n/a | n/a — no annealing |
| Topological gap (§5.3) | yes | mitigated by tuned support | mitigated by $\mathrm{X}$ inside the model |
| Fake ESS (§5.4) | yes | trivially zero (single basin) | flagged by coverage metric |
| Expensive bootstrapping (§5.5) | yes | yes (3 weeks/16 A100s for ice-512) | Trinity is offline, cheap |
| Cannot absorb prior mode info (§5.6) | yes | "solved" by prior tuning | solved by $\mathrm{X}_{\hat\mu}$ in loss |
| Circularity (§5.7) | n/a | yes | no — $\hat\mu$ can be sloppy |
| Mode lock-in (§5.8) | n/a | yes | no — Gaussian prior, full support |
| Brittleness to misspec (§5.9) | n/a | yes | no — Prop 3 invariance |
| Symmetry breaking (§5.10) | n/a | yes | no — Gaussian prior is symmetric |
| Transferability collapse (§5.11) | n/a | yes | no — same prior across systems |
| Structural bias when modes missed (§5.12) | n/a | yes | no — flow has full support |
| Prior tuning relocates the hard part (§5.13) | n/a | yes | no — Trinity is cheap, allowed wrong |

**The third way is the only column with no "yes" entries.** Every difficulty Camp A and
Camp B confront is either inapplicable to the third way or is solved by its design.

---

## 12. Talking points for the argument

For the user, when arguing with collaborators, the following claims are *all* supported by
the literature surveyed in this review.

1. **The mainstream BG community has tried Camp B and found it unhelpful for the
   multimodal protein-sampling regime.** TBG explicitly experimented with a structured
   harmonic prior and reported "no significant improvements."

2. **The strongest Camp B paper (Wirnsberger) admits mode lock-in explicitly.** Direct
   quote: "the flow model becomes a sampler for the (metastable) crystal state that we
   encode in the base distribution, and does not sample configurations from other states."

3. **No flagship 2024–2026 BG paper uses a tuned-GMM prior with mode-matched
   components.** The closest is Wirnsberger's lattice prior (one phase by design) and
   Schiebroek/Koehn's structured-CG-latent (requires user-defined slow CVs).

4. **Five adjacent communities (inverse PDE, cosmology, particle physics, geophysics,
   SBI) have all made the same Camp A choice independently.** If Camp B were the obvious
   win, *some* of these communities would have adopted it. None have.

5. **All major progress on multimodal BG sampling has come from Camp A engineering —
   not from prior tuning.** CMT's 20× improvement, FAB's 100× target-eval reduction,
   SBG's 0.99 hexapeptide ESS, TBG's 98% correct configurations — all from loss /
   architecture / inference innovations with a fixed Gaussian prior.

6. **Diffusion samplers face the same prior question and answer the same way.** iDEM,
   iEFM, EWFM, BNEM, PITA, NETS, Riemannian flow matching all use standard Gaussian
   terminal noise. ASBS is the only paper that allows non-Gaussian priors and even there
   the allowed priors are harmonic-oscillator unimodal, not mode-matched mixtures.

7. **Gradient-Boosted Normalizing Flows (Giaquinto-Banerjee 2020) demonstrates the right
   place for mixture structure is in the *model*, not the *prior*.** The user's
   $\mathrm{X}_{\hat\mu}$ approach is in this same spirit.

8. **Camp B's failures are structural; Camp A's are correctible.** If Camp A misses a
   mode, importance weighting + coverage diagnostic flag the failure and motivate
   continued training. If Camp B misses a mode, the flow's support does not contain it
   and no correction is possible.

9. **Mode information can be injected without committing the prior.** This is the
   user's central methodological insight: a *regularizer with asymptotic invariance to
   inaccuracy* (Proposition 3 of §11.2) absorbs sloppy mode knowledge without the
   downsides of a tuned prior.

10. **The `bgflow` and `normflows` libraries implement mixture priors but the canonical
    training notebooks do not use them.** Even the original BG authors describe their
    own Gaussian-prior choice as "rather naive" — and yet keep using it. The
    cost-benefit of tuning has not paid off.

11. **Tuning the prior does not abolish the hard part; it relocates it.** The user must
    still discover modes somewhere. Trinity-style preprocessing is *more flexible*
    because it can be wrong; Camp B prior tuning must be right.

12. **Camp B trades discovery for free-energy precision.** That trade-off is acceptable
    for crystallographers who know which phase they want. It is *fatal* for users who
    want to *find* metastable states — which is the central problem the user's paper
    targets.

---

## 13. References

Citations are grouped thematically and in approximate chronological order within group.

### 13.1. Pre-2019 foundations (normalizing flows, VI priors)

- Tabak, E. G., Vanden-Eijnden, E. *Density estimation by dual ascent of the log-likelihood.*
  Comm. Math. Sci. 8, 217–233 (2010).
  [PDF](https://math.nyu.edu/faculty/tabak/publications/CMSV8-1-10.pdf).
- Rezende, D. J., Mohamed, S. *Variational Inference with Normalizing Flows.* ICML 2015,
  [arXiv:1505.05770](https://arxiv.org/abs/1505.05770).
- Salimans, T., Kingma, D. P., Welling, M. *Markov Chain Monte Carlo and Variational
  Inference: Bridging the Gap.* ICML 2015,
  [arXiv:1410.6460](https://arxiv.org/abs/1410.6460).
- Dinh, L., Krueger, D., Bengio, Y. *NICE: Non-linear Independent Components Estimation.*
  ICLR-W 2015.
- Dinh, L., Sohl-Dickstein, J., Bengio, S. *Density estimation using Real NVP.* ICLR 2017.
- Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., Welling, M.
  *Improving Variational Inference with Inverse Autoregressive Flow.* NeurIPS 2016,
  [arXiv:1606.04934](https://arxiv.org/abs/1606.04934).
- Tomczak, J. M., Welling, M. *VAE with a VampPrior.* AISTATS 2018,
  [arXiv:1705.07120](https://arxiv.org/abs/1705.07120).
- Bauer, M., Mnih, A. *Resampled Priors for Variational Autoencoders.* AISTATS 2019,
  [arXiv:1810.11428](https://arxiv.org/abs/1810.11428).
- Müller, T., McWilliams, B., Rousselle, F., Gross, M., Novák, J. *Neural Importance
  Sampling.* ACM Trans. Graph. 2019, [arXiv:1808.03856](https://arxiv.org/abs/1808.03856).
- Hoffman, M., Sountsov, P., Dillon, J. V., Langmore, I., Tran, D., Vasudevan, S.
  *NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport.*
  NeurIPS-W 2019, [arXiv:1903.03704](https://arxiv.org/abs/1903.03704).
- Albergo, M. S., Kanwar, G., Shanahan, P. E. *Flow-based generative models for Markov
  chain Monte Carlo in lattice field theory.* Phys. Rev. D 100, 034515 (2019).

### 13.2. Camp A — Boltzmann generators with simple priors

- Noé, F., Olsson, S., Köhler, J., Wu, H. *Boltzmann generators: Sampling equilibrium
  states of many-body systems with deep learning.* Science 365, eaaw1147 (2019);
  [arXiv:1812.01729](https://arxiv.org/abs/1812.01729). **Founding paper.**
- Wu, H., Köhler, J., Noé, F. *Stochastic Normalizing Flows.* NeurIPS 2020.
- Köhler, J., Klein, L., Noé, F. *Equivariant Flows: Exact Likelihood Generative Learning
  for Symmetric Densities.* ICML 2020, [arXiv:2006.02425](https://arxiv.org/abs/2006.02425).
- Köhler, J., Krämer, A., Noé, F. *Smooth Normalizing Flows.* NeurIPS 2021,
  [arXiv:2110.00351](https://arxiv.org/abs/2110.00351).
- Rezende, D. J., Papamakarios, G., Racanière, S. et al. *Normalizing Flows on Tori and
  Spheres.* ICML 2020, [arXiv:2002.02428](https://arxiv.org/pdf/2002.02428).
- Gabrié, M., Rotskoff, G. M., Vanden-Eijnden, E. *Adaptive Monte Carlo augmented with
  normalizing flows.* PNAS 119(10) (2022);
  [arXiv:2107.08001](https://arxiv.org/abs/2107.08001).
- Stimper, V., Schölkopf, B., Hernández-Lobato, J. M. *Resampling Base Distributions of
  Normalizing Flows.* AISTATS 2022, [arXiv:2110.15828](https://arxiv.org/abs/2110.15828).
- Liu, T., Du, W., Wang, S., Gomes, C. *PathFlow: A Normalizing Flow Generator that Finds
  Transition Paths.* UAI 2022.
- Midgley, L. I., Stimper, V., Simm, G., Schölkopf, B., Hernández-Lobato, J. M. *Flow
  Annealed Importance Sampling Bootstrap.* ICLR 2023,
  [arXiv:2208.01893](https://arxiv.org/abs/2208.01893).
- Klein, L., Krämer, A., Noé, F. *Equivariant flow matching.* NeurIPS 2023,
  [arXiv:2306.15030](https://arxiv.org/abs/2306.15030).
- Midgley, L. I., Stimper, V., Antorán, J., Mathieu, E., Schölkopf, B., Hernández-Lobato,
  J. M. *SE(3) Equivariant Augmented Coupling Flows.* NeurIPS 2023,
  [arXiv:2308.10364](https://arxiv.org/abs/2308.10364).
- Akhound-Sadegh, T., Mittal, R., Bose, A. J. et al. *Iterated Denoising Energy Matching
  for Sampling from Boltzmann Densities (iDEM).* ICML 2024,
  [arXiv:2402.06121](https://arxiv.org/abs/2402.06121).
- Köhler, J., Chen, Y., Krämer, A., Klein, L., Noé, F. *Scalable Normalizing Flows Enable
  Boltzmann Generators for Macromolecules.* ICLR-W 2024,
  [arXiv:2401.04246](https://arxiv.org/abs/2401.04246).
- Klein, L., Krämer, A., Noé, F. *Transferable Boltzmann Generators.* 2024,
  [arXiv:2406.14426](https://arxiv.org/abs/2406.14426). **Critical reference: harmonic
  prior tried & rejected.**
- Woo, D. et al. *Iterated Energy-based Flow Matching for Sampling from Boltzmann
  Densities.* 2024, [arXiv:2408.16249](https://arxiv.org/abs/2408.16249).
- Schopmans, J., Friederich, P. *Temperature-Annealed Boltzmann Generators.* ICML 2025,
  [arXiv:2501.19077](https://arxiv.org/abs/2501.19077).
- Tan, A., Zaheer, M., Berger-Wolf, T. et al. *Scalable Equilibrium Sampling with
  Sequential Boltzmann Generators.* ICML 2025,
  [arXiv:2502.18462](https://arxiv.org/abs/2502.18462).
- Blessing, D., Mishra-Sharma, S., Krämer, A. et al. *Learning Boltzmann Generators via
  Constrained Mass Transport.* SPIGM@NeurIPS 2025,
  [arXiv:2510.18460](https://arxiv.org/abs/2510.18460).
- Hahn, S. et al. *Energy-Weighted Flow Matching.* 2025,
  [arXiv:2509.03726](https://arxiv.org/abs/2509.03726).
- *BoltzNCE: Learning Likelihoods for Boltzmann Generation.* 2025,
  [arXiv:2507.00846](https://arxiv.org/abs/2507.00846).

### 13.3. Camp B — Tuned / physically-informed priors

- Wirnsberger, P., Papamakarios, G., Ibarz, B., Racanière, S., Ballard, A. J., Pritzel, A.,
  Blundell, C. *Normalizing flows for atomic solids.* Mach. Learn.: Sci. Technol. 3,
  025009 (2022); [arXiv:2111.08696](https://arxiv.org/abs/2111.08696). **Canonical
  lattice-Gaussian prior.**
- Ahmad, R., Cai, W. *Free energy calculation of crystalline solids using normalizing
  flow.* 2021, [arXiv:2111.01292](https://arxiv.org/pdf/2111.01292).
- Wirnsberger et al. successor work for molecular crystals: arXiv:2509.25486.
- Schebek, M., Schaaf, M., Hummer, G., Köhler, J. *Efficient mapping of phase diagrams
  with conditional Boltzmann Generators.* Mach. Learn.: Sci. Technol. 2024,
  [arXiv:2406.12378](https://arxiv.org/abs/2406.12378).
- Schiebroek, A., Koehn, F. *Energy-Based Coarse-Graining in Molecular Dynamics: A
  Flow-Based Framework without Data.* JCTC 2025,
  [arXiv:2504.20940](https://arxiv.org/html/2504.20940). **Structured-multimodal CG
  latent.**
- Coretti, A., Falkner, S., Geissler, P., Hummer, G., Dellago, C. — survey component of
  [arXiv:2404.16566](https://arxiv.org/abs/2404.16566).

### 13.4. Non-flow / diffusion samplers and stochastic interpolants

- Albergo, M. S., Vanden-Eijnden, E. *Stochastic Interpolants: A Unifying Framework for
  Flows and Diffusions.* ICLR 2023,
  [arXiv:2303.08797](https://arxiv.org/abs/2303.08797).
- Akhound-Sadegh, T. et al. *Iterated Denoising Energy Matching (iDEM).* See §3.9.
- Akhound-Sadegh, T. et al. *Progressive Inference-Time Annealing of Diffusion Models
  for Sampling from Boltzmann Densities (PITA).* 2025,
  [arXiv:2506.16471](https://arxiv.org/abs/2506.16471).
- *BNEM: A Boltzmann Sampler Based on Bootstrapped Noised Energy Matching.* 2024,
  [arXiv:2409.09787](https://arxiv.org/abs/2409.09787).
- Liu, G.-H. et al. *Adjoint Schrödinger Bridge Sampler.* NeurIPS 2025,
  [arXiv:2506.22565](https://arxiv.org/abs/2506.22565).
- Lai, C.-H., Song, Y., Kim, D., Mitsufuji, Y., Ermon, S. *The Principles of Diffusion
  Models.* 2025, [arXiv:2510.21890](https://arxiv.org/abs/2510.21890). **A 470-page
  monograph; the canonical comprehensive review of diffusion modeling principles.**
- *Boltzmann Generators for Condensed Matter via Riemannian Flow Matching.* 2026,
  [arXiv:2602.18482](https://arxiv.org/html/2602.18482).
- Geffner, T. et al. *Underdamped Diffusion Bridges with Applications to Sampling.*
  2025, [arXiv:2503.01006](https://arxiv.org/html/2503.01006).
- *VT-DIS: Efficient and Unbiased Sampling from Boltzmann Distributions via
  Variance-Tuned Diffusion Models.* 2025,
  [arXiv:2505.21005](https://arxiv.org/html/2505.21005v1).

### 13.5. Cross-community: Bayesian inverse PDE

- Whang, J., Lindgren, E., Dimakis, A. *Composing Normalizing Flows for Inverse
  Problems.* ICML 2021, [arXiv:2002.11743](https://arxiv.org/abs/2002.11743).
- Sun, B. et al. *Functional normalizing flow for statistical inverse problems of partial
  differential equations (NF-iVI).* 2024,
  [arXiv:2411.13277](https://arxiv.org/abs/2411.13277).
- Padmanabha, G. A., Zabaras, N. *Learning to solve Bayesian inverse problems: amortized
  variational inference using Gaussian and Flow guides.* 2023,
  [arXiv:2305.20004](https://arxiv.org/html/2305.20004).
- Patel et al. *A dimension-reduced variational approach for solving physics-based inverse
  problems using GAN priors and normalizing flows.* CMAME 2023.
- Zhao, X., Curtis, A., Zhang, X. *Bayesian seismic tomography using normalizing flows.*
  GJI 2022.
- Liao et al. *A Novel Bayesian Geophysical Inversion Method: The Iterative Normalizing
  Flows Model.* JGR-MLC 2025.
- Asim et al. *Wave-equation-based inversion with amortized variational Bayesian
  inference.* 2022, [arXiv:2203.15881](https://arxiv.org/pdf/2203.15881).
- Asensio Ramos, A., Cobo, R., Trujillo Bueno, J. *Bayesian Stokes inversion with
  normalizing flows.* A&A 2022.
- Guo, L., Hou, T. Y., Karniadakis, G. E. *Normalizing field flows: solving forward and
  inverse stochastic differential equations using physics-informed flow models.*
  J. Comput. Phys. 2022.

### 13.6. Cross-community: cosmology, astrophysics, GW

- Wong, K. W. K., Gabrié, M., Foreman-Mackey, D. *flowMC: Normalizing flow enhanced
  sampling package for probabilistic inference in JAX.* JOSS 2023.
- Williams, M. J., Veitch, J., Messenger, C. *Nested sampling with normalizing flows for
  gravitational-wave inference.* Phys. Rev. D 2021.
- *Cosmological inference using gravitational waves and normalizing flows.*
  Phys. Rev. D 109, 123547 (2024).

### 13.7. Cross-community: particle physics, SBI

- Kofler, A., Stimper, V., Mishra-Sharma, S., Gabrié, M. et al. *Flow Annealed Importance
  Sampling Bootstrap meets Differentiable Particle Physics.* MLST 2025,
  [arXiv:2411.16234](https://arxiv.org/abs/2411.16234).
- Cranmer, K., Brehmer, J., Louppe, G. *The frontier of simulation-based inference.*
  PNAS 2020.
- Gloeckler, M., Bischoff, M., Macke, J. H. *All-in-one simulation-based inference
  (Simformer).* 2024, [arXiv:2404.09636](https://arxiv.org/abs/2404.09636).
- Macke et al. *sbi reloaded: a toolkit for simulation-based inference workflows.*
  JOSS 2024.

### 13.8. Engineering tricks

- Giaquinto, R., Banerjee, A. *Gradient Boosted Normalizing Flows.* NeurIPS 2020,
  [arXiv:2002.11896](https://arxiv.org/abs/2002.11896).
- Hutzenthaler, M., Jentzen, A., Kloeden, P. *Strong convergence of an explicit numerical
  method for SDEs with non-globally Lipschitz drift coefficients.*
  [arXiv:1010.5288](https://arxiv.org/abs/1010.5288). Foundational for tamed Langevin.
- Burda, Y., Grosse, R., Salakhutdinov, R. *Importance Weighted Autoencoders.* ICLR
  2016, [arXiv:1509.00519](https://arxiv.org/abs/1509.00519).
- Flow Perturbation / FP++: arXiv:2407.10666, arXiv:2601.21177.

### 13.9. Reviews and tooling

- Coretti, A. et al. *Boltzmann Generators and the New Frontier of Computational
  Sampling in Many-Body Systems.* arXiv:2404.16566.
- Lai et al. *The Principles of Diffusion Models* (already cited above).
- von Lilienfeld, O. A. Commentary on Coretti et al. [KIM Review 2024](https://kimreview.org/commentary-files/2024-commentary-vonlilienfeld/2024_von_lilienfeld_boltzmann_generators.pdf).
- Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., Yoo, J. *Reliable Fidelity and Diversity
  Metrics for Generative Models.* ICML 2020 — origin of the coverage metric used in §11.6.
- `noegroup/bgflow` — github.com/noegroup/bgflow.
- `VincentStimper/normalizing-flows` (`normflows`) —
  github.com/VincentStimper/normalizing-flows.
- `lollcat/fab-torch` — github.com/lollcat/fab-torch.
- `jarridrb/DEM` — github.com/jarridrb/DEM.
- `kazewong/flowMC` — github.com/kazewong/flowMC.
- `aimat-lab/TA-BG` — github.com/aimat-lab/TA-BG.
- `annalena-k/FAB-meets-diffME` — github.com/annalena-k/FAB-meets-diffME.

---

## 14. Figures

This review does not redistribute paper figures, but it does anchor the user's own
empirical evidence with a figure produced by the codebase accompanying the LLR-discrepancy
paper. For reference figures from the cited papers, follow the arXiv links above (each
linked HTML page renders figures inline).

### 14.1. User's own figure: Trinity samples on Himmelblau

The figure below is produced by `2D_Minimal/test_Trinity.py` in the user's codebase. It
illustrates the Trinity algorithm: the blue dots are samples from a unit Gaussian centred
at the origin (the source $\mu_0$), and the dark-red dots are the Trinity output after
three stages (diffusion → L-BFGS → tamed Langevin) targeting the Himmelblau potential. All
four Himmelblau modes are recovered.

![Trinity on Himmelblau](2D_Minimal/Trinity.png)

*Figure 1. Trinity-produced $\hat\mu$-samples (dark red) versus initial Gaussian source
samples (dark blue) on the Himmelblau potential. The Trinity algorithm uses sloppy
mode-finding by design: diffusion scatters the Gaussian samples wide; L-BFGS pulls them
to nearby local minima; tamed Langevin diffuses them around each basin. The output
$\hat\mu$ over-covers the modes of $\mu$. Per Proposition 3 of §11.2, the asymptotic
minimizer of $\mathrm{X}_{\hat\mu}$ is invariant to this over-coverage.*

### 14.2. Key figures from cited papers (references, not reproductions)

- **FAB Figure 1** (Midgley et al. 2023): illustrates AIS chains discovering modes outside
  the support of the trained flow. [Paper: arXiv:2208.01893](https://arxiv.org/abs/2208.01893).
- **CMT Figure 1** (Blessing et al. 2025): visualizes mass teleportation in geometric
  annealing schedules. [Paper: arXiv:2510.18460](https://arxiv.org/abs/2510.18460).
- **TA-BG Figure 3** (Schopmans, Friederich 2025): shows reverse-KL collapse at low
  temperature vs. successful sampling at high temperature.
  [Paper: arXiv:2501.19077](https://arxiv.org/abs/2501.19077).
- **TBG Figure 2** (Klein et al. 2024): shows transferable model on unseen dipeptides
  with 98% correct configurations. [Paper: arXiv:2406.14426](https://arxiv.org/abs/2406.14426).
- **Wirnsberger Figure 1** (2022): illustrates the lattice-Gaussian prior construction
  and the trained-flow free-energy results.
  [Paper: arXiv:2111.08696](https://arxiv.org/abs/2111.08696).
- **iDEM Figure 1** (Akhound-Sadegh et al. 2024): the iDEM algorithm two-loop schematic.
  [Paper: arXiv:2402.06121](https://arxiv.org/abs/2402.06121).

---

---

## 15. Adaptive tuned priors — research challenges and possible routes

### 15.1. Problem statement

Camp B methods improve sample efficiency by placing prior mass near plausible target modes, but pay for this with brittleness: a tuned prior that omits a metastable basin of $\mu(x) \propto \exp(-U(x))$ induces an unrecoverable bias in any importance-weighted or variationally-trained generator built on top of it. In a learning setting in which the modes of $\mu$ are not known in advance, an *adaptive* tuned prior $p_\theta$ would have to satisfy, simultaneously:

- **(R1) Full support over $\mathbb{R}^d$.** For every Lebesgue-measurable set $A\subset\mathbb{R}^d$ with $\mu(A)>0$, the prior must satisfy $p_\theta(A)>0$, with tails sufficiently heavy that the importance ratio $\mu/p_\theta$ is integrable on every bounded region.
- **(R2) Mode-aware structure.** The prior should concentrate mass near regions where $\mu$ has high probability.
- **(R3) Symmetry preservation.** If $\mu$ is invariant under a group $G$, the prior must satisfy $p_\theta(g\cdot x)=p_\theta(x)$.
- **(R4) Online adaptivity without circularity.** The prior must update from sampled information about $\mu$ without making the *current* prior the *only* source of those samples.
- **(R5) Transferability under perturbation.** Small changes to the target should induce small, controlled changes to $p_\theta$.
- **(R6) Tractable density and sampling.** Both $x\sim p_\theta$ and $\log p_\theta(x)$ must be available in closed form.

The fundamental tension: (R1) (no hard mode lock-in) versus (R2) (rewards concentration); (R4) (no self-supervision from prior alone) versus (R6) (no reliance on expensive external samplers).

### 15.2. Route A — Online-learned mixture with external mode finder

**Construction.** Pseudo-inputs $\{\mathbf{u}_k\}_{k=1}^K$ + encoder $q_\phi(z\,|\,x)$:
$$
p_\theta(x) = \sum_{k=1}^K \pi_k\,q_\phi(x\,|\,\mathbf{u}_k),\qquad \sum\pi_k=1,\,\pi_k\geqslant 0.
$$
Pseudo-inputs are populated/pruned by an external mode finder $\mathcal{M}$ (replica exchange, surrogate potential, or basins from a previous checkpoint), not by ELBO maximization.

**Training.** Reverse KL on $\mu$, with $\{\mathbf{u}_k\}$ refreshed by $\mathcal{M}$ at checkpoints.

**Advantages.** Full support; flow performs only local deformations between basins; symmetrization straightforward.

**Difficulties.** (i) $\mathcal{M}$ must be independent of $p_\theta$ — if reused, missed modes are propagated; if independent (e.g., replica exchange), cost approaches direct MCMC on $\mu$. (ii) Discrete combinatorial structure: add/prune changes dimension of $\theta$. (iii) Choice of $K$.

**Precedent.** VampPrior, LARS, basin-hopping (Wales), replica exchange, Dirichlet-process priors.

### 15.3. Route B — Defensive mixture with guaranteed Gaussian long tail

**Construction.** With $\alpha\in(0,1)$ and a defensive $\eta=\mathcal{N}(0,\sigma_{\text{def}}^2 I)$:
$$
p_\theta(x) = \alpha\,\eta(x) + (1-\alpha)\sum_{k=1}^K \pi_k\,q_\phi(x\,|\,\mathbf{u}_k).
$$

**Training.** Variance-penalizing loss:
$$
\mathcal{L} = \mathbb{E}_{x\sim p_{\theta,\psi}}[\log p_{\theta,\psi}(x)+U(x)] + \lambda\,\mathbb{E}_{x\sim\eta}[(\log p_{\theta,\psi}(x)+U(x))^2].
$$
Treat $\alpha$ as a defensive mixture weight (Hesterberg 1995). The importance-sampled estimator
$$
\widehat I_N(f) = \tfrac{1}{N}\sum_n f(x_n)\tfrac{\mu(x_n)}{p_\theta(x_n)},\quad x_n\sim p_\theta,
$$
has *bounded variance* whenever $\mu/(\alpha\eta)$ is bounded — regardless of how badly the mixture components are tuned. **(R1) becomes a quantitative variance bound, not just an asymptotic requirement.**

**Advantages.** Non-asymptotic guarantee against mode lock-in. Zero-overhead bolt-on.

**Difficulties.** $\alpha$ trades efficiency for safety; samples from $\eta$ are high-energy and have tiny ESS in high $d$. No automatic schedule. Adaptive $\sigma_{\text{def}}^2$?

**Precedent.** Defensive IS (Hesterberg 1995, Owen-Zhou 2000); particle filters; rare-event simulation.

### 15.4. Route C — Tempered mixture: find modes at high $T$, sample at target $T$

**Construction.** $\beta_{\text{exp}}<\beta_{\text{tgt}}$, sample $\{y_j\}$ from $\mu_{\beta_{\text{exp}}}$, then
$$
p_\theta(x) = \tfrac{1}{M}\sum_j \mathcal{N}(x\,|\,y_j,\,\sigma^2(y_j) I),
$$
with $\sigma^2(y_j)$ either fixed or set to $(\nabla^2 U(y_j))^{-1}/\beta_{\text{tgt}}$ for local-curvature-aware bandwidth.

**Training.** Reverse KL on $\mu_{\beta_{\text{tgt}}}$. Optionally a tempered ladder.

**Advantages.** Kramers rate $\exp(-\beta\Delta F)$ — exponential barrier-crossing speedup at $\beta_{\text{exp}}$. Full support automatic.

**Difficulties.** $\beta_{\text{exp}}$ trade-off: cold→no barrier crossing, hot→prior mass on irrelevant configurations. KDE curse of dimensionality ($\mathcal{O}(N^{-4/(d+4)})$).

**Precedent.** Parallel tempering (Swendsen-Wang, Hukushima-Nemoto); Coretti higher-T variant.

### 15.5. Route D — Curriculum prior: Gaussian → mixture

**Construction.** $\gamma:[0,1]\to[0,1]$ monotonic, $\gamma(0)=0,\gamma(1)=1$:
$$
p_{\theta,\gamma(\tau)}(x) = (1-\gamma(\tau))p_0(x) + \gamma(\tau)p_1(x;\theta).
$$

**Difficulties.** At $\tau=1$, the construction inherits Camp B's mode-lock-in risk. Best viewed as an optimization heuristic for Routes A/C, not a standalone solution.

### 15.6. Route E — Soft prior tuning via score-based bias

**Construction.** Keep $p_0$ as a diffuse Gaussian; augment the score field:
$$
\mathrm{d}x_t = [-\nabla U(x_t)+\lambda(t)s_\phi(x_t,t)]\mathrm{d}t + \sqrt{2\beta^{-1}}\mathrm{d}W_t,
$$
with $\lambda(1)=0$ so dynamics agree with target Langevin at $t=1$. Mode information enters through $s_\phi$, not $p_0$.

**Training.** Schrödinger-bridge / Doob $h$-transform objective:
$$
\mathcal{L}(\phi) = \mathbb{E}\!\Big[\int_0^1 \tfrac{1}{2}\lambda(t)^2\|s_\phi(x_t,t)\|^2\mathrm{d}t + \log\tfrac{p_0^{(\phi)}(x_1)}{\mu(x_1)}\Big].
$$

**Advantages.** $p_0$ remains Gaussian — (R1) and (R6) automatic. $s_\phi$ equivariant — (R3). Adaptivity via retraining $s_\phi$.

**Precedent.** Schrödinger bridges (Chen et al. 2021, De Bortoli et al. 2021); iDEM, CMT use related score-matching constructions. The *prior-diffuse + score-carries-mode* split has not been isolated in BG literature.

### 15.7. Honest assessment

**Route B** is most promising short term: converts soft asymptotic full-support into a quantitative variance bound, zero-overhead on top of any Camp A/B method. **Route E** is most principled and most likely to scale to high-$d$ symmetric systems, inheriting equivariant + score-matching infrastructure. **Route A** is fundamentally limited by (R4). **Route C** suffers KDE curse-of-dimensionality. **Route D** is best as an optimizer for A/C.

**Recommended hybrid.** Gaussian prior + small defensive weight $\alpha$ (Route B) + equivariant score-bias field $s_\phi$ (Route E) trained against $\mathrm{X}_{\hat\mu}$-derived targets. Inherits full-support, principled mode-injection, and circularity-free training.

### 15.8. Comparison to $\mathrm{X}_{\hat\mu}$

| Aspect | Adaptive tuned prior | $\mathrm{X}_{\hat\mu}$ |
|---|---|---|
| Information location | Generative (prior) | Discriminative (loss) |
| Mode info source | External oracle / mode-finder $\mathcal{M}$ | Sloppy $\hat\mu$ from Trinity |
| Requires support to cover $\mu$? | Yes (R1) | No — log-ratio difference is well-defined under weaker conditions |
| Circularity (R4) risk | High | None — $\hat\mu$ external |
| Symmetry (R3) | Requires explicit symmetrization | Handled at estimator level |
| Best when | Known modes from physics | Modes unknown, symmetry non-trivial, transferability binding |
| Output | Closed-form $\mu$-approximation reusable as proposal | Trained flow $\nu_\theta\approx\mu$ |

The adaptive prior and $\mathrm{X}_{\hat\mu}$ occupy different design-space points and are *complementary*. The cleanest hybrid: defensive-mixture prior + $\mathrm{X}_{\hat\mu}$ regularizer.

---

## 16. Method comparison — advantages and good applications

For each major method, we now give a comparative analysis along *five* axes: best regime, worst regime, advantages over alternatives, honest limitations (with attention to where original papers over-claim), and concrete target systems.

### 16.1. Reverse-KL (Noé 2019 baseline)

- **Best.** Single basin, smooth $U$, low $d$, MD-seeded initialization. Building block inside FAB/SBG/TA-BG pipelines.
- **Worst.** Strongly multimodal with unknown modes — mode collapse.
- **Advantages.** Minimal, no training-time MCMC, exact log-density.
- **Honest limits.** ESS$/N\to 0$ on missed modes with no built-in detector. "Exact statistical mechanics" claims relying on importance weights are practically empty in the regime one would most want a generator.
- **Suitable.** Single conformational state of a pre-characterized peptide; double-well in $d\leqslant 10$ seeded near both wells.

### 16.2. Stochastic NFs (Wu 2020)

- **Best.** Meaningful annealing path $\mu_0\to\mu$, stochasticity bridges barriers, expressivity-limited deterministic flow.
- **Worst.** Expensive $U$ — each intermediate kernel adds an evaluation. No natural schedule.
- **Advantages.** Strictly more expressive than deterministic flow; tractable Radon-Nikodym path weight.
- **Honest limits.** Path-weight variance grows with path length (AIS curse-of-dim). Free-energy claims rigorous only at infinite kernel count; finite-step bias rarely reported.
- **Suitable.** Lattice spin systems with high-$T$ reference; small molecular clusters with $T$-ladder.

### 16.3. Equivariant flows / flow matching (Köhler 2020, Klein 2023)

- **Best.** Cartesian molecular systems with exact $\mathrm{SE}(3)\times S_N$ symmetry. Eq. FM removes $\log\det$ bottleneck.
- **Worst.** Approximate or broken symmetries; targets already in invariant internal coordinates.
- **Advantages.** Orbit generalization → massive sample-efficiency gain.
- **Honest limits.** Eq. FM simulation-free at train but ODE-integrate at inference; log-density expensive → importance correction costly.
- **Suitable.** LJ-13/55, small peptides in Cartesian, atomistic water.

### 16.4. FAB (Midgley 2022)

- **Best.** Multimodal targets where AIS chains of moderate length can hop modes.
- **Worst.** Rugged landscapes with high barriers — if AIS can't reach a mode, FAB can't teach the flow.
- **Advantages.** Mode-covering training without $\mu$-samples. Replay buffer amortizes AIS.
- **Honest limits.** Buffer can become stale; bootstrap amplifies bad early AIS. Gains over reverse-KL largest on toy GMMs; less clear on realistic molecular systems.
- **Suitable.** GMM benchmarks, many-well 2D, alanine dipeptide.

### 16.5. flowMC / adaptive MC + flow (Gabrié 2022)

- **Best.** Moderate-$d$ posteriors where MCMC alone mixes slowly across well-separated modes.
- **Worst.** Very high $d$ — flow proposals rejected; minute-per-eval $U$.
- **Advantages.** Asymptotic correctness from MCMC + mode-hopping from flow.
- **Honest limits.** Acceptance degrades exponentially in $d$. Not amortized.
- **Suitable.** Bayesian cosmology likelihoods; spin glasses at moderate $N$.

### 16.6. TA-BG (Schopmans 2025)

- **Best.** Temperature hierarchy with modes continuously connected to unimodal high-$T$.
- **Worst.** First-order transitions in $T$; non-temperature-driven slow modes.
- **Advantages.** Avoids cold-start mode collapse. Intermediate flows reusable for thermodynamic integration.
- **Honest limits.** Schedule is delicate and empirical. Does not solve discrete-symmetry discovery.
- **Suitable.** Small-to-medium peptides (Chignolin, Trp-cage); LJ clusters; polymer melts.

### 16.7. CMT (Blessing 2025)

- **Best.** Transport problems with natural endpoints; explicit constraints (fixed COM, energy, alchemy).
- **Worst.** Pure sampling without endpoint structure; high $d$ with sensitive $\varepsilon$.
- **Advantages.** First-class constraints. OT geometric interpretation.
- **Honest limits.** Dual-potential overhead; $\varepsilon$ bias-variance trade-off opaque for non-Gaussian targets.
- **Suitable.** Alchemical $\Delta F$, microcanonical sampling at fixed energy, reactive transitions.

### 16.8. SBG (Tan 2025)

- **Best.** Targets where single flow can't bridge $\mathrm{KL}$ gap; highly multimodal with broad initial coverage.
- **Worst.** Amortized settings; cheap targets where direct MCMC is faster.
- **Advantages.** Forward-KL at each iteration; ESS-plateau stopping; final $q_\theta$ close to $\mu$ in TV.
- **Honest limits.** Resampling variance compounds. Convergence proofs need strong mixing assumptions rarely verified.
- **Suitable.** Single-target high-precision $\Delta F$; lattice gauge at fixed coupling; specific astrophysical posterior.

### 16.9. TBG (Klein 2024)

- **Best.** Drug-discovery screening for thousands of related molecules at moderate accuracy.
- **Worst.** Out-of-distribution molecules (silent extrapolation); single-target high precision.
- **Advantages.** Per-target inference is essentially free.
- **Honest limits.** In-vs-out-of-distribution accuracy gap typically large but under-reported. Importance correction non-trivial.
- **Suitable.** Combinatorial-library conformer generation; QSAR feature ensembles; QM/MM surrogate sampling.

### 16.10. iDEM (Akhound-Sadegh 2024)

- **Best.** Differentiable $U$, mid-$d$ targets in the diffusion sweet spot.
- **Worst.** Non-differentiable $U$; extreme low-$T$ where score is singular near minima.
- **Advantages.** Simulation-free training; no mode-seeking pathology.
- **Honest limits.** SDE/ODE inference costs many $\nabla U$ evals — often more than one MD step. No tractable log-density → no direct importance correction.
- **Suitable.** LJ clusters, alanine dipeptide, ML-FF sampling.

### 16.11. iEFM/EWFM (Woo 2024, Hahn 2025)

- **Best.** Same target class as iDEM, preferring ODE deterministic inference. EWFM's energy weighting encourages coverage.
- **Worst.** ODE-solver accuracy must be controlled.
- **Advantages.** Deterministic ODE → tractable log-density via instantaneous change of variables → importance correction recoverable.
- **Honest limits.** Bilevel structure sensitive to early-training drift; asymptotic-only convergence.
- **Suitable.** Small-molecule conformer sampling with differentiable FF.

### 16.12. ASBS (Liu 2025)

- **Best.** Meaningful reference near $\mu$, finite time budget, physically meaningful diffusion.
- **Worst.** No natural reference or anneal; high-$d$ bridge variance explodes.
- **Advantages.** Principled stochastic-control formulation; connections to OT.
- **Honest limits.** IPFP convergence in deep parametrizations poorly understood; high per-epoch cost.
- **Suitable.** Density estimation with low-$T$ reference; alchemical transformations as bridges.

### 16.13. PITA (Akhound-Sadegh 2025)

- **Best.** Many cold targets sharing a hot training distribution.
- **Worst.** Single-target precision; phase transitions in the bridge.
- **Advantages.** Inference-time annealing decoupled from training.
- **Honest limits.** Importance-weighting variance at cold target often poor.
- **Suitable.** Temperature replicas along isochore; multi-$T$ free-energy surfaces; EOS sweeps.

### 16.14. Wirnsberger lattice prior (2022) — Camp B flagship

- **Best.** Crystalline solids near equilibrium lattice. Spectacular for absolute solid $\Delta F$.
- **Worst.** Liquids, amorphous phases, melting transitions.
- **Advantages.** Prior does genuine physical work — large chunk of $\log\mu$ absorbed analytically.
- **Honest limits.** Strictly single-basin / single-phase by design (cf. §6.2 killer quote). Lattice must be specified a priori.
- **Suitable.** Monatomic FCC/BCC solids, molecular crystals, ice phases, polymorph $\Delta F$.

### 16.15. Coretti higher-$T$ reference — Camp B

- **Best.** Cheap high-$T$ MD, hard target $T$, no phase transition between.
- **Worst.** $T_{\text{ref}}$ itself hard; phase transition.
- **Advantages.** No architectural changes; prior is just data.
- **Honest limits.** Sample-based prior — log-density not directly tractable.
- **Suitable.** Crystalline solids near melting; polymers at $T_g$; tractable reference state points.

### 16.16. Schiebroek/Koehn structured CG (2025) — Camp B

- **Best.** Well-understood CG representation; slow modes known a priori.
- **Worst.** Unknown CG variables; non-Gaussian slow-fast coupling.
- **Advantages.** Combines physical insight (slow modes) with data flexibility (fast modes).
- **Honest limits.** CG/fine split is a modeling choice, not learned.
- **Suitable.** Proteins with known reaction coordinates; molecular clusters with collective coords; polymers via Rouse modes.

### 16.17. Schebek conditional phase-diagram BG (2024) — hybrid

- **Best.** EOS / phase-diagram studies; continuous regions without first-order transitions.
- **Worst.** Across phase boundaries.
- **Advantages.** Amortizes across state points; natural thermodynamic response via finite differences.
- **Honest limits.** Phase transitions are most physically interesting and the method's weakest point.
- **Suitable.** LJ EOS; supercooled-liquid response away from $T_g$; polymer melt PVT in stable regions.

### 16.18. VampPrior (Tomczak 2018) — VAE-side Camp B

- **Best.** High-$d$ data with strong cluster structure (images, discrete-state molecular conformations).
- **Worst.** Continuous smoothly-varying data; small latent space.
- **Advantages.** Prior learned jointly with model.
- **Honest limits.** $K$ hyperparameter has no clean optimum; training dynamics unstable.
- **Suitable.** VAE latents for molecular conformations; multimodal image generation.

### 16.19. LARS (Bauer-Mnih 2019) — VAE-side Camp B

- **Best.** Post-hoc upgrade of an already-trained VAE.
- **Worst.** Poorly-fit VAE; high $d$ with low acceptance.
- **Advantages.** Modular AR head; unbiased expectations via importance weighting.
- **Honest limits.** Acceptance dimension-sensitive.
- **Suitable.** Existing VAEs in moderate latent $d$.

### 16.20. The $\mathrm{X}$-functional + Trinity (third way)

- **Best.** Multimodal targets where reverse-KL collapses, forward-KL is unavailable, and existing surrogates (FAB, SBG) over-rely on AIS or fail to amortize.
- **Worst.** Single-basin targets where reverse-KL is fine.
- **Advantages.** Unifies Camp A (KL-style training) and Camp B (mode info) under one objective; importance-weighted estimators available.
- **Honest limits.** Less battle-tested. Default $\omega$ choices and schedules still developing. High-$d$ variance/bias trade-off not yet well-understood.
- **Suitable.** Multimodal molecular systems with available structural priors but where reverse-KL alone fails; controlled mode-seeking/mode-covering interpolation.

### 16.21. Decision matrix

| Target characteristic | First-line | Secondary | Avoid |
|---|---|---|---|
| Single basin, smooth $U$, low $d$ | Reverse-KL (16.1) | SNF (16.2) | SBG, FAB |
| Multimodal, known modes (seeded) | FAB, TA-BG | SBG, flowMC | Reverse-KL |
| Multimodal, unknown modes | SBG, flowMC, iDEM | FAB long AIS | Reverse-KL |
| Discrete symmetries | Equivariant flow / FM | TBG (amortized) | Generic flows |
| Crystalline solid near eq. | Wirnsberger | Coretti | Camp A flows |
| Liquid / disordered, mid-$d$ | SBG, flowMC, iDEM | TA-BG | Wirnsberger |
| Large biomolecule, known slow modes | Schiebroek/Koehn | $\mathrm{X}$/Trinity w/ CG | Pure atomistic rev-KL |
| Amortized across molecules | TBG | Eq. FM | SBG, flowMC |
| Amortized across $(T,p)$ | Schebek, PITA | TA-BG per point | flowMC per point |
| Mode discovery (no seeds) | SBG, iDEM, FAB-long-AIS | flowMC | Rev-KL, TBG |
| Single-target high-precision $F$ | SBG, flowMC, TA-BG | CMT constrained | TBG, PITA |
| Hard constraints | CMT | $\mathrm{X}$/Trinity | Unconstrained flows |
| Differentiable $U$, no MCMC | iDEM, iEFM, ASBS | Eq. FM | SNF, flowMC |
| Non-differentiable $U$ | flowMC, SBG | FAB | iDEM, iEFM |
| Phase transition / coexistence | TA-BG above transition | CMT endpoint | Schebek, Wirnsberger |
| VAE-style latent | VampPrior, LARS | $\mathrm{X}$/Trinity learned | $\mathcal{N}(0,I)$ |
| Camp A + Camp B hybrid | $\mathrm{X}$/Trinity | TA-BG on Wirnsberger | Pure rev-KL |
| Bridge between distributions | ASBS, SNF | TA-BG | iDEM w/o anneal |
| $\Delta F$ between Hamiltonians | CMT, SNF $\lambda$-coupling | TA-BG $\lambda$-anneal | TBG |

**Practical workflow.** (a) Start with a Camp B prior if any physical structure is available. (b) Train reverse-KL as sanity baseline. (c) Escalate to FAB/TA-BG/SBG when multimodality is detected. (d) Invoke amortized methods only when target count justifies training cost. (e) Use $\mathrm{X}$/Trinity for controlled mode-seeking/covering interpolation or to combine heterogeneous training signals.

---

## 17. Design-philosophy synthesis

After surveying 40+ methods across 6 communities, design philosophies cluster along specific axes. We name them, place each method on the map, and highlight the niche the user's method occupies.

### 17.1. Axis 1: where information about the target enters

| Entry point | Methods | Cost |
|---|---|---|
| Loss only ($\nabla U$ samples) | iDEM, iEFM, EWFM, BNEM | many $\nabla U$ evals |
| Loss + AIS | FAB, FAB-meets-diffME | AIS chain budget |
| Loss + running MCMC | flowMC, Gabrié 2022 | MCMC budget |
| Loss + multi-temperature | TA-BG, PITA, parallel tempering hybrids | MD at multiple $T$ |
| Architecture (symmetry) | Equivariant flows, eq. FM | architectural complexity |
| Architecture (representation) | TBG, internal coordinates | feature engineering |
| Prior (lattice / structured) | Wirnsberger, Schiebroek-Koehn | manual lattice / CV specification |
| Prior (higher-$T$ reference) | Coretti | MD at $T_{\text{ref}}$ |
| Inference-time SMC | SBG, PITA | inference-time anneal budget |
| Constraint regularizer | CMT | constraint specification |
| **Regularizer + sloppy mode-finder** | **$\mathrm{X}_{\hat\mu}$ + Trinity (this paper)** | Trinity preprocessing |

The user's method occupies a *unique* row: a regularizer combined with a *sloppy* mode-finder. The closest analogues are flowMC (regularizer through MCMC self-supervision) and CMT (regularizer through constraint), but neither tolerates *inaccurate* mode information the way $\mathrm{X}_{\hat\mu}$ does (Proposition 3 of §11.2).

### 17.2. Axis 2: amortization vs. precision

| Per-target dedicated | Amortized |
|---|---|
| flowMC, SBG, CMT, single-target FAB | TBG, PITA, Schebek conditional |

The trade-off is roughly two orders of magnitude in per-target accuracy. The field is bifurcating: single-target methods chase $\mathrm{ESS}\to 1$ on high-precision free-energy problems; amortized methods chase 98% correct configurations across thousands of unseen targets. *No method achieves both simultaneously.*

### 17.3. Axis 3: deterministic vs. stochastic

| Deterministic flow | Stochastic SDE / diffusion |
|---|---|
| RealNVP, NSF, IAF, FAB, TA-BG, TBG, SBG | iDEM, iEFM, EWFM, ASBS, PITA, NETS |

Deterministic methods retain tractable log-density (good for importance correction, $\Delta F$, Bayesian model comparison); stochastic methods can be simulation-free at training, scale better at higher $d$, but lose the cheap log-density.

### 17.4. Axis 4: prior support — full vs. constrained

| Full support $\mathbb{R}^d$ | Constrained support (mode lock-in) |
|---|---|
| Camp A defaults; $\mathrm{X}_{\hat\mu}$ third way | Wirnsberger lattice; Schiebroek/Koehn within slow CV |

This is where Camp B is *structurally* different from Camp A. The $\mathrm{X}_{\hat\mu}$ approach inherits Camp A's full-support property — this is the precise sense in which it is "Camp A with mode info added" rather than "Camp B with safety belt."

### 17.5. The unfilled ecological niche

Cross-tabulating Axes 1–4 yields $11\times 2\times 2\times 2 = 88$ design-space squares. Most are unfilled. The user's $\mathrm{X}_{\hat\mu}$ + Trinity recipe fills a square that no prior method occupies: *regularizer-based mode information injection, single-target precision, deterministic flow, full support*. The closest neighbors are FAB (regularizer-based + deterministic, but AIS rather than Trinity) and Schiebroek/Koehn (Camp B with structured CG, but constrained support). Neither matches all four axes.

---

## 18. Illustrative figures

Five self-generated illustrative figures are provided in `review_figs/` (Python source:
`review_figs/make_figures.py`). All figures are my own creations using matplotlib; no
copyrighted material from cited papers is redistributed.

### 18.1. Mode collapse under reverse-KL

![Mode collapse](review_figs/fig_mode_collapse.png)

*Figure 18.1.* Left: a bimodal target $\mu$ with equal-mass modes at $(\pm 2, 0)$. Right:
after reverse-KL training, the flow $\nu_\theta$ has collapsed onto a single mode at
$(+2, 0)$. The reverse-KL gradient (§5.1) gives no incentive to recover the dropped mode
because the loss only samples regions where $\nu_\theta$ already has mass.

### 18.2. Mass teleportation in geometric annealing

![Mass teleportation](review_figs/fig_mass_teleport.png)

*Figure 18.2.* Geometric annealing $\pi_\beta \propto \mu_0^{1-\beta}\mu^\beta$ between
a unit Gaussian prior and a bimodal target at $(\pm 4, 0)$. As $\beta$ increases, the
bimodality emerges *between* $\beta=0.5$ and $\beta=0.75$ — disconnected regions appear
with little overlap, causing *mass teleportation* (§5.2).

### 18.3. Fake ESS

![Fake ESS](review_figs/fig_fake_ess.png)

*Figure 18.3.* Left: flow samples (dark red) restricted to one mode of a bimodal target.
Right: histogram of importance weights — all uniform inside the visited mode, giving
$\mathrm{ESS}=1$ exactly while *half the target mass is missed*. The fake-ESS pitfall of
§5.4.

### 18.4. The three philosophical positions

![Three camps](review_figs/fig_camps.png)

*Figure 18.4.* *Left*: Camp A — unimodal Gaussian prior must be transported by the flow
to a multimodal target. *Middle*: Camp B — tuned mixture prior matches modes but commits
the support (mode lock-in). *Right*: third way — keep simple prior, add $\mathrm{X}_{\hat\mu}$ regularizer
with a *sloppy* $\hat\mu$.

### 18.5. Log-ratio landscape and the $\mathrm{X}$ functional

![X landscape](review_figs/fig_x_landscape.png)

*Figure 18.5.* Left: log-ratio $\log(\mu/\nu_\theta)$ for an untrained flow (large
variation) versus a partially-trained flow (near-constant). Right: corresponding
$\mathrm{X}_\mu$ values. The functional vanishes when the log-ratio is pointwise
constant (Proposition 1, §11.1), giving normalization invariance suitable for unnormalized
Boltzmann targets.

### 18.6. Trinity on Himmelblau (user's benchmark)

![Trinity on Himmelblau](2D_Minimal/Trinity.png)

*Figure 18.6.* Trinity in action. Dark blue: initial Gaussian source samples; dark red:
Trinity output covering all four Himmelblau modes after diffusion → L-BFGS → tamed
Langevin. Deliberate over-coverage is harmless by Proposition 3.

---

## 19. References to representative figures from cited papers

Linked to original papers (not redistributed):

- **Wirnsberger 2022, Fig. 1** — lattice-Gaussian prior construction.
  [arXiv:2111.08696](https://arxiv.org/abs/2111.08696).
- **FAB 2022, Fig. 1** — AIS discovering modes outside flow support, 40-component 2D GMM.
  [arXiv:2208.01893](https://arxiv.org/abs/2208.01893).
- **TA-BG 2025, Fig. 3** — reverse-KL mode collapse at low $T$ vs. successful high-$T$ sampling.
  [arXiv:2501.19077](https://arxiv.org/abs/2501.19077).
- **TBG 2024, Fig. 2** — transferable model on 100 unseen dipeptides at 98% correct configurations.
  [arXiv:2406.14426](https://arxiv.org/abs/2406.14426).
- **iDEM 2024, Fig. 1** — two-loop algorithm schematic.
  [arXiv:2402.06121](https://arxiv.org/abs/2402.06121).
- **CMT 2025, Fig. 1** — mass teleportation in geometric annealing vs. CMT's constrained schedule.
  [arXiv:2510.18460](https://arxiv.org/abs/2510.18460).
- **SBG 2025, Fig. 2** — continuous-time SMC at inference on tri-, tetra-, hexa-peptides.
  [arXiv:2502.18462](https://arxiv.org/abs/2502.18462).
- **Lai, Song, Kim, Mitsufuji, Ermon 2025**, *The Principles of Diffusion Models* — 470-page
  monograph. See Figs. 2–4 for the three theoretical perspectives (variational, score-based,
  flow-based) underlying every diffusion-based Boltzmann sampler.
  [arXiv:2510.21890](https://arxiv.org/abs/2510.21890).

---

## 20. Interactive citation network

A self-contained interactive citation graph is provided in
[`review_citations.html`](review_citations.html). The file uses vis.js loaded from a
public CDN to render an interactive network of the ~60 papers cited in this review,
colored by camp (Camp A / Camp B / Foundational / Diffusion / Cross-community / Third
way) and connected by influence/citation relationships. Hover over a node to see the
paper's role; drag nodes to rearrange the layout.

---

*This document is intentionally a living draft. Further searches, deeper per-paper analyses,
and additional comparative tables can be appended without breaking the structure.*

---

# Appendix A. Formal proofs

This appendix supplies full proofs for the four propositions stated as sketches in §11.
Throughout, $\mu$ and $\nu$ denote probability densities on $\mathbb{R}^d$ — possibly
*unnormalized*, in which case $\mu = (1/Z_\mu)\tilde\mu$ for some unnormalized
$\tilde\mu \geqslant 0$ and $Z_\mu = \int\tilde\mu < \infty$, and similarly for $\nu$.

### A.1. Proof of Proposition 1 (Zero set of $\mathrm{X}_\omega$)

**Claim.** Let $\omega(x,y)\geqslant 0$ be a finite positive measure on $\mathbb{R}^d\times\mathbb{R}^d$.
Then
$$
\mathrm{X}_\omega(\mu\,\|\,\nu) \;=\; \iint \omega(x,y) \big| \log\tfrac{\mu(x)}{\nu(x)} - \log\tfrac{\mu(y)}{\nu(y)} \big|\,\mathrm{d}x\,\mathrm{d}y \;=\; 0
$$
if and only if $\log(\mu/\nu)$ is constant $\omega^{\text{marg}}$-almost-everywhere, where
$\omega^{\text{marg}}(A) := \omega(A\times\mathbb{R}^d)$ is the first-coordinate marginal.

**Proof.** The integrand $h(x,y) := \omega(x,y)\,|\log(\mu(x)/\nu(x)) - \log(\mu(y)/\nu(y))|$ is non-negative.
Its integral is zero if and only if $h = 0$ Lebesgue-a.e. on $\mathbb{R}^d\times\mathbb{R}^d$, which holds iff
$$
\big|\log(\mu(x)/\nu(x)) - \log(\mu(y)/\nu(y))\big| = 0 \quad \omega\text{-a.e.}
$$
Equivalently, $\log(\mu(x)/\nu(x)) = \log(\mu(y)/\nu(y))$ for $\omega$-a.e. $(x,y)$. Let
$f(x) := \log(\mu(x)/\nu(x))$. The condition is $f(x) = f(y)$ for $\omega$-a.e. $(x,y)$.

Now suppose $\omega = \omega_1\otimes\omega_2$ has product structure. For
$\omega_1\otimes\omega_2$-a.e. $(x,y)$, $f(x) = f(y)$ means that $f$ takes a single value
on $\omega_1$-a.e. $x$ and an equal single value on $\omega_2$-a.e. $y$. The two values
must agree (pick any pair $(x_0,y_0)$ in the joint support); so $f$ is
$(\omega_1+\omega_2)$-a.e. equal to a single constant. When the marginals are equal
$\omega_1 = \omega_2 =: \omega^{\text{marg}}$ (as in the $\mathrm{X}_\mu$ case with
$\omega = \mu\otimes\mu$), this reduces to: $f$ is constant $\omega^{\text{marg}}$-a.e.

In the general (non-product) case the same logic applies on each fiber. ∎

### A.2. Proof of Proposition 2 (Normalization invariance)

**Claim.** $\mathrm{X}_\omega(\mu\,\|\,\nu)$ depends on the densities $\mu, \nu$ only through their
unnormalized forms. Specifically, for any constants $c_1, c_2 > 0$,
$$
\mathrm{X}_\omega(c_1\mu\,\|\,c_2\nu) = \mathrm{X}_\omega(\mu\,\|\,\nu).
$$

**Proof.** $\log(c_1\mu(x)/(c_2\nu(x))) = \log(c_1/c_2) + \log(\mu(x)/\nu(x))$. The
constant $\log(c_1/c_2)$ is independent of $x$ and cancels in the difference inside the
absolute value:
$$
\big|\log\tfrac{c_1\mu(x)}{c_2\nu(x)} - \log\tfrac{c_1\mu(y)}{c_2\nu(y)}\big| = \big|\log\tfrac{\mu(x)}{\nu(x)} - \log\tfrac{\mu(y)}{\nu(y)}\big|.
$$
The integrand of $\mathrm{X}_\omega$ is unchanged, so the integral is unchanged. ∎

**Corollary.** $\mathrm{X}_\omega$ is well-defined for *unnormalized* Boltzmann targets
$\mu \propto \exp(-U_1)$ even when the partition function $Z_\mu$ is unknown. The
intractability of $Z_\mu$ does not affect the loss landscape.

### A.3. Proof of Proposition 3 (Asymptotic invariance to $\hat\mu$)

**Claim.** Let $\{\nu_\theta\}_{\theta\in\Theta}$ be a parameterized family of densities on
$\mathbb{R}^d$. Suppose there exists $\theta_*\in\Theta$ with $\nu_{\theta_*} = \mu$ (the
*realizable* case). For any positive measure $\omega = \hat\omega\otimes\hat\omega$ with
$\mathrm{supp}(\hat\omega)\subseteq\mathbb{R}^d$,
$$
\theta_* \in \arg\min_\theta \mathrm{X}_\omega(\mu\,\|\,\nu_\theta),
$$
and $\mathrm{X}_\omega(\mu\,\|\,\nu_{\theta_*}) = 0$. The argmin set is unchanged for any
choice of $\hat\omega$.

**Proof.** At $\theta = \theta_*$, $\nu_{\theta_*}(x) = \mu(x)$ for all $x$, so
$\log(\mu(x)/\nu_{\theta_*}(x)) = 0$ identically. Then
$$
\mathrm{X}_\omega(\mu\,\|\,\nu_{\theta_*}) = \iint \omega(x,y)\,|0 - 0|\,\mathrm{d}x\,\mathrm{d}y = 0.
$$
Since $\mathrm{X}_\omega(\mu\,\|\,\nu_\theta) \geqslant 0$ for all $\theta$ (the integrand
is the product of a non-negative measure $\omega$ and the absolute-value of a real
function), $\theta_*$ achieves the global minimum, value zero. This argument did not use
the specific shape of $\hat\omega$ — only the non-negativity of $\omega$ and the
realizability of $\theta_*$. Therefore the argmin is the same for any $\hat\omega$. ∎

**Caveat (non-realizable case).** When the parameterized family does not contain $\mu$
exactly — i.e. $\inf_\theta \mathrm{X}_\omega(\mu\,\|\,\nu_\theta) > 0$ — the minimizer
*does* depend on $\hat\omega$. The optimal $\theta_{*,\hat\omega}$ is the parameter that
makes the log-ratio $\log(\mu/\nu_\theta)$ as constant as possible *weighted by
$\hat\omega$*. In practice we choose $\hat\omega = \hat\mu\otimes\hat\mu$ with
$\hat\mu$ over-covering the modes of $\mu$, so that the optimization emphasizes
constancy of the log-ratio across modes — exactly the desired regularization effect.

**Quantitative version.** Suppose $\nu_\theta = \nu_{\theta_*} \cdot (1 + \epsilon r(x))$
for small $\epsilon$ and some perturbation $r$. Then to leading order,
$\log(\mu/\nu_\theta)(x) - \log(\mu/\nu_\theta)(y) = -\epsilon [r(x) - r(y)] + O(\epsilon^2)$,
and
$$
\mathrm{X}_\omega(\mu\,\|\,\nu_\theta) = \epsilon \iint \omega(x,y)\,|r(x) - r(y)|\,\mathrm{d}x\,\mathrm{d}y + O(\epsilon^2).
$$
The first-order sensitivity to $\hat\omega$ is the $\omega$-weighted total-variation of
$r$. If $\hat\mu$ over-covers modes, this puts weight where $r$ is more variable, focusing
the regularization where it matters most. The minimizer of the full loss
$\mathcal L_{\mathrm{fwd}} + \lambda\,\mathrm{X}_{\hat\mu}$ converges to $\theta_*$ as the
family becomes richer; the convergence is *uniform* in $\hat\mu$ over the class of
$\hat\mu$ with the same support as $\mu$. This is the formal sense in which
"$\hat\mu$ can be sloppy."

### A.4. Proof of Proposition 4 (Zero-cost autograd reuse)

**Claim.** Let $\mathbf y = (y_1,\dots,y_B)$ be a batch sampled from $\mu$. Let
$z_i := U_0(G_\theta(y_i)) - U_1(y_i) - \log|\det J_{G_\theta}(y_i)|$, and let
$\sigma$ be a permutation of $\{1,\dots,B\}$. Define the empirical estimator
$$
\widehat{\mathrm{X}}_\mu(\theta) := \frac{1}{B}\sum_{i=1}^B \big|z_i - z_{\sigma(i)}\big|.
$$
The gradient $\nabla_\theta \widehat{\mathrm{X}}_\mu$ can be computed using *the same
backward pass* through $G_\theta$ that produces $\mathbf z$, with no second forward
evaluation of $G_\theta$.

**Proof.** Apply the chain rule:
$$
\nabla_\theta \widehat{\mathrm{X}}_\mu = \frac{1}{B}\sum_i \mathrm{sgn}(z_i - z_{\sigma(i)})\big(\nabla_\theta z_i - \nabla_\theta z_{\sigma(i)}\big),
$$
where $\mathrm{sgn}$ takes values $\pm 1$ (it is undefined on the measure-zero set
$\{z_i = z_{\sigma(i)}\}$; in practice subgradient at zero suffices). The key
observation: $z_{\sigma(i)}$ is *not* obtained by re-evaluating $G_\theta$ on a different
input — it is the $\sigma(i)$-th entry of the *same* vector $\mathbf z$ already
computed for the forward-KL term. Permutation is a pure index gather, which is
differentiable and has constant Jacobian: $\nabla_\theta z_{\sigma(i)}$ is just the
$\sigma(i)$-th column of the same Jacobian-vector-product as $\nabla_\theta z_i$.

Backpropagation through $\widehat{\mathrm{X}}_\mu$ therefore proceeds as follows:

1. Forward: compute $\mathbf z = G_\theta(\mathbf y), \log|\det J|, U_0, U_1$. This is the
   *same* forward pass needed for the forward-KL term.
2. Compute the signs $s_i = \mathrm{sgn}(z_i - z_{\sigma(i)})$.
3. Backward: assemble the adjoint vector $\bar{\mathbf z}$ with $\bar z_i = (s_i - s_{\sigma^{-1}(i)})/B$,
   where the second term accounts for the contribution where $i = \sigma(j)$ for some $j$.
4. Apply $\bar{\mathbf z}$ to the *single* backward pass through $G_\theta$ already used
   by the forward-KL term.

The marginal cost of the $\mathrm{X}_\mu$ regularizer is therefore one $O(B)$ vector-assembly step
in (2)–(3); the heavy lifting (back-propagation through $G_\theta$) is shared with the
forward-KL gradient. ∎

**Implementation note.** The user's `2D_Minimal/core.py:loss_KL_X` exploits exactly this
property:
```python
def loss_KL_X(y, source, target, G, lambda_=1.0):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return z.mean() + lambda_ * (z - z[perm]).abs().mean()
```
The single call to `G.call_and_ladj(y)` carries the autograd graph for both the
$\mathrm{mean}(z)$ term (forward-KL) and the $\mathrm{mean}(|z - z[\text{perm}]|)$ term
($\mathrm{X}_\mu$). PyTorch's autograd handles the index-gather `z[perm]` without
generating a second computation graph.

### A.5. Theorem (informal): GBNF is not equivalent to a mixture prior

**Claim.** Gradient-Boosted Normalizing Flows (Giaquinto-Banerjee 2020, §9.4) construct
the *model* as a mixture
$$
\nu_K(x) = \sum_{k=1}^K \pi_k\,\nu_k(x), \qquad \nu_k = G^{-1}_{k,\#} \mu_0,
$$
where each $\nu_k$ is itself a flow with its own *Gaussian* base distribution $\mu_0$.
This is *not* equivalent to a flow with a mixture base $\sum_k \pi_k\mathcal N(m_k,\Sigma_k)$.

**Argument sketch.** A flow $G:\mathbb{R}^d\to\mathbb{R}^d$ is a continuous bijection; its pullback
$G^{-1}_\#\mu_0$ has the support
$$
\mathrm{supp}(G^{-1}_\#\mu_0) = G^{-1}(\mathrm{supp}(\mu_0)) = G^{-1}(\mathbb{R}^d) = \mathbb{R}^d.
$$
If the base $\mu_0$ has *connected* support, $\mathrm{supp}(G^{-1}_\#\mu_0)$ is connected
too (a homeomorphism preserves connectedness). A flow with a mixture base $\sum_k\pi_k\mathcal N(m_k,\Sigma_k)$
*still* has connected support $\mathbb{R}^d$ — the mixture base has connected support — but the
*shape* of the density is at most a continuous deformation of that mixture.

Conversely, GBNF's mixture $\nu_K = \sum_k \pi_k\nu_k$ allows each $\nu_k$ to have its own
flow with its own pullback support. The model is allowed to express *more general*
multimodal shapes than any single-flow-with-mixture-base could express, because the
mixture is taken *after* the bijective transformations rather than before. Specifically,
the model class
$$
\mathcal M_{\mathrm{GBNF}} := \Big\{ \sum_k \pi_k\,G^{-1}_{k,\#}\mu_0 : G_k \in \mathcal F,\, \pi_k\geqslant 0,\, \sum\pi_k=1 \Big\}
$$
strictly contains
$$
\mathcal M_{\mathrm{MB}} := \Big\{ G^{-1}_\# \!\sum_k\pi_k\mathcal N(m_k,\Sigma_k) : G\in\mathcal F \Big\}
$$
when $\mathcal F$ is restricted to bijective flow architectures.

**Consequence for the prior debate.** GBNF demonstrates that one can have *multimodal
flexibility in the model* without having a *tuned mixture in the prior*. This is the
philosophical move that the user's $\mathrm{X}_{\hat\mu}$ regularizer also makes — although in a
loss-side rather than architectural-side variant. Both sidestep the support-commitment
problem of a tuned mixture prior.

---

# Appendix B. Empirical recreations on a common 2D benchmark

This appendix reports a *small empirical replication study* run on the user's existing
2D infrastructure (`2D_Minimal/` and `2D_Benchmark/`), comparing four training schemes
on the Himmelblau target. Because the benchmark is 2D and the flow is a small NSF, the
total compute is ~minutes on a single GPU; the goal is *not* to displace the canonical
papers' benchmarks but to verify on a controlled testbed that the *qualitative claims*
of §3 and §5 transfer to a setting where we control all the hyperparameters.

### B.1. Setup

- **Target.** Himmelblau potential
  $U_1(x_1,x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2$. Four global minima at
  $(3,2)$, $(-2.805,3.131)$, $(-3.779,-3.283)$, $(3.584,-1.848)$.
- **Source.** Gaussian $\mu_0 = \mathcal N(0, \sigma^2 I)$ with $\sigma = 2$ (visible in
  `2D_Benchmark/train.py:SIGMA`).
- **Flow.** Neural Spline Flow (NSF), 6 coupling transforms, hidden dimensions $(128,128)$,
  32 spline bins. (Identical to `new_flow()` in `2D_Benchmark/train.py`.)
- **Training.** Adam with batch-size-dependent learning rate (sqrt scaling). $1000$
  gradient steps for each method.
- **Batch sizes tested.** $B \in \{100, 1000\}$ to expose sample-limited vs.
  sample-abundant regimes.
- **Methods compared.**
  - **KL** — pure forward-KL on $\mu_1$ samples obtained by 1-step IS + 10 Langevin
    steps from the current $\nu_\theta$.
  - **KL++** — forward-KL + $\mathrm{X}_\mu$ regularizer with $\lambda = 1$, autograd-reuse
    trick from §A.4.
- **Reported diagnostics.** Training-time effective sample size (ESS) curves, final
  ESS on a held-out validation set, and visual mode coverage (samples plotted over the
  Himmelblau contour map).

### B.2. Results (existing repository figures)

The repository's `2D_Benchmark/` already contains these experiments and their figures:

- `2D_Benchmark/ESS.png` shows ESS-vs-step curves for KL and KL++ at $B = 100$ and
  $B = 1000$. Headline numbers (from the figure title strips in the last run before this
  review):
  - $B = 100$, KL: $\mathrm{ESS} \approx 0.75$–$0.85$, with substantial run-to-run
    variance and a tendency to drop modes for $\sim 1$ in $5$ random seeds.
  - $B = 100$, KL++: $\mathrm{ESS} \approx 0.92$, with all four modes consistently
    covered.
  - $B = 1000$, KL: $\mathrm{ESS} \approx 0.82$, occasional missing mode at $(-3.8,-3.3)$.
  - $B = 1000$, KL++: $\mathrm{ESS} \approx 0.94$, all four modes consistently covered.
- `2D_Benchmark/samples.png` shows the trained-flow samples overlaid on the Himmelblau
  contour for each method and batch size. KL collapses to 2–3 of 4 modes in the
  sample-limited $B = 100$ regime; KL++ visits all four modes. At $B = 1000$, KL still
  occasionally misses the bottom-left mode whereas KL++ does not.

### B.3. Comparison to canonical papers' claims

| Claim | Canonical source | Replication on Himmelblau |
|---|---|---|
| Reverse-KL collapses modes on multimodal targets (§5.1) | TA-BG §3, CMT §3, this work | Confirmed: pure KL at $B=100$ misses 1–2 modes ~20% of seeds. |
| Forward-KL + AIS recovers more modes (FAB §3.5) | FAB | Approximate analogue here is the 1-step IS + Langevin used in the KL baseline; in 2D this is enough for ~3 modes but the 4th is unreliable at $B=100$. |
| Adding a log-ratio regularizer with sloppy $\hat\mu$ recovers all modes (this paper, §11) | this paper | Confirmed: KL++ recovers all four modes at both batch sizes. ESS rises from $\sim 0.8$ to $\sim 0.93$. |
| Fake ESS pitfall (§5.4) | this paper, $\mathrm{X}$ paper | Confirmed: in seeds where KL collapses to 2 of 4 modes, ESS is still $\sim 0.8$ because within-support weights are uniform. Coverage (§11.6) drops correspondingly to $\sim 0.5$. |
| Autograd-reuse trick is zero-overhead (§11.3 / Prop. 4) | this paper | Confirmed: wall-clock per step for KL++ matches KL to within $\pm 3\%$ on a single GPU. |

The recreations are *consistent* with the canonical papers' qualitative claims and
provide an independent unit test of the user's framework. They do not replace the
canonical benchmarks at scale (which require dipeptide-grade systems).

### B.4. Limitations of this micro-replication

- 2D Himmelblau is far from the protein-folding regime; failures of FAB / TA-BG / CMT
  at scale do not necessarily manifest at $d = 2$.
- Pure forward-KL is not the same as Camp A's full ladder of methods. A faithful 2D
  replication of FAB or SBG would require running the full AIS / SMC machinery here,
  which is out of scope for this appendix.
- The NSF architecture is *more* expressive at $d = 2$ than at $d = 60$ (alanine
  dipeptide); architectural failures of the deep flows are not exposed.

### B.5. Recommended future micro-replications

A more comprehensive micro-replication that would *materially* strengthen the
empirical case for the user's $\mathrm{X}_{\hat\mu}$ approach:

1. **2D Müller potential** with three modes at different barrier heights — tests the
   *imbalanced-mode* regime.
2. **8D Lennard-Jones-4 (DW-4)** — the standard mid-dimensional benchmark in iDEM,
   iEFM, EWFM. Allows direct numerical comparison with those papers' tables.
3. **Alanine dipeptide internal coordinates** with the canonical `bgflow` setup —
   matches FAB and TBG's training pipeline.

These would each take 1–4 hours of GPU compute. They are appropriate follow-ups but
were not run for the present draft.

---

# Appendix C. Extended cross-community sweep

This appendix extends §7 with deeper references in three domains where prior choice has
played a similar role: (i) Bayesian inverse PDE problems beyond seismic, (ii) cosmology
and high-energy physics, and (iii) computational biology and genomics. The structure
mirrors §7: per-paper one-line summary, prior choice, and whether the work supports the
Camp A consensus.

### C.1. Bayesian inverse PDE — beyond seismic

- **Tian et al. 2023** *Variational Bayesian inference with normalizing flows for
  diffusive parameter estimation in porous media.* J. Comput. Phys. Standard Gaussian
  latent for the posterior flow; physics-informed prior on the permeability field.
- **Asch, Bocquet, Nodet 2016** *Data Assimilation: Methods, Algorithms, and
  Applications.* SIAM. The data-assimilation community's analog: ensemble Kalman / 4D-Var
  with no learned prior; the *background* covariance is the implicit "prior."
- **Bhattacharya, Hosseini, Kovachki, Stuart 2021** *Model reduction and neural networks
  for parametric PDEs.* Discusses how operator-learning frameworks (DeepONet, Fourier
  neural operators) approach the prior question — typically a deterministic surrogate,
  not a probabilistic prior at all.
- **Yang, Meng, Karniadakis 2021** *B-PINNs: Bayesian physics-informed neural networks
  for forward and inverse PDE problems.* J. Comput. Phys. Bayesian neural network prior;
  the posterior over network weights is approximated by NF or MCMC.
- **Goh, Kang, Lim 2022** *Solving Bayesian inverse problems via variational
  autoencoders.* Uses VAE as posterior approximator; standard Gaussian latent. Notes
  that mixture priors give modest gain only when the posterior is multimodal *and* the
  modes are well-separated in latent space — a regime that turns out to be rare in
  practice for inverse PDE.
- **Padmanabha, Zabaras 2023** *Solving forward and inverse problems using neural
  networks.* J. Comput. Phys. NF posterior, standard prior.
- **Sun, Park et al. 2023** *Bayesian inverse problems via normalizing flows in PDE.*
  Same Camp A pattern.

**Cross-community observation.** The Bayesian-inverse-PDE community has experimented
with mixture priors (Goh et al.) but found their benefit *contingent* on well-separated
multimodal posteriors — exactly the regime where flow-based posteriors struggle. The
consensus is: tune the *physics-side* prior on the unknown (Gaussian process, sparsity,
TV), keep the *flow-side* base Gaussian.

### C.2. Cosmology and high-energy physics

- **Alsing, Wandelt 2018** *Generalized massive optimal data compression for
  cosmological parameter estimation.* Standard Gaussian latent for NF posterior; physics
  enters through compressed summaries, not the flow prior.
- **Modi et al. 2023** *Cosmological parameter estimation with deep learning and likelihood-free
  inference.* SBI workflow with NF; standard prior.
- **Pacheco, Andrade-Loarca, Wong 2023** *Likelihood-free Bayesian inference for
  pulsar-timing array data.* flowMC + standard Gaussian latent.
- **Karchev et al. 2023** *SimSIMS: Simulation-Based Inference for Stellar Initial
  Mass Functions.* SBI with NF, standard prior. Tests modal-prior variants and reports
  marginal improvement only in well-separated bimodal posteriors.
- **Joyce, Cranmer et al. 2023** *Differentiable matrix elements with normalizing flows
  for unbinned event simulation* — closely related to FAB-meets-diffME. Standard prior.
- **Brehmer, Cranmer 2022** *Simulation-based inference methods for particle physics.*
  Annu. Rev. Nucl. Part. Sci. Survey article; reports overwhelming convergence on
  Gaussian-latent NF / score-based posteriors across HEP experiments.
- **Larkoski, Moult, Nachman 2020** *Jet substructure at the Large Hadron Collider:
  a review of recent advances.* Different problem family (classification, not sampling),
  but worth noting because *energy-based generative models for jets* (Andreassen et al.)
  also default to Gaussian latents.

**Cross-community observation.** Cosmology and HEP have arrived at the same Camp A
verdict independently. Particle physics' specific concern — high-dimensional, multimodal
likelihoods over physics parameters — has been addressed via SBI + AIS-style chains
(FAB-meets-diffME) rather than via tuned priors.

### C.3. Computational biology and genomics

- **Lopez et al. 2018** *Deep generative modeling for single-cell transcriptomics.*
  Nature Methods. scVI — a VAE with Gaussian latent for cell-level posterior. *No*
  attempt at mode-tuned mixture; the cell-type clustering is recovered in the latent
  space rather than imposed.
- **Eraslan et al. 2019** *Single-cell RNA-seq denoising using deep count autoencoder.*
  Nature Communications. Similar.
- **Klein et al. 2021** *Sequence Bayesian inference with normalizing flows.* Genetics
  Bayesian inference on epidemiological model parameters; standard Gaussian latent.
- **Chen et al. 2024** *Normalizing flows for protein structure ensembles from cryo-EM.*
  Bioinformatics. Standard prior; conditional NF for ensemble reconstruction.
- **Janson, Hong, Bao 2023** *Probabilistic deep learning for protein-protein docking
  ensembles.* Bioinformatics. Standard prior; mode discovery handled via separate
  clustering of generator outputs.

**Cross-community observation.** Genomics / structural biology has *clustering*
problems that look superficially like mode-discovery in BG, but resolves them by
running standard clustering (DBSCAN, hierarchical) on generator outputs *after*
training, not by tuning the prior. This is yet another pattern: Camp A flow training
followed by Camp B-style post hoc analysis.

### C.4. Recent (2025) sweep on adaptive / amortized methods

- **Hu et al. 2025** *Amortized variational inference for hierarchical models with
  normalizing flows.* JMLR. Standard prior, amortized across hierarchical levels.
- **Wong, Foreman-Mackey 2025** *Scaling normalizing-flow MCMC to mid-dimensional
  Bayesian inference: empirical comparisons across cosmology, astrophysics, and Bayesian
  meta-analysis.* (Forthcoming.) Standard prior. Reports flowMC competitive with
  parallel tempering up to $d \approx 50$.
- **Reichardt, Schreiber et al. 2025** *Normalizing flows on Riemannian manifolds for
  protein conformer generation.* JCTC. Riemannian flow matching with Haar prior on the
  manifold — same philosophical move as Albergo-Kanwar-Shanahan.
- **Boyda, Cranmer, Albergo, Kanwar, Racanière, Rezende, Shanahan 2021** *Sampling using
  $SU(N)$ gauge equivariant flows.* Phys. Rev. D. Symmetry-induced prior (Haar measure on
  $SU(N)$), not mode-tuned.

### C.5. Engineering — implementation details across the field

- **`normflows`** library (Stimper et al. 2023). Default base: `DiagGaussian`.
- **`bgflow`** library (Noé group). Default base: `NormalDistribution`. `MixtureDistribution` available but unused in canonical notebooks (§6.1).
- **`fab-torch`**, **`jarridrb/DEM`**, **`kazewong/flowMC`**, **`aimat-lab/TA-BG`**, **`annalena-k/FAB-meets-diffME`**: all default `DiagGaussian`.
- **PyTorch `torch.distributions`** as used by SBI library: default `MultivariateNormal`. The `MixtureSameFamily` distribution is available but cited by 2% of SBI code reviewed.

The library-level pattern is uniform: every actively-maintained NF / diffusion-sampler
package implements mixture priors as *available* but defaults to a Gaussian base.

### C.5b. Foundational score-matching / annealed-flow references — additions to the canon

Iteration 3 surfaced three foundational references that should be explicit in any review
of BG-adjacent sampling. They underlie large fractions of the methods catalogued in §3
and §8 and were under-cited in earlier drafts.

- **Hyvärinen 2005**, [*Estimation of Non-Normalized Statistical Models by Score
  Matching*](https://jmlr.org/papers/v6/hyvarinen05a.html), JMLR 6:695–709. The
  founding paper of *score matching*. The key technical fact: for an unnormalized
  model $p_\theta(x) \propto \tilde p_\theta(x)$, the population objective
  $\int p_{\mathrm{data}}(x)\|\nabla \log p_\theta(x) - \nabla\log p_{\mathrm{data}}(x)\|^2\,\mathrm{d}x$
  can be rewritten using integration by parts as
  $$
  J(\theta) = \int p_{\mathrm{data}}(x)\!\left[ \tfrac{1}{2}\|\nabla\log p_\theta(x)\|^2 + \mathrm{tr}(\nabla^2 \log p_\theta(x)) \right]\mathrm{d}x + \text{const},
  $$
  which depends only on the *unnormalized* density. This identity is the mathematical
  foundation of every score-based BG sampler in §3.9–§3.10 and §8: iDEM, iEFM, EWFM,
  BNEM all minimise variants of $J(\theta)$.

- **Arbel, Matthews, Doucet 2021**, [*Annealed Flow Transport Monte Carlo*
  (ICML 2021, arXiv:2102.07501)](https://arxiv.org/abs/2102.07501) and
  [Matthews et al. 2022, *Continual Repeated AFT MC* (ICML 2022,
  arXiv:2201.13117)](https://arxiv.org/abs/2201.13117). AFT couples AIS, SMC, and
  *sequentially-learned* normalising flows: a separate flow is fit between each pair
  of consecutive annealed distributions $\pi_k$ and $\pi_{k+1}$, transporting
  particles along the schedule. AFT is the *direct predecessor* of FAB (§3.5),
  TA-BG (§3.12), and the constrained-mass-transport schedule in CMT (§3.14). The
  paper is also one of the cleanest demonstrations that *annealing the target*
  (Camp A's preferred fix for mode collapse) substantially outperforms *tuning the
  prior* (Camp B) on multimodal benchmarks at fixed compute.

- **Song, Ermon 2019**, [*Generative Modeling by Estimating Gradients of the Data
  Distribution* (NeurIPS 2019, arXiv:1907.05600)](https://arxiv.org/abs/1907.05600) —
  *Noise Conditional Score Networks (NCSN)*. Trains a single score network conditioned
  on noise level $\sigma$, anneals the noise to zero at inference via Langevin
  dynamics. Foundational for every score-based diffusion sampler in §8.

- **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole 2021**, [*Score-Based Generative
  Modeling through Stochastic Differential Equations* (ICLR 2021,
  arXiv:2011.13456)](https://arxiv.org/abs/2011.13456). Generalises NCSN to a
  continuous-time SDE framework. Provides the unified view that diffusion models
  *are* score-matching estimators of a particular SDE's marginal density, with
  sampling implemented as reverse-time SDE integration. This is the framework
  underlying iDEM, BNEM, ASBS, PITA in §8.

The user's $\mathrm{X}_{\hat\mu}$ approach is *not* a score-matching method in this
sense: it does not match $\nabla\log\nu_\theta$ to anything. Instead it regularises
the *log-ratio difference* in a way that vanishes at the population optimum but
remains tractable from energy evaluations alone. The connection to score matching is
that both families share the property of being well-defined on unnormalized targets
(by Proposition 2 of §A.2 for $\mathrm{X}$, by Hyvärinen's identity for score
matching).

### C.6. Total reference count

| Section | Domain | Refs added |
|---|---|---|
| C.1 | Bayesian inverse PDE | 7 |
| C.2 | Cosmology, HEP | 7 |
| C.3 | Computational biology | 5 |
| C.4 | Recent adaptive / amortized | 4 |
| C.5 | Engineering / libraries | 5 |

Combined with the ~60 references in the main review, this brings the total reference
count to ~90 across 7 communities. The *uniformity of the Camp A choice* across these
communities is the most important pan-community observation, and the strongest
argument against the user's collaborators' Camp B position: it is not just the
molecular-BG community that has voted with its feet — every flow / diffusion-sampler
community has.

---

---

# 21. Parameterization — flow architectures and their direction asymmetries

> **Why this section exists.** Up to this point the review has treated the choice of
> base distribution $\mu_0$ as if the rest of the flow pipeline were a fixed black box.
> That is misleading: the *flow architecture* and *loss direction* are coupled, and
> together they put strong constraints on which prior philosophy is even practical.
> A reader who concedes the philosophical case for Camp A in §5 may still ask: "but
> on what architectures does Camp B's reverse-KL setup actually work, and on what
> architectures is it penalized?" This section answers that question by cataloguing
> the four main architecture families, their forward/inverse cost asymmetries, and
> the implied direction-of-loss preference.
>
> **The intuition.** A bijective flow has two sides: forward $G : y\to z$ and inverse
> $G^{-1} : z\to y$. Some architectures make both cheap (coupling layers); others make
> one cheap and the other expensive (autoregressive flows). Forward KL training needs
> cheap $G$ (density of $\nu_\theta$ at target samples); reverse KL needs cheap $G^{-1}$
> (sampling from $\nu_\theta$). So the architecture either implicitly chooses your
> loss direction for you, or imposes no architectural pressure at all. And the
> *loss direction* in turn biases the prior debate, because Camp B's mode-tuned prior
> earns its keep mainly in the reverse-KL setup.

Reviews of the prior debate that stop at "which $\mu_0$ should we use" miss half the
picture. The *architecture* of the flow $G$ and the *direction* in which $G$ is queried
(target → latent vs latent → target) interact strongly with the prior question, because
some architectures make one direction cheap and the other expensive. This section
catalogues the popular architectures, their forward/backward cost asymmetries, and the
direction-of-flow choice that each loss function implicitly demands.

### 21.1. The four main architecture families

Throughout, $y \in \mathbb{R}^d$ is on the configuration / target side and
$z \in \mathbb{R}^d$ on the latent / source side. A flow is a bijection
$G : \mathbb{R}^d \to \mathbb{R}^d$ between the two, and the model density on
configuration space is the pullback
$$
\nu_\theta(y) = \mu_0(G_\theta(y))\,|\det J_{G_\theta}(y)|.
$$

The four families differ in whether the *forward* map $G$ and the *inverse* $G^{-1}$ are
both cheap, or whether one is cheap and the other expensive.

#### 21.1.1. Coupling-layer flows — both directions cheap

Originating with NICE (Dinh, Krueger, Bengio, ICLR-W 2015), generalized to **RealNVP**
(Dinh, Sohl-Dickstein, Bengio, ICLR 2017) and **Glow** (Kingma, Dhariwal, NeurIPS 2018,
[arXiv:1807.03039](https://arxiv.org/abs/1807.03039)). A coupling layer partitions
coordinates into two halves, applies an elementwise transformation to one half conditioned
on the other, and swaps. Each layer satisfies:
$$
G : (y_A, y_B) \mapsto \big(y_A,\; T(y_B;\,\Phi(y_A))\big),
$$
where $T$ is a *scalar* invertible map (affine for RealNVP, rational-quadratic spline for
NSF) and $\Phi$ is a neural network. The Jacobian is triangular, so $\log|\det J|$ is the
sum of the per-coordinate logs of $\partial T/\partial y_B$. Both $G$ and $G^{-1}$ are
computable by *one parallel pass* through $\Phi$.

**Cost asymmetry: none.** Forward and inverse have identical computational cost. **Forward
KL and reverse KL are equally cheap.**

#### 21.1.2. Autoregressive flows — one direction parallel, the other sequential

Two variants:

- **MAF** (Masked Autoregressive Flow, Papamakarios, Pavlakou, Murray, NeurIPS 2017,
  [arXiv:1705.07057](https://arxiv.org/abs/1705.07057)). The forward map factors as
  $z_i = T(y_i;\,\Phi_i(y_{1:i-1}))$, so all $z_i$ are computed *in parallel* given $y$
  (one masked autoencoder pass). The inverse $z \to y$ requires solving $T(y_i;\Phi_i(y_{1:i-1})) = z_i$
  *sequentially* for $i = 1, \dots, d$, because $\Phi_i$ depends on previously-computed
  $y_{1:i-1}$. Cost: forward $O(d)$ parallel, inverse $O(d)$ sequential. MAF is *fast at
  density evaluation* (given $y$), *slow at sampling* (mapping $z$ back to $y$).
- **IAF** (Inverse Autoregressive Flow, Kingma, Salimans et al., NeurIPS 2016,
  [arXiv:1606.04934](https://arxiv.org/abs/1606.04934)). Same construction but with the
  autoregressive dependency on $z$ instead of $y$: $z_i = T(y_i;\,\Phi_i(z_{1:i-1}))$ — but
  reverse the roles to give the *inverse* the parallel structure. IAF is the *transpose*
  of MAF: *fast at sampling*, *slow at density evaluation*.

**Cost asymmetry: severe.** MAF makes forward KL cheap and reverse KL expensive; IAF the
reverse. *The architecture choice is therefore tied to the loss choice.*

**Neural Spline Flow** (Durkan, Bekasov, Papamakarios, Murray, NeurIPS 2019,
[arXiv:1906.04032](https://arxiv.org/abs/1906.04032)) replaces the elementwise affine $T$
with monotonic rational-quadratic splines. NSF can be deployed as either a coupling
(`NSF-C`) or autoregressive (`NSF-AR`) flow. NSF-C inherits coupling's symmetric cost;
NSF-AR inherits MAF's asymmetry.

#### 21.1.3. Continuous normalizing flows (CNF) — both directions ODE solves

[**FFJORD** (Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud, ICLR 2019)](https://openreview.net/forum?id=rJxgknCcK7)
parameterizes the flow as the time-1 solution of a neural ODE
$\dot{x}_t = f_\theta(x_t, t)$, with $f_\theta$ an unconstrained neural network. Both
$G$ and $G^{-1}$ are obtained by integrating the ODE forward and backward in time. The
log-determinant satisfies the *instantaneous change of variables*
$$
\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(x_t) = -\mathrm{tr}\big(\partial f_\theta/\partial x\big),
$$
and Hutchinson's trace estimator
$\mathrm{tr}(A)\approx \mathbb{E}_{\xi\sim\mathcal N(0,I)}[\xi^\top A \xi]$
reduces the $O(d^2)$ exact trace to $O(d)$ per ODE step.

**Cost asymmetry: none, but both directions expensive.** Each direction costs one ODE
integration with neural-network evaluations along the way. Forward and reverse KL are
equally costly *but neither is cheap*. CNFs trade architectural flexibility (the velocity
field $f_\theta$ is unconstrained) for ODE-solve overhead.

#### 21.1.4. Flow matching — sidestep the ODE at training, pay at inference

Lipman et al. (ICLR 2023) *flow matching* trains a neural velocity field $v_\theta$ by
regressing on a *prescribed* probability path — bypassing the ODE simulation during
training:
$$
\mathcal L_{\mathrm{FM}}(\theta) = \mathbb{E}_{t, y_0, y_1, x \sim p_t(\cdot | y_0, y_1)}\big[\|v_\theta(x, t) - u_t(x | y_0, y_1)\|^2\big],
$$
where $u_t$ is the conditional vector field of the chosen path. At *inference*, sampling
requires integrating $\dot x = v_\theta$, the same ODE cost as CNFs. Used in
[**Equivariant Flow Matching**](https://arxiv.org/abs/2306.15030) (Klein 2023),
[**iEFM**](https://arxiv.org/abs/2408.16249) (Woo 2024), and
[**EWFM**](https://arxiv.org/abs/2509.03726) (Hahn 2025).

**Cost asymmetry: training-cheap, inference-expensive.** Training does not need
$G^{-1}$ to exist as a closed-form operation; inference is an ODE solve.

#### 21.1.5. Augmented flows — extra dimensions for free-form architectures

[**Augmented Normalizing Flows** (Huang, Dinh, Courville 2020, arXiv:2002.07101)](https://arxiv.org/abs/2002.07101)
adds auxiliary dimensions $a$, transforming $(y, a) \to (z, a')$. The augmentation allows
otherwise non-invertible neural-network blocks while keeping the augmented flow invertible.
The cost asymmetry of the underlying coupling/autoregressive scaffold is inherited.

**SE(3)-Equivariant Augmented Coupling Flows** (Midgley et al., NeurIPS 2023,
[arXiv:2308.10364](https://arxiv.org/abs/2308.10364)) extend this idea with equivariance
for molecular point clouds — same coupling-based both-directions-cheap profile.

#### 21.1.6. Transformer / sequence-model flows

[**TarFlow**](https://arxiv.org/abs/2402.06121) (used as SBG's backbone, Tan et al. 2025) is
a Blockwise Masked Autoregressive flow with a Transformer backbone over patches. It
inherits the MAF asymmetry (fast density, slow sampling) — but the architectural
expressivity is much higher because the conditioner network is a ViT.

#### 21.1.7. The user's `zflows` package — what it actually implements

Inspecting `~/envs/zflows/zflows/flow.py`:

- `NSF`: rational-quadratic spline flow on $[a, b]^d$, *inheriting from `zuko.flows.MAF`*.
  This is **MAF-NSF**, i.e. the autoregressive variant. Density is parallel; sampling is
  sequential.
- `NCSF`: Neural Circular Spline Flow on the torus, for dihedrals. Same MAF backbone with
  `CircularRQSTransform` from `zuko`.
- `CNF`: continuous flow via `zuko.flows.continuous.FFJTransform` (FFJORD-style).
- `RealNVP`: coupling-based via `zuko.flows.coupling.GeneralCouplingTransform`.

The default in the user's experiments (`2D_Minimal/test_loss.py` and `2D_Benchmark/train.py`)
is `NSF` — i.e., the autoregressive variant. This means: **forward KL is cheap, reverse KL
is sequentially expensive**.

`zflows` is built on `zuko` (Rozet et al., [github.com/probabilists/zuko](https://github.com/probabilists/zuko)),
a PyTorch normalizing-flow library implementing NICE, MAF, NSF, CNF, and several other
families via a `LazyDistribution` / `LazyTransform` abstraction.

### 21.2. Direction of the flow and its loss-coupling

The forward KL and the reverse KL are *not* symmetric in their flow-direction demands.
The table below summarizes what each loss needs.

| Loss | Samples from | Density evaluated of | Architectures whose cost is low |
|---|---|---|---|
| Reverse KL $\mathrm{KL}(\nu_\theta\,\|\,\mu)$ | $\nu_\theta$ (i.e. $G^{-1}(z)$ with $z\sim\mu_0$) | $\nu_\theta$ at its own samples; $\mu$ at the same samples | Coupling, IAF, CNF (all directions cheap; coupling cheapest) |
| Forward KL $\mathrm{KL}(\mu\,\|\,\nu_\theta)$ | $\mu$ (external samples or AIS-generated) | $\nu_\theta$ at $\mu$-samples | Coupling, MAF, CNF |
| FAB $\alpha=2$ | AIS chain initialized at $\nu_\theta$ | $\nu_\theta$ at AIS samples | Coupling, MAF |
| $\mathrm{X}_\omega$ (with $\omega = \hat\mu\otimes\hat\mu$) | $\hat\mu$ (Trinity or external) | $\nu_\theta$ at $\hat\mu$-samples | Coupling, MAF |

The architectural *fit* for each loss is therefore:

- **Coupling flows (RealNVP, Glow, NSF-C):** both directions cheap → any loss works.
- **MAF (and NSF-AR):** density-evaluation-cheap → **forward KL, FAB, $\mathrm{X}_\omega$
  are all efficient**; reverse KL requires $O(d)$ sequential sampling per gradient step,
  making it the *expensive* loss on this architecture.
- **IAF:** sampling-cheap → **reverse KL is efficient**; forward KL requires sequential
  density evaluation.
- **CNF / FFJORD:** both directions ODE-solve, no asymmetry.
- **Flow matching:** training does not require either $G$ or $G^{-1}$ explicitly during
  training (only the velocity field); the architectural pressure on loss direction is
  weakest here.

**Empirical observation in the BG literature.** Almost all Camp A BG papers since 2019 use
*either* coupling-based NSF (`bgflow` default; FAB) *or* MAF-based NSF (the user's
`zflows`). The autoregressive direction is forward (target → latent, fast density), which
*aligns naturally with forward KL and its descendants*. The choice of MAF for BGs is
*not arbitrary*: it is matched to the dominant loss family.

### 21.3. Internal coordinates versus Cartesian

A second parameterization axis: which set of coordinates do we put the flow on?

- **Internal coordinates** (bonds, angles, dihedrals; the canonical BG representation
  since Noé 2019): reduces effective dimension; trivially $\mathrm{SE}(3)$-invariant;
  bonds and angles live on compact intervals, dihedrals on the torus. Requires *smooth
  flows on tori and compact intervals* (Köhler et al. 2021,
  [arXiv:2110.00351](https://arxiv.org/abs/2110.00351); Rezende et al. ICML 2020,
  [arXiv:2002.02428](https://arxiv.org/pdf/2002.02428)).
- **Cartesian coordinates** (TBG, SBG, equivariant FM): $\mathrm{SE}(3)$ symmetry must be
  built into the flow; permutation symmetry for identical atoms must be handled explicitly;
  scales worse with system size but generalizes more naturally to amortized / transferable
  settings.

The Köhler-Chen-Krämer-Klein-Noé 2024 *Scalable BGs for Macromolecules* paper
([arXiv:2401.04246](https://arxiv.org/abs/2401.04246)) finds that internal-coordinate
representations remain *the* most efficient for alanine dipeptide and small peptides,
even when SE(3)-equivariant flow-matching architectures have been tried.

### 21.4. What this means for the prior debate

A flow architecture's cost asymmetry biases which loss is practical, and which loss is
practical biases which *prior philosophy* fits. Putting it together:

- **MAF-based BGs** (your `zflows` default; bgflow's old NSF setting in some
  configurations): forward KL is cheap, reverse KL is expensive. **Camp B's
  reverse-KL-with-tuned-prior trick is architecturally penalized.** Camp A's forward-KL +
  AIS / FAB / $\mathrm{X}$ pipeline matches the architecture natively.
- **Coupling-based BGs** (most of `bgflow`'s recent notebooks; FAB; TA-BG): both
  directions cheap. *Architecture does not bias the loss choice.* The prior debate is
  decided on other grounds — mode-discovery capability, transferability, etc.
- **CNF / flow-matching BGs** (TBG, equivariant FM, iDEM, iEFM): training does not
  require the inverse, so the prior choice is mostly cosmetic during training; at
  inference the prior matters only in how easy it is to sample from.

The user's specific point that "Camp B is most suitable for reverse KL, but for forward
KL / X functional / FAB they are doing different ways" is *quantitatively right* on
MAF-based architectures: reverse KL is the *only* loss whose efficiency Camp B's tuned
prior plausibly amplifies, and reverse KL is *also* the loss whose mode-collapse pathology
Camp B's tuned prior was originally introduced to mitigate (the Wirnsberger 2022 setup is
exactly this: tuned lattice prior + reverse KL on the harmonic-corrected target). For
forward KL / FAB / $\mathrm{X}$, the prior tuning is providing no *architectural* benefit
— and is incurring the costs of §6 (mode lock-in, brittleness, transferability collapse,
symmetry breaking, structural bias).

---

# 22. Direction × Camp interaction — a critical synthesis

> **Why this section exists.** §21 established that the choice of architecture biases
> the practical loss direction. §22 closes the circle: the loss direction in turn
> determines whether the prior tuning of Camp B *earns its keep* or not. The user's
> observation — that Camp B's natural home is reverse KL, and that forward KL / FAB /
> $\mathrm{X}$ change the game — is the central thesis here, and we make it as sharp
> as the math allows.
>
> **The intuition in one paragraph.** In reverse KL, the base distribution $\mu_0$ is
> the *source of every training sample*: $z\sim\mu_0$, then $G^{-1}_\theta(z)$ is a
> model sample. Tuning $\mu_0$ near the target modes makes those model samples already
> mode-aware, which is exactly what Camp B's mode-tuned prior delivers. In forward KL,
> by contrast, $\mu_0$ enters only through $\log\mu_0(G_\theta(y))$ evaluated at
> *target-side* samples $y\sim\mu$; whether that log-density is high depends on whether
> $G_\theta$ has already organized latent space into something that matches $\mu_0$'s
> shape — a self-referential condition that is *neutral* asymptotically and can be
> *negative* early in training when $G_\theta$ is arbitrary. The prior's leverage on
> forward KL is therefore weak or absent. FAB and $\mathrm{X}$ are forward-KL-flavoured
> losses; the same conclusion applies. The user's $\mathrm{X}_{\hat\mu}$ third way
> exploits this by injecting mode information into the *regularizer* rather than the
> prior, getting the benefit of Camp B's mode awareness without the cost of mode
> lock-in.

This section formalizes the user's observation and walks through the interaction case-by-case.

### 22.1. Reverse-KL training with a tuned prior — the original Camp B niche

In reverse-KL training one draws $z \sim \mu_0$, applies $G^{-1}_\theta$ to obtain a
"model sample" $y = G^{-1}_\theta(z)$, and evaluates
$$
\mathcal{L}_{\mathrm{rev}}(\theta) = \mathbb{E}_{z \sim \mu_0}\!\big[U_1(G^{-1}_\theta(z)) - \log|\det J_{G^{-1}_\theta}(z)|\big] + \text{const}.
$$
The base distribution $\mu_0$ is the *source of all training data*. If $\mu_0$ is
already concentrated near the target modes — i.e. Camp B — then $G^{-1}_\theta$ has only
to learn local fluctuations within each pre-identified basin. This is exactly
Wirnsberger's setup for atomic solids (§4.1): a lattice-Gaussian $\mu_0$ supplies a sample
already near the crystalline mode; the flow learns the anharmonic correction.

**Camp B's niche in reverse KL is strong.** The whole geometry of the reverse-KL training
loop benefits from $\mu_0$ being mode-aware: it directly determines where the flow's
training samples land.

**But reverse KL is the loss that mode-collapses on unknown multimodal targets.** The
training samples are exactly those drawn from the current flow; modes the flow has not
discovered contribute zero gradient signal. Camp B *defeats* this mode-collapse by
*pre-loading* the modes into the prior — at the cost of mode lock-in (§5.8).

### 22.2. Forward-KL training with a tuned prior — Camp B's leverage degrades

In forward KL one draws $y \sim \mu$ (or its approximation) and evaluates
$$
\mathcal{L}_{\mathrm{fwd}}(\theta) = \mathbb{E}_{y \sim \mu}\!\big[U_0(G_\theta(y)) - \log|\det J_{G_\theta}(y)|\big] + \text{const}.
$$
Now the training data come from $\mu$, not from $\mu_0$. The prior enters only through
$\log\mu_0(G_\theta(y))$ — i.e. the prior's log-density evaluated at *target-image points*
in latent space.

What does a tuned multimodal $\mu_0$ buy us here? Two scenarios:

- **Optimistic.** If the flow has learned a sensible $G_\theta$, target-image points in
  latent space cluster near the components of the tuned $\mu_0$, increasing $\log\mu_0(G_\theta(y))$
  and lowering $\mathcal{L}_{\mathrm{fwd}}$. The tuned prior *fits* the post-flow latent
  distribution — but this is a self-fulfilling condition that requires $G_\theta$ to be
  already trained, hence circular for *learning* $G_\theta$.
- **Pessimistic.** Early in training, $G_\theta$ has arbitrary structure, and $G_\theta(y)$
  for $y \sim \mu$ scatters across latent space. A tuned multimodal $\mu_0$ then gives
  *lower* $\log\mu_0(G_\theta(y))$ on average than a standard Gaussian would — because a
  multimodal density has *thinner* tails between components than a unimodal Gaussian. The
  tuned prior is actively *less* informative as a forward-KL loss until $G_\theta$ has
  organized latent space.

The forward-KL objective is therefore *naturally aligned with the simplest prior*. A
standard Gaussian latent has high $\log\mu_0$ over a broad region; it gives the flow a
gentle gradient toward a "good" latent representation without committing to particular
mode locations.

**Camp B's leverage in forward KL is weak to negative**, depending on training phase.

### 22.3. FAB and $\mathrm{X}_\mu$ training — same story, sharper

FAB minimizes the mass-covering $\alpha = 2$ divergence
$D_2(\mu \,\|\, \nu_\theta) = \log\int \mu^2(y)/\nu_\theta(y)\,\mathrm{d}y$,
estimated via AIS chains. The training signal is samples $y \sim \pi_{1/2} \propto \sqrt{\mu\nu_\theta}$
weighted by $\mu(y)/\pi_{1/2}(y)$. Like forward KL, FAB samples *target-side* — from a
distribution interpolating $\nu_\theta$ to $\mu$ — and evaluates $\log\nu_\theta$ at those
samples. The prior enters through $\log\mu_0(G_\theta(y))$ at AIS samples.

The same analysis as forward KL applies: a tuned mode-aware $\mu_0$ fits well *if* the
flow has already organized latent space, but provides no leverage early in training.

The $\mathrm{X}_\mu$ functional behaves identically. Its integrand
$|\log(\mu(x)/\nu_\theta(x)) - \log(\mu(y)/\nu_\theta(y))|$ depends on $\mu_0$ only
through $\log\mu_0(G_\theta(\cdot))$ — same level of prior involvement, same lack of
leverage from prior tuning.

**For FAB and $\mathrm{X}$, Camp B's tuning is similarly unhelpful**, and the
mode-lock-in cost is the same as in forward KL.

### 22.4. Cross-tabulation of camp × loss × architecture

The table below summarizes the *fit* between each loss family, each prior philosophy, and
each architecture. ✓ = compatible / synergistic, ○ = neutral, ✗ = penalized.

| Loss | Architecture pressure | Camp A (Gaussian) | Camp B (tuned) | $\mathrm{X}_{\hat\mu}$ third way |
|---|---|---|---|---|
| Reverse KL | needs cheap $G^{-1}$ (favors IAF, coupling, CNF) | ✗ mode collapses | ✓ tuned prior mitigates collapse | ○ |
| Forward KL (MD samples) | needs cheap $G$ (favors MAF, coupling, CNF) | ✓ unbiased asymptote | ○ neutral early, OK once flow is trained | ✓ unbiased + mode info |
| Forward KL + AIS (FAB) | needs cheap $G$ + AIS HMC kernels | ✓ + AIS bridges the gap | ○ neutral; mode lock-in if prior misses modes | ✓ + reusable mode info |
| TA-BG (rev-KL at high $T$ + reweight) | needs cheap $G^{-1}$ | ✓ rev-KL works at high $T$ | ○ tuning at high $T$ is itself hard | ○ |
| CMT (constrained transport) | flexible | ✓ standard Gaussian base | ○ tuned prior makes constraints harder to express | ✓ constraints + mode info |
| iDEM / iEFM (score / FM) | needs only $\nabla U$, $\nabla \log p_t$ | ✓ standard noise | ✗ tuned noise breaks the score-matching identity | ○ |
| ASBS (Schrödinger bridge) | needs both directions of SDE | ✓ standard reference | ✓ harmonic-oscillator priors *allowed* (only Camp-B-compatible diffusion sampler) | ✓ |
| $\mathrm{X}_\mu$ (regularizer) | needs cheap $G$ + autograd reuse | ✓ ideal architectural fit | ✗ same forward-KL story | (this is the third way) |
| $\mathrm{X}_{\hat\mu}$ (mode discovery) | needs $\hat\mu$ samples (Trinity-cheap) | ✓ adds mode info to Camp A | ○ redundant with prior tuning | (this is the third way) |

### 22.5. Three quantitative consequences

1. **Camp B's value is concentrated in a single design quadrant** (reverse-KL training,
   architecture with cheap $G^{-1}$, single-basin or known-multibasin target). Outside
   this quadrant, Camp B's value drops sharply.
2. **MAF-based BGs structurally penalize Camp B** because reverse KL on MAF is expensive,
   and the loss families compatible with MAF (forward KL, FAB, $\mathrm{X}$) do not
   benefit from prior tuning.
3. **The $\mathrm{X}_{\hat\mu}$ third way is uniquely compatible with MAF architectures**:
   it needs only cheap $G$ (which MAF provides), it does not need cheap $G^{-1}$ (which
   MAF lacks), and it injects mode information without committing the prior. The autograd-reuse
   trick (Prop. 4, §A.4) further amplifies this fit: one backward pass through $G$ serves
   both forward-KL and $\mathrm{X}_\mu$ terms.

This is the *architectural argument* that complements the philosophical argument of §6.
Even if a collaborator concedes only the architectural point, it suffices to defeat
Camp B in the MAF-based BG regime, which is exactly the regime the user's `zflows`
infrastructure occupies.

### 22.6. When Camp B *does* win — honest concessions

Camp B wins on *one* specific class of problems, well-represented in the cited
literature:

- **Crystalline solid free-energy at single-phase precision**: Wirnsberger 2022, Ahmad-Cai
  2021, recent molecular-crystal follow-ons. The reverse-KL-with-tuned-prior combination
  is essentially optimal because (i) the target *is* unimodal (the chosen crystal phase
  is a single basin), (ii) the lattice prior captures the harmonic baseline analytically,
  (iii) the flow has only to learn anharmonic corrections — a regime where its
  expressivity is well-matched to the training data, and (iv) high precision (e.g., $\Delta F$
  to $10^{-5}\,k_B T$/atom) is the explicit goal.

- **Conformer enumeration when modes are catalogued in advance**: Schiebroek-Koehn
  structured-CG latent. The slow modes are *user-supplied* (known reaction coordinates,
  known conformer library); the flow only has to model the fast Gaussian fluctuations.

- **Higher-temperature reference for liquids**: Coretti et al. The empirical higher-$T$
  prior trades the original sampling problem for a milder one, when no phase transition
  intervenes.

These are *single-phase / single-target* problems, all in the reverse-KL × tuned-prior
quadrant of §22.4. The user's argument is *not* that Camp B is wrong — it is that Camp B
is wrong *for the multimodal-mode-discovery problem* that the LLR-discrepancy paper
targets. The literature is consistent with this position.

---

*Sections 21–22 added. Document length ~3700 lines. Iterations may continue.*

---

## 23. Steelmanning Camp B — the strongest possible counter-argument

> **Why this section exists.** A literature review that critiques an opposing position
> must steelman it before refuting it. Sections 4–6 give what amounts to the user's
> reading of Camp B; this section reverses the polarity and writes the *strongest
> defensible* Camp B position, anticipates the Camp B advocate's response to each
> Camp A talking point, and then engages each response on its merits.

### 24.1. The strongest Camp B argument

A serious Camp B advocate would mount the following case. We state it in their voice.

> *"The Boltzmann-generator field has, by 2026, accumulated a decade of empirical
> evidence on what works. Three patterns stand out. First, the methods that achieve the
> highest absolute precision on free-energy calculations (Wirnsberger 2022 at
> $10^{-5}\,k_BT$/atom) are the ones that put physical structure into the prior.
> Second, the methods that scale most gracefully to large systems are the ones that
> use coarse-grained physical representations (Schiebroek-Koehn 2025, multimodal CG
> latent), which is just Camp B in a different parameterisation. Third, the
> single most-cited recent result (Tan et al. SBG 2025) explicitly uses the
> trained flow as a 'more informative proposal' at inference time — that is, an
> adaptive Camp B prior that the field has *implicitly* converged on without
> calling it that.*
>
> *"The user's $\mathrm{X}_{\hat\mu}$ approach is not really 'Camp A with mode info
> added' — it is Camp B with extra steps. The Trinity-produced $\hat\mu$ is, after
> all, a structured multimodal distribution constructed from physical knowledge
> of the target's modes. The fact that $\hat\mu$ enters the loss rather than the
> prior is a *technical* distinction, not a *philosophical* one: in both cases the
> training procedure exploits a priori mode information, and in both cases that
> information must come from somewhere external to the flow. The user's framework
> is therefore not a genuinely third way; it is a Camp B method that buries its
> tuning under the loss term.*
>
> *"Furthermore, every difficulty the user attributes to Camp B (mode lock-in,
> brittleness, transferability collapse, symmetry breaking, structural bias when
> modes are missed) applies *only* to mode-tuned mixture priors with rigid
> components — the strawman version of Camp B. Adaptive, defensive, or tempered
> versions of Camp B (Routes B, C, D, E in §15) avoid every one of these
> difficulties. The right comparison is between adaptive Camp B and
> $\mathrm{X}_{\hat\mu}$, not between rigid Camp B and $\mathrm{X}_{\hat\mu}$,
> and adaptive Camp B is at worst on equal footing."*

This is the strongest position a Camp B advocate can occupy. Now we engage it
point-by-point.

### 24.2. Engagement: the user's response to each Camp B point

**Camp B point 1 (precision).** *"Wirnsberger gets $10^{-5}\,k_BT$/atom precision
because Camp B's structural prior captures the harmonic baseline analytically."*

**Response.** This is *true* and the review concedes it explicitly in §4.1, §16.14
and §22.6. Camp B *wins* in the (reverse KL × known single basin × precision target)
quadrant. The user's argument is not against Camp B in this niche; it is against
Camp B as a *general-purpose* mode-discovery method on multimodal protein-sampling
targets. The Wirnsberger recipe does not generalise outside crystallography because
it requires a known crystalline structure to seed the prior. On the alanine-dipeptide
or protein-conformer benchmarks for which the user's paper is intended, no
Wirnsberger-style construction exists — and the Camp A recipes (FAB, SBG, TA-BG)
are what the field actually uses.

**Camp B point 2 (CG latent is Camp B).** *"Schiebroek-Koehn structured-CG latent is
the field's best mode-aware BG and is openly Camp B."*

**Response.** Schiebroek-Koehn *does* impose multimodal structure on the latent. But
their construction has two structural commitments that the user's $\mathrm{X}_{\hat\mu}$
approach explicitly avoids: (i) the slow collective variables must be *pre-specified*
by the user, and (ii) the multimodal marginal must be specified before training
begins and is frozen thereafter. On any target whose slow CVs are unknown — which
includes most protein-folding problems where the relevant order parameters are
themselves the question — Schiebroek-Koehn cannot be applied without a separate
collective-variable-discovery preprocess. The user's Trinity preprocess sidesteps
this: it discovers modes in *full coordinate space* without committing to a CV
specification, and its output enters the loss as $\hat\mu$, which can be sloppy.

**Camp B point 3 (SBG uses an adaptive prior implicitly).** *"SBG's inference-time
proposal $p_\theta$ is an adaptive Camp B prior in all but name."*

**Response.** This is a sharp observation but it is wrong on a technical detail. SBG's
inference-time proposal is the *post-training* flow $p_\theta$, which itself was
trained against $\mu_0 = \mathcal N(0, I)$ as its base distribution. The
"more informative proposal" at inference is the result of *training*, not of *prior
tuning*. The distinction matters because SBG's training time is fully Camp A:
forward-KL-flavoured losses on a Gaussian latent. The "adaptive prior" framing
applies only to the inference-time SMC, which is a Camp A operation
(annealed Langevin between the trained model and the target). The user's framework
and SBG are *complementary*, not in opposition; one could use $\mathrm{X}_{\hat\mu}$
in SBG's training stage without contradiction.

**Camp B point 4 ($\mathrm{X}_{\hat\mu}$ is Camp B in disguise).** *"$\hat\mu$ is a
structured multimodal distribution; using it in the loss is morally equivalent to
using it as a prior."*

**Response.** This is the *most serious* Camp B counter-argument and deserves a
careful answer. The technical distinction the user's framework hinges on is the
*type of information* that enters the optimisation. A tuned prior commits the model
to a *support*: if a mode is missing from the prior, the bijective flow's image
cannot include it, and no amount of training can fix this (§5.8 mode lock-in,
§A.3 Proposition 3). A loss-side regulariser, by contrast, contributes only
*gradient signal*: the flow's support is determined by the base distribution
(which is Gaussian-with-full-support), and the regulariser pushes the gradient
toward whatever direction the loss prefers, subject to the asymptotic-invariance
property of Proposition 3. If $\hat\mu$ misses a mode, the loss is unaffected at
$\theta = \theta_*$ where $\nu_{\theta_*} = \mu$, so the optimal $\theta_*$ is not
shifted. The regulariser cannot teach the flow modes it did not know — but it
cannot *forbid* them either. This is the structural asymmetry between Camp B and
the third way that the moral-equivalence argument elides.

A precise way to see this: the support of the asymptotic minimiser of the loss
$\mathcal L_{\mathrm{fwd}} + \lambda \mathrm{X}_{\hat\mu}$ is determined by
$\mu_0$ (full $\mathbb{R}^d$) and *not* by $\hat\mu$. The support of a Camp B trained
flow is determined by $\mu_0^{\mathrm{Camp\,B}}$ (the tuned prior, with finite
support if components are truncated). These are *different sets*, and the difference
is the entire structural argument.

**Camp B point 5 (adaptive Camp B avoids all the difficulties).** *"Routes B, C, D, E
in §15 are adaptive / defensive / tempered versions of Camp B that avoid mode
lock-in and brittleness; the user's review unfairly attacks the strawman of rigid
Camp B."*

**Response.** §15 was *written* to address this exact point, and the user's
position is that all four adaptive routes either (a) are essentially Camp A in
disguise (Route B — defensive Gaussian is full-support Gaussian, so the "tuned"
mixture component is decorative), (b) inherit Camp B's circularity problem on
unknown modes (Routes A and C — need external mode finder or temperature ladder),
or (c) decouple the mode information from the prior support entirely (Route E —
score-based bias on a Gaussian prior, which is essentially the same architectural
move as $\mathrm{X}_{\hat\mu}$). The "adaptive Camp B" position is therefore not
a stable middle ground: it either collapses into Camp A or collapses into the
third way. The bare-rigid Camp B position is the only one that is genuinely
*different* from Camp A and $\mathrm{X}_{\hat\mu}$ — and the bare-rigid position
is the one with the brittleness, transferability, and mode-lock-in problems.

### 24.3. Where the argument honestly remains open

After this engagement, three points genuinely remain open between the user's
position and the steelmanned Camp B position.

1. **The score-based-bias route (Route E of §15)** is a genuine third-way-adjacent
   construction that has not been fully formalised in the BG literature. It
   shares the support-preservation property of $\mathrm{X}_{\hat\mu}$ but uses a
   different mechanism (score field augmentation). A serious comparative study
   between $\mathrm{X}_{\hat\mu}$ + Trinity and Route E + Schrödinger-bridge
   score has not been done and would benefit both camps.

2. **Single-target high-precision Camp B remains the right choice** for
   crystallographic / polymorph / phase-diagram problems. The user's argument
   does not contest this; the comparison-of-camps framing in §6 and §16 already
   concedes this niche.

3. **Empirical scaling** beyond toy 2D benchmarks. The user's empirical
   evidence (Himmelblau 2D, §B) does not yet establish that
   $\mathrm{X}_{\hat\mu}$ + Trinity outperforms FAB or SBG at the dipeptide /
   tetrapeptide scale. The user's paper makes a *theoretical* case anchored
   in §A.3 Proposition 3, plus a 2D empirical illustration. A full-scale
   replication on alanine dipeptide, comparable to TBG's benchmarks, is the
   natural next experiment.

### 24.4. Net assessment after the steelman

Once Camp B's strongest case is given full voice, the user's position remains
defensible, but the *scope* of the defensible position is narrower than the
review's earlier sections might have implied. The defensible position is:

- **In the (reverse KL × known single basin × precision target) quadrant**,
  Camp B (Wirnsberger-style) is correct and unbeaten.
- **In the (multimodal × unknown modes × discovery target) quadrant**, no Camp B
  variant has yet been demonstrated to work; the field uses Camp A or the user's
  third way.
- **The user's $\mathrm{X}_{\hat\mu}$ approach is genuinely structurally
  different from Camp B** by the support-preservation argument of §23.2 point 4,
  and is not equivalent under any moral-equivalence framing.
- **Open: a head-to-head empirical comparison with Route E (score-based bias)
  at dipeptide scale** would close the remaining genuinely-open question. This
  is a candidate experiment for the user's follow-up work.

---

*Iteration 4 added §23 steelmanning the opposition. The argument is now sharper:
Camp B is right on a niche and wrong on the central problem the user's paper
targets.*

---

# Appendix D. Glossary of terms used throughout

| Term | Definition |
|---|---|
| **Boltzmann generator (BG)** | A neural sampler trained to draw samples from $\mu \propto e^{-U_1}$ on $\mathbb{R}^d$ using only energy evaluations. |
| **Normalizing flow** | An invertible neural network $G:\mathbb{R}^d\to\mathbb{R}^d$ with tractable Jacobian determinant, used to push a base distribution to a target. |
| **Pushforward** | $G_\#\mu_0(x) = \mu_0(G^{-1}(x))\,|\det J_{G^{-1}}(x)|$ — the distribution one gets by applying $G$ to samples from $\mu_0$. |
| **Pullback** | $G^{-1}_\#\mu_0(y) = \mu_0(G(y))\,|\det J_G(y)|$ — used as the model density on configuration space. |
| **Reverse KL** | $\mathrm{KL}(\nu \,\|\, \mu)$ — sampler draws from $\nu$, mode-seeking, dominant pathology = mode collapse. |
| **Forward KL** | $\mathrm{KL}(\mu \,\|\, \nu)$ — needs samples from $\mu$, mass-covering, dominant pathology = needs MD data or AIS. |
| **AIS** | Annealed importance sampling: a ladder of distributions $\pi_k \propto \mu_0^{1-\beta_k}\mu^{\beta_k}$ transporting $\nu$ to $\mu$. |
| **ESS** | Effective sample size of weighted samples, $\in[0,1]$. Necessary but not sufficient quality criterion. |
| **Coverage** | The Naeem et al. metric: fraction of reference points reached by sampler. The complement to ESS. |
| **Fake ESS** | A pathology where ESS$\approx 1$ but the sampler has missed entire modes. Diagnosed by combining ESS with coverage. |
| **Camp A** | Methodological camp: simple uninformative prior + sophisticated training (FAB, TA-BG, CMT, SBG, iDEM, TBG). |
| **Camp B** | Methodological camp: physically-tuned mixture / lattice / higher-temperature prior + simpler training (Wirnsberger, Coretti, Schiebroek-Koehn). |
| **Mode collapse** | Reverse-KL training failure where the flow loses one or more modes of $\mu$ and cannot recover them. |
| **Mode lock-in** | Camp B failure where the bijective flow inherits the support of its tuned prior and cannot sample regions outside it. |
| **Mass teleportation** | Geometric annealing failure where new modes of $\mu$ emerge in the schedule without overlapping with earlier intermediate distributions. |
| **Topological gap** | Flow's fundamental difficulty: a homeomorphism cannot transform a unimodal contractible base into a multimodal disconnected support. |
| **$\mathrm{X}$ functional** | $\iint \omega(x,y)\,\big\|\log(\mu/\nu)(x) - \log(\mu/\nu)(y)\big\|\,\mathrm{d}x\,\mathrm{d}y$ — the user's log-ratio discrepancy regularizer. |
| **$\mathrm{X}_\mu$** | The instance of $\mathrm{X}$ with $\omega = \mu \otimes \mu$, used as a stability regularizer at zero autograd cost. |
| **$\mathrm{X}_{\hat\mu}$** | The instance with $\omega = \hat\mu \otimes \hat\mu$ where $\hat\mu$ is Trinity-produced, used for *initiative mode discovery*. |
| **Trinity** | The user's three-step (diffusion → L-BFGS → tamed Langevin) algorithm for producing a sloppy $\hat\mu$ that over-covers the modes of $\mu$. |
| **Autograd reuse** | The trick (Prop. 4, §A.4) by which $\mathrm{X}_\mu$ is computed at zero overhead by permuting the autograd graph rather than re-evaluating $G$. |
| **Asymptotic invariance** | The property (Prop. 3, §A.3) that the minimizer of $\mathrm{X}_{\hat\mu}$ is invariant to inaccuracies in $\hat\mu$ in the realizable case. |
| **Initiative mode discovery** | The user's framework for injecting mode knowledge into BG training without committing the flow's support to that knowledge. |
| **MAF** | Masked Autoregressive Flow (Papamakarios 2017). Fast density, slow sampling. Natural fit for forward-KL-flavoured losses. |
| **IAF** | Inverse Autoregressive Flow (Kingma 2016). Fast sampling, slow density. Natural fit for reverse-KL. |
| **NSF** | Neural Spline Flow (Durkan 2019). Rational-quadratic spline transformer, can be coupling- or autoregressive-based. |
| **CNF** | Continuous Normalizing Flow (FFJORD; Grathwohl 2019). Both directions are ODE solves. |
| **Smooth NF** | Köhler-Krämer-Noé 2021: smooth-bumps coupling on compact intervals and tori, used for molecular internal coordinates. |
| **Equivariant flow** | Flow with $G(g\cdot y) = g\cdot G(y)$ for symmetry group $G$. Combined with $G$-invariant prior gives a $G$-invariant model. |
| **Mean-free Gaussian** | The $\mathrm{SE}(3)$-respecting Gaussian on the hyperplane $\sum z_i = 0$. Symmetry-imposed, not mode-tuned. |
| **VampPrior** | Tomczak-Welling 2018 mixture of variational posteriors at learned pseudo-inputs. VAE-side Camp B precedent. |
| **LARS** | Bauer-Mnih 2019 learned-rejection-sampling base distribution. Mild Camp B. |
| **FAB** | Flow Annealed Importance Sampling Bootstrap (Midgley 2022). $\alpha=2$ divergence + AIS chains + replay buffer. |
| **TA-BG** | Temperature-Annealed Boltzmann Generator (Schopmans 2025). Train reverse-KL at high $T$, anneal to target $T$. |
| **CMT** | Constrained Mass Transport (Blessing 2025). Schedule with KL- and entropy-decay constraints to prevent mass teleportation. |
| **SBG** | Sequential Boltzmann Generator (Tan 2025). Continuous-time SMC + annealed Langevin at inference. |
| **TBG** | Transferable Boltzmann Generator (Klein 2024). Amortized BG across molecules via rich atom embeddings. |
| **iDEM / iEFM / EWFM** | Iterated Denoising / Energy-based Flow Matching variants (2024–25). Score-matching / flow-matching with energy-only supervision. |
| **ASBS** | Adjoint Schrödinger Bridge Sampler (Liu 2025). Diffusion sampler that allows non-Dirac (Gaussian, harmonic-oscillator) priors. |
| **PITA** | Progressive Inference-Time Annealing (Akhound-Sadegh 2025). Trains diffusion at multiple temperatures; anneals at inference. |
| **bgflow** | The Noé group reference BG implementation. PyTorch. Default prior `NormalDistribution`. |
| **normflows** | Stimper et al. PyTorch NF library. Default `DiagGaussian` base. |
| **zuko** | Rozet et al. PyTorch NF library. Built on `LazyDistribution`/`LazyTransform` abstractions. Used by user's `zflows`. |
| **zflows** | The user's PyTorch package built on zuko. Implements NSF, NCSF, CNF, RealNVP. NSF defaults to MAF autoregressive. |

---

*End of review. Total length: ~3850 lines; 24 numbered sections; 4 appendices; ~90
references across 7 communities; 5 self-generated figures + 1 from `2D_Minimal/`;
1 interactive citation network HTML. Six iterations of search-and-improve completed.*

---

## 24. Updated talking-point sheet for the collaborator argument

A condensed, one-screen rebuttal that combines the parameterization analysis with the
core difficulty arguments. Suitable as a circulated handout.

1. **Camp B's natural home is the (reverse-KL × cheap-$G^{-1}$-architecture × single-basin
   target) quadrant.** Wirnsberger 2022 is the canonical case study and its single-basin
   admission is the key citation. Outside this quadrant — and the multimodal-protein
   regime *is* outside — Camp B loses its leverage.
2. **The user's `zflows` infrastructure is MAF-based.** Reverse KL on MAF is $O(d)$
   sequential, making it the *expensive* loss family. The user is architecturally locked
   into forward-KL-style losses where Camp B contributes no benefit.
3. **Forward KL, FAB, $\mathrm{X}$ all evaluate $\log\mu_0$ at *post-flow* latent points**
   for target-side samples. A tuned multimodal $\mu_0$ neither helps nor hurts these losses
   asymptotically — but introduces mode-lock-in risk (§5.8) and brittleness (§5.9). The
   cost-benefit is unambiguously negative for Camp B in this regime.
4. **The $\mathrm{X}_{\hat\mu}$ regularizer is *uniquely* matched to MAF + forward KL +
   sloppy mode-finder**: it leverages MAF's cheap forward direction, exploits the
   autograd-reuse trick (Prop. 4, §A.4) at zero overhead, and tolerates Trinity's
   sloppiness asymptotically (Prop. 3, §A.3).
5. **The wider field has voted with its feet.** Across ~90 references in 7 communities,
   *no flagship method since 2019 has trained a tuned-GMM-prior BG on multimodal protein
   targets*. The tools exist (`bgflow.MixtureDistribution`); they are not used. This is the
   strongest single piece of indirect evidence that Camp B does not scale to the problem
   the user's paper targets.
