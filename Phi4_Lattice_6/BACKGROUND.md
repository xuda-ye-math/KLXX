# Background: the 2D lattice $\varphi^4$ theory as a mode-collapse benchmark

*Written for a reader familiar with the X-regularized forward KL framework but not with lattice field theory.*

## 1. What the target distribution is

Put one real number $\varphi_x$ (a "field value") on every site $x$ of a $6\times6$ grid with periodic boundaries, so a configuration is a vector $\varphi \in \mathbb R^{36}$. The probability of a configuration is a Boltzmann weight $\pi(\varphi) \propto e^{-S[\varphi]}$ with the action

$$S[\varphi] = \sum_{x} \Big[ -2\kappa\, \varphi_x \big(\varphi_{x+\hat e_1} + \varphi_{x+\hat e_2}\big) + \varphi_x^2 + \lambda\,(\varphi_x^2 - 1)^2 \Big] + h \sum_x \varphi_x ,$$

where $x + \hat e_\mu$ is the right/up neighbor. Three terms, three intuitions:

- $\lambda(\varphi_x^2-1)^2 + \varphi_x^2$: each site sits in a **double-well potential** with minima near $\varphi_x = \pm v$ — every single site already "wants" to choose a sign. This is the same quartic double well as the Threewell/HD benchmarks, one per site.
- $-2\kappa\,\varphi_x \varphi_{x+\hat e_\mu}$: the **hopping (ferromagnetic) coupling** rewards neighboring sites for agreeing in sign. Large $\kappa$ makes the whole lattice align.
- $h\sum_x \varphi_x$: a small **tilt** that makes one sign slightly cheaper, so the two aligned states have unequal probability (with the convention above, $h > 0$ favors $\varphi < 0$).

## 2. Why this distribution has exactly two far-apart modes

When $\kappa$ is large enough (the **broken phase**), the lattice orders: typical configurations are *all sites near $+v$* or *all sites near $-v$*. In $\mathbb R^{36}$ these are two well-separated basins, at Euclidean distance $\approx 2v\sqrt{36} = 12v$ from each other. To move continuously from one to the other the system must pass through configurations that are half-positive, half-negative — these contain a **domain wall** (an interface between regions of opposite sign), which costs action proportional to the wall length. The probability valley between the two modes is therefore exponentially deep: this is *spontaneous symmetry breaking*, the field-theory version of the two wells of the clock/HD benchmarks, except the barrier height is now a tunable collective effect rather than a hand-drawn potential.

The natural order parameter is the **magnetization**

$$m(\varphi) = \frac{1}{36}\sum_x \varphi_x ,$$

whose distribution $p(m)$ has two peaks near $\pm v$ separated by a deep valley at $m = 0$. The depth of that valley, $B = \log\big[p(\text{peak})/p(0)\big]$, *is* the mode-separation barrier in units of $k_BT$ — measured, not assumed, by the pilot run.

## 3. Why a flow trained by bare forward KL should collapse

The flow's source is an isotropic Gaussian at $\varphi = 0$ — which is exactly the *barrier top* (the symmetric point). Training drives the pushforward downhill; stochastic asymmetries in the first steps tip it into one basin, and the AIS chain that supplies the forward KL with "target samples" is initialized at the pushforward and cannot cross a $\sim 10\,k_BT$ collective barrier. From then on the loss never sees the other vacuum. The ESS stays near 1, because within the captured basin the flow matches the target well — the **fake ESS** signature: an internal diagnostic that looks perfect while half (or, with tilt, the *favored* half) of the probability mass is missing, and the estimated free-energy difference between the vacua is infinitely wrong.

The X repair: quench and temper on $S$ finds *both* aligned states (gradient descent from a melted start ends at $+v\,\mathbf 1$ or $-v\,\mathbf 1$ with roughly equal probability), so the wide-coverage measure $\hat\mu$ covers both basins, and the mixture term $\mathrm X_{(\hat\mu + \bar\nu)/2}$ — whose sample pairs come directly from $\hat\mu$, no Markov chain involved — penalizes the log-ratio discrepancy between the basins and pulls flow mass into the missing one.

## 4. The reference: parallel tempering

There is no closed-form $p(m)$ at $6\times6$, so the ground truth comes from **parallel tempering (PT)**: many MCMC chains run simultaneously at scaled actions $t\cdot S$ with $t$ from $0.15$ (nearly flat, mixes freely) to $1$ (the target), and neighboring chains swap configurations. Hot chains carry the walkers over the barrier, cold chains refine them — the same physics as the temperature ladder of Algorithm 4, used here purely as a brute-force referee. Mixing is certified by counting round trips of configurations through the ladder.

## 5. What the experiment measures

| Quantity | Meaning | Failure it exposes |
|---|---|---|
| final ESS | flow's own quality estimate | *fake* under collapse |
| reweighted $p(m>0)$ | fraction of mass in the $+$ vacuum | collapse → 0 or 1 |
| $\Delta F = -\log\frac{p(m>0)}{p(m<0)}$ vs PT | free-energy difference between vacua | the physical observable a collapsed flow gets infinitely wrong |
| $p(m)$ histogram vs PT | full order-parameter distribution | shape errors near the peaks |
| coverage vs $\hat\mu$ | mode coverage (Naeem et al.) | the silent missing mode |

Four losses are compared with identical parameters, architecture, and data (staged: bare forward KL first, then the X variants): forward KL; $+\mathrm X_\mu$; $+\mathrm X_{\hat\mu}$; the balanced mixture.

## 6. Parameters (frozen by the pilot, see `pilot_results.md`)

$6\times6$ lattice, $\lambda = 0.5$, $\kappa$ from the pilot scan (target: barrier $\approx 10\,k_BT$), tilt $h$ set so $|\Delta F| \approx 1\text{--}1.5\,k_BT$ (so mode *weights* are informative, not just coverage). Flow: NSF on $[-3,3]^{36}$, 6 coupling transforms, 16 bins, hidden $(256,256)$ — the HD-product architecture at $d = 36$; batch $500$, $2000$ Adam steps, lr $10^{-3}$. Everything fits in well under 2 GB VRAM.

## 7. How this connects to "real" lattice field theory

This $6\times6$ toy is the smallest member of the family the full benchmark targets: at larger $L$ the barrier deepens (domain walls cost more), flows for lattice theories are known to mode-collapse exactly this way, and the long-term plan adds (a) low-momentum Fourier truncation of the flow with complete-action reweighting — the cost-control trick — and (b) winding-number sectors of an XY chain with a transfer-matrix-exact referee. See `../New_Benchmark_Ideas/IDEA_REPORT.md`.
