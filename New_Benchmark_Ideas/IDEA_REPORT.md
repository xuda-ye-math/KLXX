# Idea report: computational-physics benchmarks for X-regularized forward KL flows — beyond molecules and PIMD

**Status:** draft v1, Workflow-1 (idea discovery). PIMD rate idea shelved (`.archive/PIMD_Rate_Idea/`, judged insufficiently novel vs arXiv:2604.05303). Constraint from the user: no molecular/atomic structure (reserved for the final test), no PIMD-style path integrals; DFT explicitly floated as a direction.

---

## 0. Where the framework bites

The X-regularized forward KL framework pays off when four ingredients coexist: (i) an unnormalized target $e^{-S}$ with **isolated or near-isolated modes**; (ii) a **cheap proposal family** that can be trained fast but is structurally blind to some modes; (iii) an **exact (expensive) action available pointwise** so importance reweighting and ESS are well-defined; (iv) a **physically meaningful observable that amplifies silent mode loss** (fake ESS). The ranking below scores candidate domains on these four ingredients plus novelty and referee availability.

## 1. Ranked candidates

### #1 Lattice field theory: degenerate vacua and topological sectors (recommended)

**Target.** Euclidean lattice action, two concrete tiers.
- *Tier A — scalar $\varphi^4$ in the broken phase* on an $L \times L$ lattice,
  $$S[\varphi] = \sum_{x} \Big[ -2\kappa \sum_{\mu} \varphi_x \varphi_{x+\hat\mu} + \varphi_x^2 + \lambda_4 (\varphi_x^2 - 1)^2 \Big],$$
  with $\mathbb Z_2$-degenerate vacua $\pm v$: the field-theory double well. A small explicit breaking $h\sum_x \varphi_x$ makes vacuum *weights* informative, exactly like the asymmetric wells of the 2D suite.
- *Tier B — topological sectors:* 1D XY/O(2) chain (or 2D U(1)) where configurations split into integer **winding-number sectors** $w \in \mathbb Z$. Topological freezing — MCMC and flows failing to move between sectors — is a famous, current obstruction in lattice QCD-adjacent sampling; flow-based mitigation is an active research front through 2026.

**Cheap proposal (the user's architecture transfers verbatim).** The PIMD normal-mode truncation has an exact lattice analog: train the flow only on the **low-momentum Fourier modes** of the field ($|p| \le p_0$, an $N_0$-dimensional block), keep high-momentum modes at their free-field Gaussians, and reweight with the **complete action** $S$. The log-ratio field $z$, ESS, and all X terms follow unchanged. Truncation $N_0$ vs correlation length replaces truncation vs temperature as the new physical axis.

**Mode collapse and the X repair.** Flows for LFT are documented to mode-collapse between the $\pm v$ vacua and to freeze in a single winding sector (Nicoli et al., PRD 108, 114501; arXiv:2302.14082). **Positioning against that literature, stated carefully:** existing responses fall into three families — (a) estimator-level corrections and mode-collapse *detection* metrics (Nicoli et al.), (b) data-augmented or mixed training that adds samples from known modes (including symmetry copies) to the training set, and (c) *architectural* solutions, equivariant or topology-aware flows that build the sector structure into the map. The X framework is none of these: it is a **loss-level pairwise regularizer with a proven accuracy-floor guarantee**, in which the oracle enters only as the sampling weight of a discrepancy term — by the sample-robustness property the minimizer is insensitive to the oracle's shape, which is what mixed training (where oracle samples enter the likelihood itself and bias the fit) cannot offer. The honest claim is therefore: *the first accuracy-guaranteed loss-level mode insurance for flow samplers of LFT*, complementary to (b) and (c), not the first use of mode knowledge per se. Two concrete X-specific moves:
1. $\hat\mu$ by **QT on the lattice action** — quench finds both vacua; for winding sectors, **injected sector representatives**: in the XY/U(1) toy tiers these are closed-form minimal-action configurations, and in theories without closed forms the same slot is filled by short cooled/instanton-update MCMC per sector — the injection *mechanism* is general, the analytic convenience is tier-specific.
2. The fake ESS diagnosis, stated quantitatively: a flow frozen in the $w = 0$ sector estimates $\chi_t = \langle w^2\rangle/V$ with a negative bias equal to the entire inter-sector contribution; the benchmark *chooses* $(\beta, L)$ such that the transfer-matrix-exact sector weights make this bias an $O(1)$ fraction of $\chi_t$ (at sector-dominated parameters the bias would vanish and the test would be empty — this choice is part of the design, verified exactly before any training).

**Exact referee.** Tier A at small $L$: exhaustive checks via long parallel-tempering HMC, plus the exactly solvable Ising limit $\lambda_4 \to \infty$. Tier B in 1D: **transfer-matrix exact** partition function, sector weights, and $\chi_t$ — a true DVR-grade referee, in a regime where freezing already bites.

**Scores.** Modes 5/5, cheap-proposal fit 5/5 (Fourier truncation), exact action 5/5, observable 5/5 ($\chi_t$, vacuum weights, free-energy differences), referee 4/5, novelty 4/5 (field crowded, but loss-level oracle repair + forward KL + injected sector representatives is a distinct, defensible slice). **Risk:** the LFT flow literature is large and fast-moving; the contribution must be framed as "loss-level mode insurance for flow samplers", not "we sample LFT better than the state of the art".

### #2 Alloy configurational thermodynamics: cluster-expansion surrogate + ab-initio reweighting (the honest "DFT" mapping)

**Why DFT itself is not a sampling problem.** A DFT calculation is a deterministic SCF minimization — there is no distribution to sample. Statistical sampling enters materials physics one level up: the **configurational ensemble** of site occupations in disordered alloys, $\pi(\sigma) \propto e^{-\beta E_{\mathrm{DFT}}(\sigma)}$ over occupation vectors $\sigma \in \{0,1\}^{M}$, where each exact energy is one DFT run. The universal workaround is a **cluster expansion (CE)**: a cheap polynomial surrogate $E_{\mathrm{CE}}(\sigma)$ fitted to a few hundred DFT energies, then sampled by Metropolis MC.

**The mapping to the user's architecture is exact:** train the flow on the *cheap surrogate* ($E_{\mathrm{CE}}$, the "classical potential"), reweight with the *complete* energy (DFT, or in a self-contained benchmark a fixed reference MLIP / a higher-order CE standing in for DFT). Mode structure: competing **ordered phases** (e.g. L1$_0$ vs L1$_2$ orderings, antiphase domains) separated by interface free-energy barriers — isolated modes in configuration space at low $T$; a flow trained from a disordered source collapses onto one ordering variant. Observable: order-parameter distributions and the configurational free energy / transition temperature, both exponentially sensitive to missing variants.

**Scores.** Modes 4/5, cheap-proposal fit 5/5 (surrogate-vs-exact is *native* to the field), exact action 4/5 (DFT stand-in needed for a tractable benchmark), observable 4/5, referee 3/5 (exact enumeration only for small cells), novelty 5/5 (flows for CE/alloy ensembles: essentially unexplored as of this search). **Risk (major but not a dead end):** $\sigma$ is **discrete** — flows need one of the standard adaptations: variational dequantization, argmax/discrete flows, or a Gumbel-style continuous relaxation of the occupation variables; these are established techniques, but none exists in zflows today, so #2 carries a genuine architecture work package before the benchmark starts. Cost of honest DFT is prohibitive; the benchmark must openly use a surrogate-of-surrogate ladder.

### #3 Multimodal Bayesian inverse problems with a physical wave-equation forward model

Extend the Sensor_Array/Darcy line to a flagship: **Helmholtz/seismic source inversion or gravitational-wave-like posteriors**, where reflection and degeneracy symmetries produce well-separated posterior modes, the *cheap proposal* is a reduced-order/coarse-grid forward model and the *complete* likelihood (fine-grid PDE solve) enters only reweighting. Honest continuation of the paper's Bayesian section rather than a new domain; novelty 3/5 (flows+IS standard in GW, e.g. DINGO-IS), fit 5/5, referee 3/5 (no exact posterior; pseudo-truth by exhaustive tempered MCMC). Good fallback if #1 is judged too crowded.

### #4 SPDE rare events / stochastic Burgers instantons — path-space sampling of shock nucleation. Strong physics, but structurally a sibling of PIMD (path measures, kinks) and excluded by the same "not new here" judgment.

### #5 Knotted ring-polymer topological classes — isolated knot-type sectors, cute and unexplored, but the observable physics is niche and no cheap referee exists beyond small chains.

## 2. Recommendation

**The ranking tradeoff, stated explicitly.** #1 and #2 fail on different axes: #1's risk is *framing* (a crowded, fast-moving field where the claim must be scoped precisely as loss-level insurance), #2's risk is *prerequisite work* (a discrete-flow architecture package before any benchmark exists). The ranking below weights plug-compatibility with the existing pipeline and referee availability over raw novelty; a user who weights novelty first should invert #1 and #2 — this is flagged as the run's main human decision.

**Primary: #1 (lattice $\varphi^4$ + winding sectors), Tier A first.** It reuses the entire pipeline (NSF on $\mathbb R^{L^2}$ or on the truncated Fourier block, QT, AIS, adaptive $\kappa$/$\beta$ ladder), has a transfer-matrix-exact referee in Tier B, attacks a problem the lattice community actively cares about (topological freezing / mode collapse of flow samplers), and carries the distinctive X-framework move — *hand-injected analytic sector representatives as the wide-coverage oracle* — that no estimator-level mitigation has. **Secondary: #2** if the user prefers the materials/DFT flavor and accepts the discrete-variable architecture work; it is the most *novel* but the least plug-compatible.

## 3. Minimal first experiment (for the primary)

**Phase 0 — $6\times6$ pilot ($d = 36$, user-mandated entry point).** Full pipeline at toy scale: all four losses, QT oracle, PT-HMC reference, vacuum occupancy and $\Delta F$ vs reference. At $d = 36$ even an untruncated flow is cheap, so Phase 0 also calibrates the truncation by comparing $N_0 \in \{4, 9, 36\}$ against the untruncated flow. Everything downstream is conditional on Phase 0 reproducing the expected collapse-and-repair story.

**Cost discipline for the large lattice.** The $16\times16$ stage never trains a $256$-dimensional flow: the trained object is the $N_0$-dimensional low-momentum block ($N_0 \le 64$), high modes exact Gaussians, complete action only in the weights — the truncation is the cost model, not just a physics axis. Architecture: coupling layers with checkerboard masking (closed-form inverse at any $d$; the $O(d)$-sequential inverse of autoregressive NSF is the known wall-clock bottleneck of the HD runs) and convolutional coupling nets (translation invariance; parameters independent of $L$, enabling $6\times6 \to 16\times16$ warm starts). A single untruncated $d = 256$ run is retained only as one equal-wall-time comparison point under a hard budget cap; if it underperforms at the cap, that is the datum, not a failure of the plan.

$\varphi^4$ on $16\times 16$ ($d = 256$, matching the HD suite's top dimension), broken phase, breaking $h$ s.t. $\Delta F \approx 1\,k_BT$; four losses (forward KL; $+\mathrm X_\mu$; $+\mathrm X_{\hat\mu}$; mixture); $\hat\mu$ by QT on $S$ (quench = gradient descent finds both vacua); source = free-field Gaussian. **Parameter selection is a measured pilot, not an assertion:** starting from the known 2D $\varphi^4$ phase diagram, a preliminary PT-HMC scan (hours, CPU-feasible) locates $(\kappa, \lambda_4)$ where the constrained free energy at zero magnetization sits $8$–$12\,k_BT$ above the vacua — the barrier is *measured* on the magnetization profile of the pilot, then frozen for the benchmark. Metrics: vacuum occupancy of reweighted samples, magnetization histogram vs the same PT-HMC reference, $\Delta F$ vs reference, ESS, and the Fourier-truncation sweep $N_0 \in \{4, 16, 64, 256\}$. Expected: bare forward KL collapses to one vacuum at ESS $\approx 1$; mixture restores both at correct weights — the Threewell story, now in a genuine field theory at $d = 256$. Cost: the action is local and vectorizes; the in-house HD-product column at $d = 256$ (same flow size, comparable steps) ran within a single-GPU day, and the pilot reference is a one-off.

## 4. Pre-experiment gate checklist (reviewer conditions, all cheap, all before GPU time)

1. **Reference mixing certificate.** The PT-HMC pilot reference must demonstrate inter-vacuum (and inter-sector) mixing: report round-trip counts between modes and require $\ge 50$ per chain; a frozen reference invalidates the benchmark.
2. **Oracle coverage check.** Verify that QT's quench on $S$ recovers both vacua from $\ge 99\%$ of melted starts at the chosen parameters, and characterize any additional metastable states it finds (domain-wall configurations are expected and harmless as extra oracle mass).
3. **$\chi_t$ amplification check (Tier B).** Compute transfer-matrix-exact sector weights at the candidate $(\beta, L)$ and confirm the inter-sector contribution to $\chi_t$ is $\ge 30\%$, so a frozen flow's bias is unambiguous; the transfer-matrix computation is numerically benign at these sizes (dense $\le 512^2$ matrices).
4. **Truncation cost sanity.** Confirm that full-action reweighting per batch at $16 \times 16$ costs $\ll$ one flow gradient step (the action is local; expected orders of magnitude below the NSF inverse).
5. **Targeted novelty re-search.** One focused literature pass immediately before writing: (a) loss-level regularizers for LFT flows beyond mixed training, (b) flows for alloy configurational ensembles, to re-certify the two novelty claims this report rests on.

## 5. What this run rejects, for the wiki

- DFT-as-direct-sampling: ill-posed (deterministic minimization), mapped instead to configurational ensembles (#2).
- SPDE instantons (#4): structurally PIMD-adjacent, excluded by the user's novelty judgment.
- Molecular/atomic structure: reserved by the user for the final test.
