## KLXX

### Normalizing flow with KL divergence

For two distributions $\mu,\nu$ on $\mathbb R^d$, the KL divergence is defined by
$$
\mathrm{KL}(\mu\|\nu) = \int_{\mathbb R^d} \mu(x) \log \frac{\mu(x)}{\nu(x)}\mathrm{d} x,
$$
which is always nonnegative by Jensen's inequality and is a standard measure of discrepancy between distributions.

Given potential functions $U_0$ and $U_1$ on $\mathbb R^d$, define the source and target distributions by $\mu_0\propto \exp(-U_0)$ and $\mu_1 \propto \exp(-U_1)$. We parameterize the inverse flow $G:\mathbb R^d\to\mathbb R^d$ mapping target to source, so that $G_{\#} \mu_1 \approx \mu_0$. Its pushforward density is
$$
    (G_{\#}\mu_1)(x) = \mu_1(G^{-1}(x)) |\det J_{G^{-1}}(x)|,
$$
where $J_{G^{-1}}(x)$ is the Jacobian matrix of $G^{-1}$ at $x$. Taking the reverse KL as the loss,
$$
\begin{aligned}
    \mathrm{KL}(\mu_0\|G_{\#}\mu_1) & = -\int_{\mathbb R^d} \mu_0(x) \log (G_{\#}\mu_1)(x)\mathrm{d}x + \mathrm{const} \\
    & = \int_{\mathbb R^d} \mu_0(x) \Big(U_1(G^{-1}(x)) - \log|\det J_{G^{-1}}(x)|\Big)\mathrm{d}x + \mathrm{const},
\end{aligned}
$$
so the loss function reads
$$
    \mathcal L_{\mathrm{reverse}}[G] = \mathbb E_{x\sim \mu_0} \Big[U_1(G^{-1}(x)) - \log|\det J_{G^{-1}}(x)|\Big].
$$
The core idea of the Boltzmann generator is to train a series of flow maps connecting an easy-to-sample prior to a multimodal target distribution.

Equivalently, one can parameterize the forward flow $F = G^{-1}:\mathbb R^d\to\mathbb R^d$ mapping source to target, $F_{\#}\mu_0\approx\mu_1$, in which case the reverse KL reads
$$
    \mathcal L_{\mathrm{reverse}}[F] = \mathbb E_{x\sim\mu_0}\Big[U_1(F(x)) - \log|\det J_F(x)|\Big],
$$
which differs from $\mathcal L_{\mathrm{reverse}}[G]$ only by the relabeling $F = G^{-1}$. We adopt the $G$-form throughout, since the auxiliary losses introduced below — the forward KL and the XX functional — are most cleanly expressed as expectations under $\mu_1$ pushed forward through $G$, avoiding repeated inversion.

## Mitigating the mode collapse failure

The reverse KL $\mathcal L_{\mathrm{reverse}}$ suffers from mode collapse when the target distribution is multimodal, so special techniques are required.
In the following, we first introduce the leading solution FAB, then propose the Jeffreys divergence and KLXX approaches. All these methods are purely energy-driven, requiring no explicit samples from the target distribution $\mu_1$ to train.

### Flow AIS Bootstrap

The flow annealed importance sampling bootstrap (FAB) [https://arxiv.org/abs/2208.01893] avoids the mode-seeking pathology of the reverse KL by minimizing the mass-covering $\alpha$-divergence with $\alpha=2$,
$$
    D_2(\mu_1\|G^{-1}_{\#}\mu_0) \propto \int_{\mathbb R^d} \frac{\mu_1(y)^2}{(G^{-1}_{\#}\mu_0)(y)}\mathrm{d}y.
$$
Writing $G = G_\theta$ and differentiating in $\theta$,
$$
\begin{aligned}
    \nabla_\theta D_2 & \propto \int_{\mathbb R^d} \nabla_\theta \frac{\mu_1(y)^2}{(G^{-1}_{\#}\mu_0)(y)}\mathrm{d}y \\
    & = -\int_{\mathbb R^d} \frac{\mu_1(y)^2}{(G^{-1}_{\#}\mu_0)(y)} \nabla_\theta\log(G^{-1}_{\#}\mu_0)(y)\mathrm{d}y \\
    & \propto -\mathbb E_{y\sim g}\Big[\nabla_\theta\log(G^{-1}_{\#}\mu_0)(y)\Big], \qquad g(y)\propto \frac{\mu_1(y)^2}{(G^{-1}_{\#}\mu_0)(y)},
\end{aligned}
$$
since the integrand is proportional to $g$. Sampling from $g$ directly is intractable, so we estimate the $g$-expectation by one-step importance sampling from $G^{-1}_{\#}\mu_0$: draw $x_i\sim\mu_0$, push to $\bar y_i = G^{-1}(x_i)$, and form weights
$$
    w_i \propto \frac{g(\bar y_i)}{(G^{-1}_{\#}\mu_0)(\bar y_i)} \propto \frac{\mu_1(\bar y_i)^2}{(G^{-1}_{\#}\mu_0)(\bar y_i)^2}.
$$
The self-normalized estimate is the gradient of the surrogate
$$
    \mathcal L_{\mathrm{FAB}}[G] = \sum_{i=1}^N \frac{\bar w_i}{\sum_{j=1}^N \bar w_j}\Big[U_0(G(\bar y_i)) - \log|\det J_G(\bar y_i)|\Big],
$$
which has the same integrand as $\mathcal L_{\mathrm{forward}}[G]$ defined below, with the $\mu_1$ integration measure replaced by the weighted empirical distribution from $g$. The bars on $\bar y_i$ and $\bar w_i$ denote a stop-gradient: both depend on $\theta$ through $G_\theta$, but the identification $\nabla_\theta\mathcal L_{\mathrm{FAB}}[G]\propto \nabla_\theta D_2$ only holds when they are treated as constants when differentiating.

Explicitly, the single-step IS algorithm proceeds in parallel to the Jeffreys $\mu_1$-sampler, with the squared log-weight (and ignoring the resulting high variance):
- Draw $x_1,\ldots,x_N \sim \mu_0$ (source);
- Push forward to $y_i = G^{-1}(x_i)$, so $y_i \sim G^{-1}_{\#}\mu_0$ (detach to stop gradient);
- Compute the log-weights $\log w_i = 2\big[U_0(x_i) - U_1(y_i) - \log|\det J_G(y_i)|\big]$;
- Resample $\{y_i\}$ by weights $w_i$ to obtain approximate samples from $g$;
- Apply a few Langevin steps on $\{y_i\}$ targeting $g$ to refine the samples.

The full FAB algorithm replaces this single-step IS with annealed importance sampling between $G^{-1}_{\#}\mu_0$ and $g$ for variance reduction and mode discovery, but the surrogate keeps the same form.

### Jeffreys divergence

The Jeffreys divergence balances the reverse KL with a forward KL to prevent mode collapse. The forward KL is constructed analogously to the reverse KL: the pushforward density of $G_{\#}^{-1} \mu_0$ is
$$
    (G^{-1}_{\#}\mu_0)(y) = \mu_0(G(y)) |\det J_{G}(y)|,
$$
so the forward KL reads
$$
\begin{aligned}
    \mathrm{KL}(G_{\#}\mu_1\|\mu_0) & = \mathrm{KL}(\mu_1\|G_{\#}^{-1}\mu_0) \\
    & = -\int_{\mathbb R^d} \mu_1(y) \log (G_{\#}^{-1}\mu_0)(y)\mathrm{d}y + \mathrm{const}  \\ 
    & = \int_{\mathbb R^d} \mu_1(y) \Big(U_0(G(y)) - \log|\det J_G(y)|\Big)\mathrm{d}y + \mathrm{const}.
\end{aligned}
$$
Hence the forward loss is
$$
\mathcal L_{\mathrm{forward}}[G] = \mathbb E_{y\sim \mu_1} \Big[U_0(G(y)) - \log|\det J_{G}(y)|\Big].
$$

The Jeffreys divergence combines the reverse and forward losses:
$$
\begin{aligned}
\mathcal L_{\mathrm{Jeffreys}}[G] & = \mathcal L_{\mathrm{reverse}}[G] + \lambda \,
\mathcal L_{\mathrm{forward}}[G] \\
& = \mathbb E_{x\sim \mu_0} \Big[U_1(G^{-1}(x)) - \log|\det J_{G^{-1}}(x)|\Big] + 
\lambda \, \mathbb E_{y\sim \mu_1} \Big[U_0(G(y)) - \log|\det J_{G}(y)|\Big].
\end{aligned}
$$
The Jeffreys approach has been studied extensively in our previous paper [https://arxiv.org/abs/2604.05303] and shown to be stable, but it explicitly requires samples from the target $\mu_1$ to compute the forward loss.

We approximate $\mu_1$ samples by importance sampling against the trained flow:
- Draw $x_1,\ldots,x_N \sim \mu_0$ (source);
- Push forward to $y_i = G^{-1}(x_i)$, so $y_i \sim G^{-1}_{\#}\mu_0$ (detach to stop gradient);
- Compute the log-weights $\log w_i = U_0(x_i) - U_1(y_i) - \log|\det J_G(y_i)|$;
- Resample $\{y_i\}$ by weights $w_i$ to obtain approximate samples from $\mu_1$;
- Apply a few Langevin steps on $\{y_i\}$ to obtain approximate samples of $\mu_1$.

Because of the resampling, the approximation to $\mu_1$ is biased, so a large batch size is recommended; the parameter $\lambda$ then becomes a trade-off between overcoming mode collapse and sampling accuracy.


## XX: not a divergence

The forward KL requires accurate samples from $\mu_1$, which is precisely what we wish to draw. To bypass this, we propose the XX functional
$$
\mathrm{XX}(\mu\|\nu) = \frac12\iint_{\mathbb R^d\times\mathbb R^d} \mu(x) \mu(x')
\bigg|\log \frac{\mu(x)}{\nu(x)} - \log \frac{\mu(x')}{\nu(x')}\bigg|\mathrm{d}x \mathrm{d}x'.
$$
The XX also measures the discrepancy between $\mu$ and $\nu$ and superficially resembles the standard KL, but operates in a fundamentally different way:
- XX is nonnegative because its integrand is nonnegative, not via Jensen's inequality.
- XX vanishes iff $\log \frac{\mu(x)}{\nu(x)}$ is constant (a.e.), which forces $\mu \equiv \nu$. In other words, minimizing XX amounts to reducing the variation of the log-likelihood ratio $\log\frac{\mu(x)}{\nu(x)}$.

Moreover, one can verify the KL-like invariance
$$
    \mathrm{XX}(F_{\#}^{-1}\mu\|\nu) = \mathrm{XX}(\mu\|F_{\#}\nu)
$$
for any invertible flow map $F$. Given source and target distributions $\mu_0$ and $\mu_1$, the forward XX expands as
$$
\begin{aligned}
    \mathrm{XX}(F_{\#}^{-1}\mu_1\| \mu_0) & = \frac12\iint_{\mathbb R^d\times\mathbb R^d} \mu_1(y) \mu_1(y') 
    \bigg|\log \frac{\mu_1(y)}{\mu_0(F^{-1}(y))|\det J_{F^{-1}}(y)|} - 
    \log \frac{\mu_1(y')}{\mu_0(F^{-1}(y'))|\det J_{F^{-1}}(y')|}\bigg| \mathrm{d} y \mathrm{d} y' \\
    & = \frac12\iint_{\mathbb R^d\times\mathbb R^d} \mu_1(y) \mu_1(y')  \Big|
    \big(U_0(F^{-1}(y)) - U_1(y) - \log|\det J_{F^{-1}}(y)|\big) - 
     \big(U_0(F^{-1}(y')) - U_1(y') - \log|\det J_{F^{-1}}(y')|\big) 
    \Big| \mathrm{d} y \mathrm{d} y'.
\end{aligned}
$$

For notational convenience, parameterize the flow map $G$ such that $G_{\#}\mu_1 \approx \mu_0$, so that the reverse KL and forward XX read
$$
    \boxed{\mathrm{KL}(\mu_0\|G_{\#} \mu_1) = \mathbb E_{x\sim \mu_0} \Big[U_1(G^{-1}(x)) - \log|\det J_{G^{-1}}(x)|\Big], }
$$
and
$$
    \boxed{\mathrm{XX}(G_{\#} \mu_1\| \mu_0) = \frac12\mathbb E_{y,y'\sim \mu_1} \Big[\Big|
    \big(U_0(G(y)) - U_1(y) - \log|\det J_{G}(y)|\big) - 
     \big(U_0(G(y')) - U_1(y') - \log|\det J_{G}(y')|\big) 
    \Big|\Big].}
$$
The XX has exactly the same scale as the usual KL, so standard KL training parameters can be applied to XX without additional tuning.

The choice of the forward XX (which requires target samples) is actually subtle. By the invariance, minimizing the forward XX is equivalent to minimizing $\mathrm{XX}(\mu_1\| G^{-1}_{\#}\mu_0)$, i.e., fitting $G^{-1}_{\#}\mu_0$ towards $\mu_1$. Since $G$ is a diffeomorphism, $G^{-1}_{\#}\mu_0$ stays connected and inherits artificial bridges between the modes of $\mu_1$, and on these bridges the XX integrand can be very large. The reverse $\mathrm{XX}(\mu_0\| G_{\#}\mu_1)$ would weight these bridges in its integration measure, making them a serious obstacle to training. The forward XX, in contrast, integrates against $\mu_1$, whose density $\propto \exp(-U_1)$ decays faster than the log-likelihood error grows on these bridges, so the bridge contribution to the integral is negligible and training is stable on multimodal targets.

By contrast, the reverse KL is far less sensitive to these bridges: via Jensen's inequality, it only asks that the expected log-ratio under $\mu_0$ be small, whereas XX demands the log-ratio be constant *everywhere* — a much stricter condition that the bridges violate.

In practice we cannot directly sample from $\mu_1$, so we approximate $\mu_1$ samples by importance sampling against the trained flow:
- Draw $x_1,\ldots,x_N \sim \mu_0$ (source);
- Push forward to $y_i = G^{-1}(x_i)$, so $y_i \sim G^{-1}_{\#}\mu_0$ (detach to stop gradient);
- Compute the log-weights $\log w_i = U_0(x_i) - U_1(y_i) - \log|\det J_G(y_i)|$;
- Resample $\{y_i\}$ by weights $w_i$ to obtain approximate samples from $\mu_1$;
- Apply one Langevin step on $\{y_i\}$ and reweight each sample by its Metropolis acceptance ratio $\alpha_i$, yielding weighted unbiased samples of $\mu_1$.

A small step size is still recommended: it keeps the samples within $\mu_1$'s support, preventing drift onto the bridges that would corrupt the $\mu_1$-weighted suppression argument established above.

## KLXX: the joint framework

The reverse KL and the forward XX target opposite failure modes — reverse KL is mode-seeking while XX is mass-covering — so we combine them into the KLXX framework, defined by the joint loss
$$
\boxed{\mathcal L_{\mathrm{KLXX}}[G] = \mathrm{KL}(\mu_0\|G_{\#}\mu_1) + \mathrm{XX}(G_{\#}\mu_1\|\mu_0).}
$$
Because XX shares the same scale as the standard KL, the two terms can be summed without any tuning weight.