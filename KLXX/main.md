## KLXX

### Normalizing flow with KL divergence

For two distributions $\mu,\nu$ on $\mathbb R^d$, the KL divergence is defined by
$$
\mathrm{KL}(\mu\|\nu) = \int_{\mathbb R^d} \mu(x) \log \frac{\mu(x)}{\nu(x)}\mathrm{d} x,
$$
which is always nonnegative by Jensen's inequality. The KL divergence is a standard measure of discrepancy between distributions.

Given the potential functions $U_0$ and $U_1$ on $\mathbb R^d$, define the source and target distributions by $\mu_0\propto \exp(-U_0)$ and $\mu_1 \propto \exp(-U_1)$. The goal of the normalizing flow is to find a map $F:\mathbb R^d\to\mathbb R^d$ such that $F_{\#} \mu_0 \approx \mu_1$, or equivalently, $F_{\#}^{-1} \mu_1 \approx \mu_0$. 
The pushforward density of $F_{\#}^{-1} \mu_1$ is given by
$$
    (F_{\#}^{-1}\mu_1)(x) = \mu_1(F(x)) |\det J_F(x)|,
$$
where $J_F(x)$ is the Jacobian matrix of $F$ at $x$. To find the flow $F$, we choose the reverse KL as the loss:
$$
\begin{aligned}
    \mathrm{KL}(\mu_0\|F_{\#}^{-1}\mu_1) & = \mathrm{KL}(F_{\#}\mu_0\|\mu_1) \\
    & = -\int_{\mathbb R^d} \mu_0(x) \log (F_{\#}^{-1}\mu_1)(x)\mathrm{d}x + \mathrm{const} \\
    & = \int_{\mathbb R^d} \mu_0(x) \Big(U_1(F(x)) - \log|\det J_F(x)|\Big)\mathrm{d}x +
    \mathrm{const}.
\end{aligned}
$$
That is, the loss function is 
$$
\mathcal L_{\mathrm{reverse}}[F] = \mathbb E_{x\sim \mu_0} \Big[U_1(F(x)) - \log|\det J_F(x)|\Big].
$$

## Mode collapse failure

The reverse KL $\mathcal L_{\mathrm{reverse}}$ suffers from mode collapse when the target distribution is multimodal, and several approaches have been proposed to address it. For example, FAB replaces the loss by an $\alpha=2$ Rényi divergence estimated via annealed importance sampling, and Jeffreys Flow augments the reverse KL with a forward KL term on parallel-tempering samples.

The forward KL is constructed analogously to the reverse KL. The density of the pushforward distribution $F_{\#} \mu_0$ is given by
$$
    (F_{\#}\mu_0)(y) = \frac{\mu_0(x)}{|\det J_F(x)|} = \mu_0(F^{-1}(y)) |\det J_{F^{-1}}(y)|, \qquad y = F(x),
$$
so the forward KL divergence becomes
$$
\begin{aligned}
    \mathrm{KL}(F_{\#}^{-1}\mu_1\|\mu_0) & = \mathrm{KL}(\mu_1\|F_{\#}\mu_0) \\
    & = -\int_{\mathbb R^d} \mu_1(y) \log (F_{\#}\mu_0)(y)\mathrm{d}y + \mathrm{const}  \\ 
    & = \int_{\mathbb R^d} \mu_1(y) \Big(U_0(F^{-1}(y)) - \log|\det J_{F^{-1}}(y)|\Big)\mathrm{d}y + \mathrm{const}.
\end{aligned}
$$
Hence the loss function is
$$
\mathcal L_{\mathrm{forward}}[F] = \mathbb E_{y\sim \mu_1} \Big[U_0(F^{-1}(y)) - \log|\det J_{F^{-1}}(y)|\Big].
$$
The forward KL effectively mitigates mode collapse, but requires accurate samples from $\mu_1$ to remain unbiased, so its applicability is limited by the available sample quality. Note the symmetry: $\mathcal L_{\mathrm{reverse}}[F]$ and $\mathcal L_{\mathrm{forward}}[F]$ are related by exchanging $\mu_0 \leftrightarrow \mu_1$, $U_0 \leftrightarrow U_1$, and $F \leftrightarrow F^{-1}$.


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