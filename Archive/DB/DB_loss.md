# DB loss

Suppose we are given a source distribution $\mu_0 \propto \exp(-U_0)$ on $\mathbb{R}^d$ and a target distribution $\mu_1 \propto \exp(-U_1)$ on the same space, both specified through their negative log-densities $U_0$ and $U_1$. We want to learn a bijection $G : \mathbb{R}^d \to \mathbb{R}^d$ that pushes the target onto the source, i.e., $G_{\#}\mu_1 \approx \mu_0$, or equivalently, $G_{\#}^{-1}\mu_0 \approx \mu_1$.

The pushforward density of $G_{\#}\mu_1$ at $x = G(y)$ is
$$
(G_{\#}\mu_1)(x) = \frac{\mu_1(y)}{|\det J_G(y)|} = \mu_1(G^{-1}(x)) |\det J_{G^{-1}}(x)|.
$$

The reverse KL loss to train the flow $G$ is
$$
\begin{aligned}
\mathrm{KL}(\mu_0\|G_{\#}\mu_1)
& = \mathrm{KL}(G_{\#}^{-1}\mu_0\|\mu_1) \\
& = \int_{\mathbb R^d} \mu_0(x) \log \frac{\mu_0(x)}{(G_{\#}\mu_1)(x)} \mathrm{d}x \\
& = -\int_{\mathbb R^d} \mu_0(x) \log (G_{\#}\mu_1)(x) \mathrm{d}x + \mathrm{const} \\
& = \int_{\mathbb R^d} \mu_0(x)\Big(U_1(G^{-1}(x)) - \log|\det J_{G^{-1}}(x)|\Big)\mathrm{d}x + \mathrm{const}.
\end{aligned}
$$
The reverse KL is an energy-based loss driven by Jensen's inequality, and is suitable for **adjusting the global shape of the pushforward distribution**.

The chain rule gives the gradient of $\log(G_{\#}\mu_1)$ at $x = G(y)$,
$$
\nabla\log(G_{\#}\mu_1)(x) = -J_G(y)^{-\top} \Big(\nabla U_1(y) +
\nabla \log |\det J_G(y)|\Big).
$$
Hence the forward DB loss is
$$
\begin{aligned}
\mathrm{DB}(G_{\#}\mu_1\|\mu_0)
& = \int_{\mathbb R^d} (G_{\#}\mu_1)(x) \Big|\nabla\log\mu_0(x) - \nabla\log(G_{\#}\mu_1)(x)\Big|\mathrm{d}x \\
& = \int_{\mathbb R^d} \mu_1(y) \bigg|\nabla U_0(G(y)) - J_G(y)^{-\top} \Big(\nabla U_1(y) +
\nabla \log |\det J_G(y)|\Big)\bigg|\mathrm{d}y.
\end{aligned}
$$
The KL and DB losses operate in different regimes: KL controls the global shape, while DB matches gradient details locally.
