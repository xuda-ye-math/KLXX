# Optimality condition under a biased outer measure

## Setup

The exact loss is
$$
\mathcal L[\nu] \;=\; \int\!\mu(x)\log w(x)\,\mathrm{d}x
\;+\; \lambda\!\iint\!\mu(x)\mu(y)\,\bigl|\log w(x) - \log w(y)\bigr|\,\mathrm{d}x\,\mathrm{d}y,
\qquad w(x) := \frac{\mu(x)}{\nu(x)}.
$$
In practice the outer $\mu$ is replaced by an approximation $\tilde\mu$ ($\int\tilde\mu = 1$, $\tilde\mu\geqslant 0$), while $w(x) = \mu(x)/\nu(x)$ is preserved:
$$
\widetilde{\mathcal L}[\nu] \;=\;
\int\!\tilde\mu(x)\log w(x)\,\mathrm{d}x
\;+\;
\lambda\!\iint\!\tilde\mu(x)\tilde\mu(y)\,\bigl|\log w(x)-\log w(y)\bigr|\,\mathrm{d}x\,\mathrm{d}y.
$$

## 1. First variations

Perturb $\nu\to\nu+\varepsilon\sigma$, $\int\sigma\,\mathrm{d}x=0$; then $\delta\log w(x) = -\sigma(x)/\nu(x)$.

**KL piece.**
$$
\frac{\delta\widetilde{\mathrm{KL}}}{\delta\nu}(x) \;=\; -\,\frac{\tilde\mu(x)}{\nu(x)}.
$$

**$\mathrm X_\mu$ piece.** Using $\partial_u|u| = \mathrm{sgn}(u)$ and $\mathrm{sgn}(\log w(x)-\log w(y)) = \mathrm{sgn}(w(x)-w(y))$,
$$
\frac{\delta\widetilde{\mathrm X}_\mu}{\delta\nu}(x) \;=\; -\,\frac{2\,\tilde\mu(x)\,\widetilde S(x)}{\nu(x)},
\qquad
\widetilde S(x) \;:=\; \int\!\tilde\mu(y)\,\mathrm{sgn}(w(x)-w(y))\,\mathrm{d}y.
$$

**Combined.**
$$
\frac{\delta\widetilde{\mathcal L}}{\delta\nu}(x) \;=\; -\,\frac{\tilde\mu(x)}{\nu(x)}\bigl[\,1 + 2\lambda\,\widetilde S(x)\,\bigr].
$$

## 2. $\nu$-mean

$$
\Big\langle\frac{\delta\widetilde{\mathcal L}}{\delta\nu}\Big\rangle_\nu
\;=\; -\!\int\!\tilde\mu\,\mathrm{d}x \;-\; 2\lambda\!\iint\!\tilde\mu(x)\tilde\mu(y)\,\mathrm{sgn}(w(x)-w(y))\,\mathrm{d}x\,\mathrm{d}y
\;=\; -1,
$$
since $\int\tilde\mu = 1$ and the double integral vanishes by antisymmetry.

## 3. Fisher–Rao gradient

$$
\mathrm{grad}^{\mathrm{FR}}\!\bigl[\widetilde{\mathcal L}\bigr](\nu)(x)
\;=\; \nu(x)\!\left(\frac{\delta\widetilde{\mathcal L}}{\delta\nu}(x) + 1\right)
\;=\; \nu(x) \;-\; \tilde\mu(x)\bigl[\,1 + 2\lambda\widetilde S(x)\,\bigr].
$$

## 4. Optimality condition

Setting $\mathrm{grad}^{\mathrm{FR}}\widetilde{\mathcal L}(\nu^\star) = 0$:
$$
\boxed{\;\nu^\star(x) \;=\; \tilde\mu(x)\left[\,1 + 2\lambda\!\int_{\mathbb R^d}\!\tilde\mu(y)\,\mathrm{sgn}\!\left(\frac{\mu(x)}{\nu^\star(x)} - \frac{\mu(y)}{\nu^\star(y)}\right)\mathrm{d}y\,\right].\;}
$$

## 5. Accuracy: $\mathrm{KL}(\mu\|\nu^\star) \leqslant \mathrm{KL}(\mu\|\tilde\mu)$ for $\lambda \leqslant 1/2$

Write $w^\star(x) := \mu(x)/\nu^\star(x)$ and $\widetilde S^\star(x) := \int\tilde\mu(y)\,\mathrm{sgn}(w^\star(x)-w^\star(y))\,\mathrm{d}y$, so that the optimality condition reads $\nu^\star = \tilde\mu(1+2\lambda\widetilde S^\star)$. Since $\widetilde S^\star \in [-1,1]$,
$$
1 + 2\lambda\widetilde S^\star(x) \;\geqslant\; 1 - 2\lambda \;\geqslant\; 0
\qquad\text{whenever}\qquad \lambda \leqslant 1/2,
$$
so $\nu^\star$ is a (nonnegative) probability density in this regime.

**Theorem.** *If $\lambda \in [0,1/2]$ and $\mu \ll \tilde\mu$, then*
$$
\boxed{\;\mathrm{KL}(\mu\|\tilde\mu) - \mathrm{KL}(\mu\|\nu^\star)
\;\geqslant\; \lambda\!\iint\!\tilde\mu(x)\tilde\mu(y)\,\left|\frac{\mu(x)}{\nu^\star(x)} - \frac{\mu(y)}{\nu^\star(y)}\right|\,\mathrm{d}x\,\mathrm{d}y \;\geqslant\; 0.\;}
$$

*Proof.* Compute the gap directly:
$$
\mathrm{KL}(\mu\|\tilde\mu) - \mathrm{KL}(\mu\|\nu^\star)
\;=\; \int\!\mu(x)\log\frac{\nu^\star(x)}{\tilde\mu(x)}\,\mathrm{d}x
\;=\; \int\!\mu(x)\log\!\bigl[1 + 2\lambda\widetilde S^\star(x)\bigr]\,\mathrm{d}x.
$$
Substitute $\mu = \nu^\star w^\star = \tilde\mu(1+2\lambda\widetilde S^\star)w^\star$ to obtain
$$
\mathrm{KL}(\mu\|\tilde\mu) - \mathrm{KL}(\mu\|\nu^\star)
\;=\; \int\!\tilde\mu(x)\,w^\star(x)\,(1+2\lambda\widetilde S^\star(x))\log\!\bigl[1 + 2\lambda\widetilde S^\star(x)\bigr]\,\mathrm{d}x.
$$
Apply the elementary inequality $u\log u \geqslant u - 1$ for $u \geqslant 0$ to $u = 1+2\lambda\widetilde S^\star(x) \geqslant 0$:
$$
\mathrm{KL}(\mu\|\tilde\mu) - \mathrm{KL}(\mu\|\nu^\star)
\;\geqslant\; \int\!\tilde\mu(x)\,w^\star(x)\cdot 2\lambda\widetilde S^\star(x)\,\mathrm{d}x
\;=\; 2\lambda\!\iint\!\tilde\mu(x)\tilde\mu(y)\,w^\star(x)\,\mathrm{sgn}(w^\star(x)-w^\star(y))\,\mathrm{d}x\,\mathrm{d}y.
$$
Symmetrize $(x,y)\!\leftrightarrow\!(y,x)$:
$$
\iint\!\tilde\mu(x)\tilde\mu(y)\,w^\star(x)\,\mathrm{sgn}(w^\star(x)-w^\star(y))\,\mathrm{d}x\,\mathrm{d}y
\;=\; \frac{1}{2}\!\iint\!\tilde\mu(x)\tilde\mu(y)\,|w^\star(x)-w^\star(y)|\,\mathrm{d}x\,\mathrm{d}y \;\geqslant\; 0.
$$
Combining gives the claimed bound. $\quad\square$

The gap is strictly positive unless $w^\star$ is $\tilde\mu$-a.e.\ constant, i.e.\ $\nu^\star \propto \mu$ on the support of $\tilde\mu$ — exactly the case where the $\mathrm X_\mu$ regulariser is unnecessary.
