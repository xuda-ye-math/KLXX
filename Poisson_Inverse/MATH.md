# Mathematical setup: screened-Poisson Bayesian source inversion

## 1. The unknown and its prior

The unknown is a real periodic field on the unit square,

    v(x; theta) = sum_{m=1}^{d} theta_m phi_m(x),     x in [0,1]^2,

with the real Fourier basis (half-plane convention, each wave vector counted once)

    phi_0(x) = 1                                  (constant mode, k = 0)
    phi_m(x) = sqrt(2) cos(2 pi k_m . x)   or   sqrt(2) sin(2 pi k_m . x),

where k_m = (k1, k2) runs over the half-plane k1 > 0, or k1 = 0 and k2 > 0, so that
{phi_m} is an L^2([0,1]^2)-orthonormal family. The modes are enumerated by
increasing |k|^2 (cos before sin at equal k, lexicographic tie-break), and an
"m x m" set is DEFINED as the first m^2 modes of this enumeration -- an isotropic
low-frequency ball holding exactly m^2 real coefficients, the low block
automatically preceding the extension:

    d_low = M_LOW^2 trained modes,   d_full = M_FULL^2 total. Smoke geometry: 4x4 in 8x8 (16 in 64); headline: 6x6 in 12x12
(36 in 144).

Prior: independent zero-mean Gaussians with polynomially decaying variances,

    theta_m ~ N(0, sigma_m^2),   sigma_m^2 = A / (1 + |k_m|^2)^s,   s = 2.

The decay exponent s is a free design choice; s = 2 (Matern-like, H^{s-1}-regular
draws) is chosen deliberately steep so that the high-mode coefficients are small --
the technique under test requires fast high-frequency decay.

## 2. Whitening (hard rule)

All sampling dynamics operate on whitened coordinates

    xi_m = theta_m / sigma_m,     theta = S^{1/2} xi,   S = diag(sigma_m^2),

so the prior is exactly N(0, I_d) and every coordinate is O(1). Langevin, quench
and temper, SMC, the flow, and the identity extension all act on xi; the scaling
S^{1/2} appears only inside the likelihood's forward solve. Rationale: the
Langevin/QT implementations use a single isotropic step size, which is correct
only when all coordinates share the same scale; raw theta_m at |k| = 6 has
sigma_m^2 ~ A/37^2, three orders below the constant mode.

## 3. Forward model and discretization

PDE (screened Poisson, periodic boundary conditions):

    (-Lap + c^2) u(x) = g(x; theta),
    g(x; theta) = G0 (1 + delta cos(alpha v(x; theta))) + eps v(x; theta).

The screening c^2 > 0 removes the k = 0 null space (no compatibility condition)
and adds uniform smoothing.

Discretization: collocation on the N x N cell-centered grid x_ij = ((i+1/2)/N,
(j+1/2)/N), N = N_GRID. The source g is evaluated pointwise on the grid (the basis
matrix Phi in R^{N^2 x d} is precomputed once), and the PDE is solved exactly in
the discrete Fourier space:

    u_hat(k) = g_hat(k) / (|2 pi k|^2 + c^2),     k in fftfreq(N)^2,

via one fft2/ifft2 pair per batch. Dealiasing: N >= 4 * (M_FULL/2) = 2 M_FULL
resolves the products cos(alpha v) up to the band limit of v without aliasing the
observed low-frequency content (alpha v is band-unlimited through the cosine; the
grid truncation error decays with N and is part of the fixed, exactly-evaluable
discrete model -- the Bayesian target is DEFINED on the discrete forward).

Observation: u interpolated bilinearly at N_S sensors on a ring of radius r
centered at (1/2, 1/2),

    F(theta) = (u(x_s))_{s=1}^{N_S},
    y_obs = F(theta_truth) + sigma_obs eta,   eta ~ N(0, I_{N_S}).

## 4. Posterior, training target, and the bridge

Whitened potentials (negative log densities up to constants):

    U_full(xi)    = 1/2 |xi|^2 + Phi(S^{1/2} xi),                xi in R^{d_full},
    U_low(xi_L)   = 1/2 |xi_L|^2 + Phi(S^{1/2} [xi_L; 0]),       xi_L in R^{d_low},
    Phi(theta)    = |y_obs - F(theta)|^2 / (2 sigma_obs^2).

The Boltzmann generator (paper Algorithm 4) bridges the whitened prior
U_0 = 1/2|xi|^2 to U_low. Because the prior term is common,

    U_t = (1 - t) U_0 + t U_low = 1/2 |xi|^2 + t Phi_low,

the temperature ladder tempers exactly the likelihood -- classical Bayesian
annealing. This identity concerns the low-mode TRAINING problem only.

## 5. The extension test (the new idea)

Let G = G_1^{-1} o ... o G_K^{-1} be the composed trained generator on R^{d_low}.
The extended generator on R^{d_full} is

    G_ext(xi) = [ G(xi_L) ; xi_H ],     xi = [xi_L; xi_H] ~ N(0, I_{d_full}),

i.e. identity on the whitened high modes. Its exact density is known (the high
block contributes log-Jacobian zero), so the full-posterior importance weight is

    log w(xi) = U_0_full(xi) - U_full(G_ext(xi)) + sum_k ladj_k(xi_L),

with ladj_k the per-stage inverse log-Jacobians on the low block. Reported:
full-dimension ESS of w, well occupancies, reweighted well weights, Delta F
against the PT-MALA referee at d_full.

Validity heuristic: the posterior factorizes approximately as
pi(xi) ~ pi_low(xi_L) x N(xi_H; 0, I) when the likelihood is insensitive to
xi_H. Sensitivity of the misfit to mode m scales like the squared Green's
multiplier at frequency k_m,

    H^lik_mm ~ |F'|^2 / (|2 pi k_m|^2 + c^2)^2 x sigma_m^2  (whitened),

against whitened prior curvature 1 (identity). With sigma_m^2 ~ |k|^{-4} and the
Green's factor |k|^{-4}, the whitened curvature ratio decays like |k|^{-8}: the
nonlinearity cos(alpha v) shifts a mode-m perturbation into a band around k_m
(bandwidth of v), so the smoothing applies at frequency ~ k_m -- unlike the
lattice phi^4 action, whose site-wise quartic coupled all modes at O(1).

## 6. Multimodality

cos(alpha v) is invariant under

    (i)  v -> -v          (theta -> -theta: sign flip),
    (ii) v -> v + 2 pi n / alpha,  n in Z  (constant-mode shift
                                            theta_0 -> theta_0 + 2 pi n / alpha).

The prior restricts (ii) to |n| <~ alpha sigma_0 / (2 pi) reachable shifts; alpha
and A are chosen on the pilot so that approximately 3 shifts x 2 signs = 6 wells
carry non-negligible prior mass. The tilt eps v in the source breaks (i) exactly
(g is no longer even in v) and weakly orders the shift wells, so the well weights
are nontrivial numbers the sampler must earn -- the phi^4 design principle. The
referee (parallel tempering MALA at d_full over U_t = prior + t Phi) certifies
the true well weights with roundtrip counts.

## 7. Truncation-ceiling diagnostic (gate, before any training)

(a) Whitened curvature ratio rho_m = H^lik_mm(xi*) (finite differences at the
    truth and at 0) for extension modes; demand max rho_m < 0.1.
(b) Decay curve Var_{xi_m ~ N(0,1)} U_full vs |k_m|, other modes at truth.
(c) Training-free ceiling: freeze xi_L at referee posterior samples, draw
    xi_H ~ N(0, I), measure ESS of w against U_full; demand > 0.5.
(c') Stress (reviewer round 1): repeat (c) with xi_L from HEATED referee runs
    (t = 0.7, 0.5) -- an imperfect flow explores such regions; demand no collapse.
