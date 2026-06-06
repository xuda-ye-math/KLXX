# 2D phi^4 lattice, 6x6 periodic (D = 36), broken Z2 phase
L = 8                   # lattice side; D = L*L sites
D = L * L

# action S[phi] = sum_x [ -2*KAPPA*phi_x*(phi_{x+e1}+phi_{x+e2}) + phi_x^2
#                         + LAMBDA*(phi_x^2-1)^2 ] + H*sum_x phi_x
# KAPPA and H are FROZEN from the pilot scan (pilot.py); see pilot_results.md
KAPPA = 0.40            # frozen by pilot8: barrier 10.19 kT at h=0, v=1.082
LAMBDA = 0.50           # quartic self-coupling
H = 0.0144              # frozen by pilot8: Delta F = -1.82 kT, p_+ = 0.139, roundtrips 3687

# boundary of the domain
SIGMA = 0.5             # standard deviation of the isotropic Gaussian source mu_0
PLT_LIM = 2.0           # half-width of magnetization plot window
NSF_LIM = 3.0           # half-width of the NSF spline domain per site

# NSF flow architecture
BINS: int = 16                       # rational-quadratic spline bins per coupling transform
TRANSFORMS: int = 6                  # coupling transforms stacked in the flow
HIDDEN_FEATURES = (256, 256)         # hidden widths of each coupling-transform MLP

# training parameters
N_TRAIN: int = 100000   # source-sample pool drawn once and reused across steps
N_VALID: int = 100000   # fresh source batch for final ESS / coverage evaluation
BATCH:   int = 500
STEPS:   int = 2000     # Adam optimization steps
LR:      float = 1e-3   # Adam learning rate
