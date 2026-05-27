# boundary of the domain
SIGMA = 0.25            # standard deviation of the isotropic Gaussian source mu_0
PLT_LIM = 2.0          # half-width of the plot window; axes span [-PLT_LIM, +PLT_LIM]
NSF_LIM = 2.0          # half-width of the NSF spline domain; the flow acts on [-NSF_LIM, +NSF_LIM]^2

# NSF flow architecture
BINS: int = 32                       # number of rational-quadratic spline bins per coupling transform
TRANSFORMS: int = 6                  # number of coupling transforms stacked in the flow
HIDDEN_FEATURES = (128, 128)         # widths of the hidden layers in each coupling-transform MLP

# training parameters
N_TRAIN: int = 100000   # size of the source-sample pool drawn once and reused across steps
N_VALID: int = 100000   # size of the fresh source batch used for final ESS / coverage evaluation
BATCH:   int = 2000     # number of source samples per training step
STEPS:   int = 1000    # number of Adam optimization steps
LR:      float = 1e-3  # Adam learning rate
