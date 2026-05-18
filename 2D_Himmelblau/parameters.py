# boundary of the domain
SIGMA = 2.0 # deviation of Gaussian prior
PLT_LIM = 6.0
NSF_LIM = 6.0

# training parameters
N_TRAIN: int = 50000   # number of training samples
N_VALID: int = 50000   # number of validation samples
LR = {100: 1e-3, 1000: 3e-3}      # learning rate per batch (sqrt scaling: ~sqrt(10) per 10x batch)
STEPS = {100: 1000, 1000: 1000}   # uniform step count across batch sizes