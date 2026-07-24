"""Full-size (2R,3R)-2,3-butanediol Boltzmann-generator parameters."""

MOLECULE = "rr_2_3_butanediol"
FORMULA = "C4H10O2"
DIMENSION = 42
TEMPERATURE_KELVIN = 300.0
SEED = 0

# Match neutral diethanolamine and sharpen to the selected endpoint.
RG_PARAM_0 = (50.0, 0.2)
RG_PARAM_1 = (100.0, 0.1)

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

# Full (2R,3R)-2,3-butanediol configuration, matching diethanolamine 48D.
VALID_SIZE = 240000
POOL_SIZE = 0
BATCH_SIZE = 12000
TRAIN_STEPS = 250
LR = 1e-3
LR_WARMUP = 25
U_CLIP = 1e3
G_CLIP = 1e2

LADDER = 8
MC_DT = 1e-3
MC_STEPS = 100
MC_IMAGE_RADIUS = 3
CHUNKS = 32

MELT = 1.0
OPT_ALPHA = 1e-2
OPT_STEPS = 200
CHECKPOINT = True
INITIALIZE_FROM_IDENTITY = True

BG_PARAM = {
    # Match the neutral diethanolamine bridge controls exactly.
    "t_safe": 0.1,
    "shrink_factor": 0.6,
    "enlarge_factor": 1.4,
    "tau_smc": 0.8,
    "tau_ess": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

# The monitor prints every tenth step; the driver saves ESS for every step.
MONITOR_EVERY = 10
