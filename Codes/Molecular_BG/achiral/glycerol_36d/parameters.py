"""Full-size glycerol Boltzmann generator parameters."""

MOLECULE = "glycerol"
FORMULA = "C3H8O3"
DIMENSION = 36
TEMPERATURE_KELVIN = 300.0
SEED = 0

# Match the NMA start and sharpen to the selected glycerol endpoint.
RG_PARAM_0 = (50.0, 0.2)
RG_PARAM_1 = (100.0, 0.1)

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

# Full glycerol configuration, matching NMA 30D.
VALID_SIZE = 200000
POOL_SIZE = 0
BATCH_SIZE = 10000
TRAIN_STEPS = 400
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
    # Match the completed NMA 30D stage-schedule controls exactly.
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.4,
    "tau_smc": 0.8,
    "tau_ess": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

# The monitor prints every tenth step; the driver saves ESS for every step.
MONITOR_EVERY = 10
