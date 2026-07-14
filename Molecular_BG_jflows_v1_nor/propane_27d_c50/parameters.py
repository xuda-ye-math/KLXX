"""Parameters for the propane adaptive KLXX c50 test."""

MOLECULE = "propane"
FORMULA = "C3H8"
DIMENSION = 27
SEED = 200

ENERGY_CUT_KJ_MOL = 50.0
ENERGY_SCALE_KJ_MOL = 50.0
TAIL_FRACTION = 0.01

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

N_VALID = 400000
POOL_SIZE = 0
BATCH_SIZE = 50000
TRAIN_STEPS = 500
LR = 1e-4
LR_WARMUP = 40

LADDER = 8
MC_DT = 1e-2
MC_STEPS = 50
MC_IMAGE_RADIUS = 3
CHUNKS = 64

MELT = 1.0
OPT_ALPHA = 1e-2
OPT_STEPS = 200
COEFF_LAMBDA = 1.0
COEFF_ALPHA = 0.5
COEFF_BETA = 0.5
E_CLIP = 1e3
G_CLIP = 1e2
CHECKPOINT = True

BG_PARAM = {
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.5,
    "tau_smc": 0.7,
    "tau_ess": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

MONITOR_EVERY = 10

# Independent physical OpenMM reference. CUDA runs in single precision and
# all persisted floating-point arrays remain float32.
OPENMM_SEED = 141421
OPENMM_BURNIN_STEPS = 100000
OPENMM_FRAMES = 50000
OPENMM_STRIDE_STEPS = 100
OPENMM_DT_FS = 0.5
OPENMM_FRICTION_PER_PS = 1.0

DIHEDRAL_BINS = 100
DIHEDRAL_SMOOTH_SIGMA = 1.25
