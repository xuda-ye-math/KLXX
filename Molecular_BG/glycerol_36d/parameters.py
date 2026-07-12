"""Parameters for the 36D glycerol Boltzmann-generator run.

Edit this file to tune the experiment. ``train.py`` contains only the training
workflow and should not need parameter edits.
"""

# target: neutral glycerol, GAFF2/AM1-BCC/OBC1 at 300 K
RUN_NAME = "glycerol_36d"
BUNDLE = "glycerol_gaff2_am1bcc_obc1"
DIMENSION: int = 36
SEED: int = 0

# domain / flow: Mixed_NSF on R^25 x T^11
NSF_LIM: float = 5.0
BINS: int = 32
TRANSFORMS: int = 6
HIDDEN_FEATURES = (256, 256)
SLOPE: float = 1e-3

# basic training parameters (standard valid / pool / batch sizes)
N_VALID: int = 1000000  # validation/particle set and every stage-ESS sample count
N_POOL: int = 200000   # SMC selection and stage-training pool size
N_BATCH: int = 40000   # Adam batch size
STEPS: int = 1000      # gradient steps per stage
LR: float = 1e-3       # Adam learning rate
LR_WARMUP: int = 25    # linear warmup; controls Adam's normalized first steps
SELECTION_STEPS = (10, 25, 50, 100, 250, 500, 750, 1000)
# Full-N_VALID proposal ESS selects identity, warm start, or a sparse checkpoint.

# mixed MALA / SMC / AIS (MALA everywhere)
LADDER: int = 12        # levels of the SMC selection gate and molecular AIS
MC_STEP: float = 1e-3   # MALA step size
MC_ITERS: int = 100     # MALA steps per ladder level and post-stage advance
WRAPPED_IMAGES: int = 3 # wrapped-normal images on each side of the center
CHUNK: int = 8          # full-set device-memory partition count

# loss coefficients
COEFF_LAMBDA: float = 1.0
COEFF_ALPHA: float = 0.5
COEFF_BETA: float = 0.5

# KLXX quench-and-temper coverage pool
MELT: float = 1.0       # source already spans the whitened chart and uniform torus
OPT_STEP: float = 1e-2  # L-BFGS Armijo trial step
OPT_ITERS: int = 200    # L-BFGS quench iterations

# training screens (optimizer only; neither value changes the target)
E_CLIP: float = 1e3  # relative energy screen in the loss
G_CLIP: float = 1e2  # global gradient-norm clip before Adam
CHECKPOINT: bool = False  # rematerialize the stage loss backward pass when needed

# adaptive ladder (bg_param of the KL+X driver)
BG_PARAM = {
    "t_safe": 0.02,        # stage-1 bridge coefficient
    "shrink_factor": 0.6, # rejected stage: shrink the proposed increment
    "enlarge_factor": 1.6,# accepted stage: extrapolation growth factor
    "tau_smc": 0.5,      # SMC pre-selection threshold
    "tau_ess": 0.3,       # full-N_VALID incremental ESS threshold
    "t_tol": 1e-3,        # snap the final coefficient to one
    "max_stages": 25,     # adaptive-ladder safety cap
    "max_retry": 8,       # training attempts per stage
}

# monitoring
MONITOR_EVERY: int = 10

# ``python train.py --smoke``: one order below production workload
SMOKE_N_VALID: int = 60000
SMOKE_N_POOL: int = 12000
SMOKE_N_BATCH: int = 120
SMOKE_STEPS: int = 100
SMOKE_LADDER: int = 6
SMOKE_MC_ITERS: int = 20
SMOKE_CHUNK: int = 128  # more partitions, hence fewer samples per physical chunk
