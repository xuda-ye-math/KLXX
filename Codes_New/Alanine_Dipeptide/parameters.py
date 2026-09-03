"""L-alanine dipeptide: data-driven KL+X_pi training on the published 300 K reference."""

MOLECULE = "adp"
FORMULA = "C6H12N2O2"
DIMENSION = 60
TEMPERATURE_KELVIN = 300.0
SEED = 0

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 8
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

# One training from the source straight to the target on batches of the
# reference; the source population of VALID_SIZE only measures the final ESS.
VALID_SIZE = 1000000
BATCH_SIZE = 25000
TRAIN_STEPS = 10000
LR = 1e-3
LR_WARMUP = 25
U_CLIP = 1e3
G_CLIP = 1e2
CHUNKS = 32
CHECKPOINT = True

# KLXX only: the quench-and-temper pool of the mixture term (unused by klx and kll1).
POOL_SIZE = 0          # 0: the complete source population
MELT = 1.0
OPT_ALPHA = 1e-2
OPT_STEPS = 100
MC_DT = 1e-3
MC_STEPS_2 = 100       # MALA steps of the pool temper and of the mixture rows
MC_IMAGE_RADIUS = 3
COEFF_QT = 0.5

# Ending regularization rho = (e [kJ/mol], r [nm]) of the Boltzmann generator run.
RG_PARAM = (125.0, 0.1)
# Fraction of the largest log weights screened from the loss and the ESS.
SCREEN_FRACTION = 1e-4

# The published equilibrium reference: 10^7 Cartesian frames (nm) at 300 K.
REFERENCE = "reference/test.h5"   # relative to this folder
REFERENCE_CHUNK = 200000   # frames converted to internal coordinates per pass

# The monitor prints every tenth step; the driver keeps the ESS of every step.
MONITOR_EVERY = 10
