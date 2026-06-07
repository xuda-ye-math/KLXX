"""Potential-invariant core for the adaptive-temperature Boltzmann generator
(Algorithm 4 of Paper/main.tex). Potential-specific code lives in each project
folder; this package must stay target-agnostic."""
from .boltzmann import (
    wrap_torus, identity_wrap, loss_KL, loss_X, fused_stage_loss, fused_kl_loss, bridge,
    quench_and_temper_torus, adaptive_step, train_stage,
    validation_update, run_boltzmann,
)
