"""KL + X over the equal mixture of ground-truth subsamples and the detached pushforward.

A data-driven variant of the KLXX mixture term for this folder: the target
batch of the forward KL is drawn from the target sample set, and the mixture
batch of the variation term is resampled in equal proportion from a second,
independent subsample of the same set (the ``hat_pi`` half, in place of the
quench-and-temper pool) and from the detached pushforward of the source batch
through the current map (the ``bar_nu`` half). There is no target variation
term and no pool. Screens, Adam, clips, warm-up, the manual rejection, and the
batch ESS of the pushforward are those of ``jflows_md.train``.

    trained, batch_ess = train_klxm_G(
        x_valid, source, target, flow, domain, target_data,
        batch_size, steps_total, lr, coeff_theta=1.0, coeff_alpha=0.5, ...,
    )

``coeff_alpha`` is the proportion of the mixture drawn from the target data;
``mixture_data`` may supply a different array for that half (external data),
otherwise ``target_data`` serves both roles.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jflows.train import _masked_mean, _variation
from jflows.utils import resample
from jflows_md.train import (
    _adam, _energy_keep, _learning_rate, _pushforward_weights, _run_chunks, _screen_top,
)
from jflows_md.utils.screen import SCREEN_FRACTION, compute_ESS_log


__all__ = ["train_klxm_G"]


@eqx.filter_jit
def _chunk(
    x_valid, target_data, mixture_data, source, target, state, static, domain, key, *,
    step_offset, chunk, batch_size, steps_total, lr, coeff_theta, coeff_alpha,
    monitor, checkpoint, u_clip, g_clip, lr_warmup, screen_fraction,
):
    params, first, second, updates = state
    count = x_valid.shape[0]
    mixture_weight = jnp.concatenate((
        jnp.full((batch_size,), coeff_alpha), jnp.full((batch_size,), 1.0 - coeff_alpha),
    ))

    def body(state, step):
        params, first, second, updates = state
        index_key, data_key, hat_key, mixture_key = jax.random.split(
            jax.random.fold_in(key, step), 4
        )
        flow_now = eqx.combine(params, static)
        x = x_valid[jax.random.choice(index_key, count, (batch_size,), replace=False)]
        y = target_data[
            jax.random.choice(data_key, target_data.shape[0], (batch_size,), replace=False)
        ]
        y_hat = mixture_data[
            jax.random.choice(hat_key, mixture_data.shape[0], (batch_size,), replace=False)
        ]
        y_bar, proposal_log_weight = _pushforward_weights(flow_now, x, source, target, domain)
        y_mix = resample(
            mixture_key, jnp.concatenate((y_hat, y_bar), axis=0), mixture_weight, N=batch_size,
        )
        energy = lax.stop_gradient(target(y))
        mixture_energy = lax.stop_gradient(target(y_mix))
        energy_keep = _energy_keep(energy, u_clip)
        mixture_keep = _energy_keep(mixture_energy, u_clip)

        def loss_fn(values):
            candidate = eqx.combine(values, static)
            latent, ladj = candidate.call_and_ladj(y)
            z = source(latent) - energy - ladj
            mixture_latent, mixture_ladj = candidate.call_and_ladj(y_mix)
            z_mix = source(mixture_latent) - mixture_energy - mixture_ladj
            valid = lax.stop_gradient(
                _screen_top(z, energy_keep & jnp.isfinite(z), screen_fraction)
            )
            mixture_valid = lax.stop_gradient(
                _screen_top(z_mix, mixture_keep & jnp.isfinite(z_mix), screen_fraction)
            )
            loss = _masked_mean(z, valid) + coeff_theta * _variation(z_mix, mixture_valid)
            return loss, (valid, mixture_valid)

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, (valid, mixture_valid)), grads = jax.value_and_grad(
            evaluated, has_aux=True
        )(params)
        params, first, second, updates = _adam(
            params, first, second, grads, loss, updates,
            _learning_rate(lr, lr_warmup, step), g_clip,
            jnp.any(valid) & jnp.any(mixture_valid),
        )
        ess = compute_ESS_log(proposal_log_weight, screen_fraction)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, 0.0, 1.0)
        return (params, first, second, updates), ess

    steps = step_offset + jnp.arange(1, chunk + 1)
    return lax.scan(body, (params, first, second, updates), steps)


def train_klxm_G(
    x_valid, source, target, flow, domain, target_data, batch_size, steps_total, lr,
    *, coeff_theta=1.0, coeff_alpha=0.5, mixture_data=None, monitor=None, seed=0,
    checkpoint=False, initialize_from_identity=False, u_clip=float("inf"),
    g_clip=float("inf"), lr_warmup=0, screen_fraction=SCREEN_FRACTION,
    reject_requested=None,
):
    """Train ``G`` with KL + coeff_theta X over the data/pushforward mixture; returns ``(flow, batch_ess)``."""
    if initialize_from_identity:
        flow = flow.zeros()
    target_data = jnp.asarray(target_data)
    mixture_data = target_data if mixture_data is None else jnp.asarray(mixture_data)
    key = jax.random.fold_in(jax.random.key(53), seed)
    return _run_chunks(
        lambda **kw: _chunk(
            jnp.asarray(x_valid), target_data, mixture_data, source, target,
            kw.pop("state"), kw.pop("static"), domain, key, **kw,
        ),
        flow, steps_total, reject_requested,
        batch_size=batch_size, lr=lr, coeff_theta=coeff_theta, coeff_alpha=coeff_alpha,
        monitor=monitor, checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip,
        lr_warmup=lr_warmup, screen_fraction=screen_fraction,
    )
