"""Generic model/trainer."""

import functools
import jax
import jax.numpy as jnp
import optax
import lib.utils.utils as utils
import lib.models.model_utils as model_utils
import lib.sampling.sampling_utils as sampling_utils

# Aufbau:
# forward Model => Q matrices => x_0 to noisy image x_t
# backward model => Loss calculation
# DiffusionModel => specify forward model, how to calc loss, how to sample
# CategoricalDiffusion => inherits from DiffusionModel => just specifiy BackwardsModel and Samples


class CategoricalDiffusionModel:
    """Model interface."""

    def __init__(self, config, fwd_model, backwd_model, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.backwd_model = backwd_model
        self.fwd_model = fwd_model

    def _build_loss_func(self, rng, x0):
        rng, loss_rng = jax.random.split(rng)
        # if len(x0.shape) == 4:
        #    B, H, W, C = x0.shape
        #    x0 = jnp.reshape(x0, (B, H * W * C))

        # sample xt => noise data
        bsize = x0.shape[0]  # B
        t_rng, sample_rng = jax.random.split(rng)
        t = jax.random.uniform(t_rng, (bsize,))
        t = t * self.config.time_duration
        qt = self.fwd_model.transition(t)
        b = jnp.expand_dims(jnp.arange(bsize), tuple(range(1, x0.ndim)))  #
        qt0 = qt[b, x0]
        logits = jnp.where(qt0 <= 0.0, -1e9, jnp.log(qt0))
        xt = jax.random.categorical(sample_rng, logits)

        loss_fn = functools.partial(
            self.backwd_model.loss, rng=loss_rng, x0=x0, xt=xt, t=t
        )
        return loss_fn

    def training_step(self, state, rng, batch):
        """Single gradient update step."""
        # batch: B, H, W, C or B, D
        if len(batch.shape) == 4:
            B, H, W, C = batch.shape
            batch = jnp.reshape(batch, (B, H * W * C))
        params, opt_state = state.params, state.opt_state
        loss_fn = self._build_loss_func(rng, batch)

        # calc grad and loss (has auxiliary output )
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )  # aux = {"loss": neg_elbo}
        print("loss", aux['loss'])
        # got only cpu
        # grads = jax.lax.pmean(grads, axis_name='shard')
        # aux = jax.lax.pmean(aux, axis_name='shard')

        # update optimizer and params of (weights)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # weight decay if step > 0
        ema_params = model_utils.apply_ema(
            decay=jnp.where(state.step == 0, 0.0, self.config.ema_decay),
            avg=state.ema_params,
            new=params,
        )
        # update values in struct state
        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
        )

        return new_state, aux

    # wrapper to give self as parameter
    def _sample_step(self, params, rng, tau, xt, t):
        return sampling_utils.get_sampler(self.config)(self, params, rng, tau, xt, t)

    # wrapper to give self as parameter
    def _sample_from_prior(self, rng, num_samples, conditioner=None):
        del conditioner

        if isinstance(self.config.discrete_dim, int):
            shape = (num_samples, self.config.discrete_dim)
        else:
            shape = tuple([num_samples] + list(self.config.discrete_dim))
        return self.fwd_model.sample_from_prior(
            rng, shape
        )  #  shape: B, discrete_dim kann sein: B, H*W*C oder B, H, W, C: muss hier B, D sein

    # wrapper to give self as parameter
    def _corrector_step(self, params, rng, tau, xt, t):
        return sampling_utils.lbjf_corrector_step(self, params, rng, tau, xt, t)

    def sample_loop(self, state, rng, num_samples, conditioner=None):
        """Sampling loop."""
        print("Sampling")
        rng, prior_rng = jax.random.split(rng)

        x_noisy = self._sample_from_prior(prior_rng, num_samples, conditioner)  # shape:
        ones = jnp.ones((num_samples,), dtype=jnp.float32)
        tau = 1.0 / self.config.sampling_steps

        def sample_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            new_y = self._sample_step(state.ema_params, local_rng, tau, xt, t)
            return new_y

        def sample_with_correct_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            xt = self._sample_step(state.ema_params, local_rng, tau, xt, t)
            scale = self.config.get("corrector_scale", 1.0)

            def corrector_body_fn(cstep, cxt):
                c_rng = jax.random.fold_in(local_rng, cstep)
                cxt = self._corrector_step(state.ema_params, c_rng, tau * scale, cxt, t)
                return cxt

            new_y = jax.lax.fori_loop(
                0, self.config.get("corrector_steps", 0), corrector_body_fn, xt
            )
            return new_y

        cf = self.config.get("corrector_frac", 0.0)
        corrector_steps = int(cf * self.config.sampling_steps)
        x0 = jax.lax.fori_loop(
            0, self.config.sampling_steps - corrector_steps, sample_body_fn, x_noisy
        )
        if corrector_steps > 0:
            x0 = jax.lax.fori_loop(
                self.config.sampling_steps - corrector_steps,
                self.config.sampling_steps,
                sample_with_correct_body_fn,
                x0,
            )
        return x0

    # wrapper to give self as parameter
