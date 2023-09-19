import functools
import jax
import jax.numpy as jnp
import optax
import lib.utils.utils as utils
import lib.sampling.sampling_utils as sampling_utils


class Sampler:
    def __init__(self, config, sample_step, num_samples, conditioner=None):
        self.config = config
        self.sample_step = sample_step
        self.num_samples = num_samples
        self.conditioner = conditioner

    def _corrector_step(self, params, rng, tau, xt, t):
        return sampling_utils.lbjf_corrector_step(self, params, rng, tau, xt, t)

    def _sample_from_prior(self, rng, num_samples, conditioner=None):
        del conditioner
        if isinstance(self.config.discrete_dim, int):
            shape = (num_samples, self.config.discrete_dim)
        else:
            shape = tuple([num_samples] + list(self.config.discrete_dim))
        return self.fwd_model.sample_from_prior(rng, shape)

    def sample_loop(self, rng):
        """Sampling loop."""
        rng, prior_rng = jax.random.split(rng)

        x_start = self._sample_from_prior(prior_rng, self.num_samples, self.conditioner)
        print("x_start from prior", x_start.shape)
        ones = jnp.ones((self.num_samples,), dtype=jnp.float32)
        tau = 1.0 / self.config.sampling_steps

        def sample_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            new_y = self.sample_step(self.state.ema_params, local_rng, tau, xt, t)
            return new_y

        def sample_with_correct_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            xt = self.sample_step(self.state.ema_params, local_rng, tau, xt, t)
            scale = self.config.get("corrector_scale", 1.0)

            def corrector_body_fn(cstep, cxt):
                c_rng = jax.random.fold_in(local_rng, cstep)
                cxt = self._corrector_step(
                    self.state.ema_params, c_rng, tau * scale, cxt, t
                )
                return cxt

            new_y = jax.lax.fori_loop(
                0, self.config.get("corrector_steps", 0), corrector_body_fn, xt
            )
            return new_y

        cf = self.config.get("corrector_frac", 0.0)
        corrector_steps = int(cf * self.config.sampling_steps)
        x0 = jax.lax.fori_loop(
            0, self.config.sampling_steps - corrector_steps, sample_body_fn, x_start
        )
        if corrector_steps > 0:
            x0 = jax.lax.fori_loop(
                self.config.sampling_steps - corrector_steps,
                self.config.sampling_steps,
                sample_with_correct_body_fn,
                x0,
            )
        return x0



