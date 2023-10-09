"""Generic model/trainer."""

import functools
import jax
import jax.numpy as jnp
import optax
import lib.utils.utils as utils
from lib.sampling import sampling_utils
from lib.models import ebm, hollow_model, tauldr_model
from lib.optimizer import optimizer as optim
from lib.models import forward_model

# Aufbau:
# forward Model => Q matrices => x_0 to noisy image x_t
# backward model => Loss calculation
# DiffusionModel => specify forward model, how to calc loss, how to sample
# CategoricalDiffusion => inherits from DiffusionModel => just specifiy BackwardsModel and Samples


class CategoricalDiffusionModel:
    """Model interface."""

    def build_backwd_model(self, config):
        if config.model_type == "ebm":
            backwd_model = ebm.CategoricalScoreModel(config)
        elif config.model_type == "hollow":
            backwd_model = hollow_model.HollowModel(config)
        elif config.model_type == "tauldr":
            backwd_model = tauldr_model.TauLDRBackward(config)
        else:
            raise ValueError("Unknown model type %s" % config.model_type)
        return backwd_model

    def __init__(self, config):
        self.config = config
        self.optimizer = optim.build_optimizer(config)
        self.fwd_model = forward_model.build_fwd_model(config)
        self.backwd_model = self.build_backwd_model(config)

    def init_state(self, model_key):
        state = utils.init_host_state(
            self.backwd_model.make_init_params(model_key), self.optimizer
        )
        return state

    def _build_loss_func(self, rng, x0):
        rng, loss_rng = jax.random.split(rng)
        if len(x0.shape) == 4:
            B, H, W, C = x0.shape
            x0 = jnp.reshape(x0, [B, -1]) 

        # sample xt => noise data
        
        xt, t = self.fwd_model.sample_xt(x0, self.config.time_duration, rng) # B, D oder B, H, W, C
        loss_fn = functools.partial(
            self.backwd_model.loss, rng=loss_rng, x0=x0, xt=xt, t=t
        )
        return loss_fn

    def training_step(self, state, rng, batch):
        """Single gradient update step."""
        # batch: B, H, W, C or B, D
        #if len(batch.shape) == 4:
        #    B, H, W, C = batch.shape
        #    batch = jnp.reshape(batch, (B, H * W * C))

        params, opt_state = state.params, state.opt_state
        loss_fn = self._build_loss_func(rng, batch)
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        if jax.process_count() > 1:  # or check some other condition indicating parallel execution
            grads = jax.lax.pmean(grads, axis_name="shard")
            aux = jax.lax.pmean(aux, axis_name="shard")

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = utils.apply_ema(
            decay=jnp.where(state.step == 0, 0.0, self.config.ema_decay),
            avg=state.ema_params,
            new=params,
        )
        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
        )
        return new_state, aux

    # wrapper to give self as parameter
    def sample_step(self, params, rng, tau, xt, t):
        return sampling_utils.get_sampler(self.config)(self, params, rng, tau, xt, t)

    def corrector_step(self, params, rng, tau, xt, t):
        return sampling_utils.lbjf_corrector_step(self, params, rng, tau, xt, t)

    def sample_from_prior(self, rng, num_samples, conditioner=None):
        del conditioner
        if isinstance(self.config.discrete_dim, int):
            shape = (num_samples, self.config.discrete_dim) # B, D
        else:
            shape = tuple([num_samples] + list(self.config.discrete_dim)) # B, H, W, C
        return self.fwd_model.sample_from_prior(rng, shape)

    def sample_loop(self, state, rng, num_samples=None, conditioner=None):
        """Sampling loop."""
        rng, prior_rng = jax.random.split(rng)
        if num_samples is None:
            num_samples = self.config.plot_samples // jax.device_count()
        x_start = self.sample_from_prior(prior_rng, num_samples, conditioner) # B, D oder B, H, W, C
        #print("sampled from prior", x_start.shape)
        ones = jnp.ones((num_samples,), dtype=jnp.float32)
        tau = 1.0 / self.config.sampling_steps

        def sample_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            new_y = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            #print("sample step")
            #new_y = self.sample_step(state.params, local_rng, tau, xt, t)
            return new_y

        def sample_with_correct_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            xt = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            #xt = self.sample_step(state.params, local_rng, tau, xt, t)
            scale = self.config.get("corrector_scale", 1.0)

            def corrector_body_fn(cstep, cxt):
                c_rng = jax.random.fold_in(local_rng, cstep)
                #cxt = self.corrector_step(state.ema_params, c_rng, tau * scale, cxt, t)
                cxt = self.corrector_step(state.params, c_rng, tau * scale, cxt, t)
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
        # wrapper to give self as parameter
