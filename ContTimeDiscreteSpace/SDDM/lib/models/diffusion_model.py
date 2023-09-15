



"""Generic model/trainer."""

import functools
import jax
import jax.numpy as jnp
import optax
import lib.utils.utils as utils
from sddm.model import backward_model
from sddm.model import ebm
from lib.models import forward_model
from lib.models import hollow_model
from lib.models import tauldr_model

# Aufbau:
# forward Model => Q matrices => x_0 to noisy image x_t
# backward model => Loss calculation
# DiffusionModel => specify forward model, how to calc loss, how to sample
# CategoricalDiffusion => inherits from DiffusionModel => just specifiy BackwardsModel and Samples


def lbjf_corrector_step(cls, params, rng, tau, xt, t, xt_target=None):
    """Categorical simulation with lbjf."""
    if xt_target is None:
        xt_target = xt
    ll_all, ll_xt = cls.backwd_model.get_logprob(params, xt, t, xt_target)
    log_weight = ll_all - jnp.expand_dims(ll_xt, axis=-1)
    fwd_rate = cls.fwd_model.rate(xt, t)

    xt_onehot = jax.nn.one_hot(xt_target, cls.config.vocab_size)
    posterior = tau * (jnp.exp(log_weight) * fwd_rate + fwd_rate)
    off_diag = jnp.sum(posterior * (1 - xt_onehot), axis=-1, keepdims=True)
    diag = jnp.clip(1.0 - off_diag, a_min=0)
    posterior = posterior * (1 - xt_onehot) + diag * xt_onehot
    posterior = posterior / jnp.sum(posterior, axis=-1, keepdims=True)
    log_posterior = jnp.log(posterior + 1e-35)
    new_y = jax.random.categorical(rng, log_posterior, axis=-1)
    return new_y

# Interface
class CategoricalDiffusionModel:
    """Model interface."""

    def __init__(self, config, fwd_model, backwd_model, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.backwd_model = backwd_model
        self.fwd_model = fwd_model

    def _build_loss_func(self, rng, x0):
        rng, loss_rng = jax.random.split(rng)

        # sample xt => noise data
        bsize = x0.shape[0]
        t_rng, sample_rng = jax.random.split(rng)
        t = jax.random.uniform(t_rng, (bsize,))
        t = t * self.config.time_duration
        qt = self.transition(t)
        b = jnp.expand_dims(jnp.arange(bsize), tuple(range(1, x0.ndim)))
        qt0 = qt[b, x0]
        logits = jnp.where(qt0 <= 0.0, -1e9, jnp.log(qt0))
        xt = jax.random.categorical(sample_rng, logits)

        loss_fn = functools.partial(self.backwd_model.loss, rng=loss_rng,
                                    x0=x0, xt=xt, t=t)
        return loss_fn

    def training_step(self, state, rng, batch):
        """Single gradient update step."""
        params, opt_state = state.params, state.opt_state
        loss_fn = self._build_loss_func(rng, batch)
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        grads = jax.lax.pmean(grads, axis_name='shard')
        aux = jax.lax.pmean(aux, axis_name='shard')
        
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = utils.apply_ema(
            decay=jnp.where(state.step == 0, 0.0, self.config.ema_decay),
            avg=state.ema_params,
            new=params,
        )
        new_state = state.replace(
            step=state.step + 1, params=params, opt_state=opt_state,
            ema_params=ema_params)
        return new_state, aux

    def sample_step(self, params, rng, tau, xt, t):
        raise NotImplementedError

    def corrector_step(self, params, rng, tau, xt, t):
        return lbjf_corrector_step(self, params, rng, tau, xt, t)

    def sample_from_prior(self, rng, num_samples, conditioner=None):
        del conditioner
        if isinstance(self.config.discrete_dim, int):
            shape = (num_samples, self.config.discrete_dim)
        else:
            shape = tuple([num_samples] + list(self.config.discrete_dim))
        return self.fwd_model.sample_from_prior(rng, shape)

    def sample_loop(self, state, rng, num_samples=None, conditioner=None):
        """Sampling loop."""
        rng, prior_rng = jax.random.split(rng)
        if num_samples is None:
            num_samples = self.config.plot_samples // jax.device_count()
        x_start = self.sample_from_prior(
            prior_rng, num_samples, conditioner)
        ones = jnp.ones((num_samples,), dtype=jnp.float32)
        tau = 1.0 / self.config.sampling_steps

        def sample_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            new_y = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            return new_y

        def sample_with_correct_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = jax.random.fold_in(rng, step)
            xt = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            scale = self.config.get('corrector_scale', 1.0)
            def corrector_body_fn(cstep, cxt):
                c_rng = jax.random.fold_in(local_rng, cstep)
                cxt = self.corrector_step(state.ema_params, c_rng, tau * scale, cxt, t)
                return cxt

            new_y = jax.lax.fori_loop(0, self.config.get('corrector_steps', 0),
                                    corrector_body_fn, xt)
            return new_y

        cf = self.config.get('corrector_frac', 0.0)
        corrector_steps = int(cf * self.config.sampling_steps)
        x0 = jax.lax.fori_loop(0, self.config.sampling_steps - corrector_steps,
                                sample_body_fn, x_start)
        if corrector_steps > 0:
            x0 = jax.lax.fori_loop(self.config.sampling_steps - corrector_steps,
                                    self.config.sampling_steps,
                                    sample_with_correct_body_fn, x0)
        return x0


def binary_sample_step(cls, params, rng, tau, xt, t, xt_target=None):
    if xt_target is None:
        xt_target = xt
    ratio = cls.backwd_model.get_ratio(params, xt, t, xt_target)
    cur_rate = cls.fwd_model.rate_const * ratio
    nu_x = jax.nn.sigmoid(cur_rate)
    flip_rate = nu_x * jnp.exp(utils.log1mexp(-tau * cur_rate / nu_x))
    flip = jax.random.bernoulli(rng, flip_rate)
    new_y = (1 - xt_target) * flip + xt_target * (1 - flip)
    return new_y


def lbjf_sample_step(cls, params, rng, tau, xt, t, xt_target=None):
    """Categorical simulation with lbjf."""
    if xt_target is None:
        xt_target = xt
    ll_all, ll_xt = cls.backwd_model.get_logprob(params, xt, t, xt_target)
    log_weight = ll_all - jnp.expand_dims(ll_xt, axis=-1)
    fwd_rate = cls.fwd_model.rate(xt, t)

    xt_onehot = jax.nn.one_hot(xt_target, cls.config.vocab_size)
    posterior = tau * jnp.exp(log_weight) * fwd_rate
    off_diag = jnp.sum(posterior * (1 - xt_onehot), axis=-1, keepdims=True)
    diag = jnp.clip(1.0 - off_diag, a_min=0)
    posterior = posterior * (1 - xt_onehot) + diag * xt_onehot
    posterior = posterior / jnp.sum(posterior, axis=-1, keepdims=True)
    log_posterior = jnp.log(posterior + 1e-35)
    new_y = jax.random.categorical(rng, log_posterior, axis=-1)
    return new_y


def tau_leaping_step(cls, params, rng, tau, xt, t, xt_target=None):
    """Categorical simulation with tau leaping."""
    if xt_target is None:
        xt_target = xt
    ll_all, ll_xt = cls.backwd_model.get_logprob(params, xt, t, xt_target)
    log_weight = ll_all - jnp.expand_dims(ll_xt, axis=-1)
    fwd_rate = cls.fwd_model.rate(xt, t)

    xt_onehot = jax.nn.one_hot(xt_target, cls.config.vocab_size)
    posterior = tau * jnp.exp(log_weight) * fwd_rate
    posterior = posterior * (1 - xt_onehot)

    flips = jax.random.poisson(rng, lam=posterior)
    choices = jnp.expand_dims(jnp.arange(cls.config.vocab_size, dtype=jnp.int32),
                            axis=list(range(xt.ndim)))
    if not cls.config.get('is_ordinal', True):
        tot_flips = jnp.sum(flips, axis=-1, keepdims=True)
        flip_mask = (tot_flips <= 1).astype(jnp.int32)
        flips = flips * flip_mask
    diff = choices - jnp.expand_dims(xt, axis=-1)
    avg_offset = jnp.sum(flips * diff, axis=-1)
    new_y = xt + avg_offset
    new_y = jnp.clip(new_y, a_min=0, a_max=cls.config.vocab_size - 1)
    return new_y


def exact_sampling(cls, params, rng, tau, xt, t, xt_target=None):
    """Exact categorical simulation."""
    del xt_target
    # in HollowModel: get_logits = model.forward() = model()
    logits = cls.backwd_model.get_logits(params, xt, t)
    log_p0t = jax.nn.log_softmax(logits, axis=-1)
    t_eps = t - tau
    q_teps_0 = cls.fwd_model.transition(t_eps)
    q_teps_0 = jnp.expand_dims(q_teps_0, axis=list(range(1, xt.ndim)))
    q_t_teps = cls.fwd_model.transit_between(t_eps, t)
    q_t_teps = jnp.transpose(q_t_teps, (0, 2, 1))

    b = jnp.expand_dims(jnp.arange(xt.shape[0]), tuple(range(1, xt.ndim)))
    q_t_teps = jnp.expand_dims(q_t_teps[b, xt], axis=-2)
    qt0 = q_teps_0 * q_t_teps
    log_qt0 = jnp.where(qt0 <= 0.0, -1e9, jnp.log(qt0))
    log_p0t = jnp.expand_dims(log_p0t, axis=-1)

    log_prob = jax.nn.logsumexp(log_p0t + log_qt0, axis=-2)
    new_y = jax.random.categorical(rng, log_prob, axis=-1)
    return new_y


def get_sampler(config):
    """Get generic categorical samplers."""
    if config.get('sampler_type', 'lbjf') == 'lbjf':
        fn_sampler = lbjf_sample_step
    elif config.sampler_type == 'tau_leaping':
        fn_sampler = tau_leaping_step
    elif config.sampler_type == 'exact':
        fn_sampler = exact_sampling
    else:
        raise ValueError('Unknown sampler type %s' % config.sampler_type)
    return fn_sampler



    def sample_step(self, params, rng, tau, xt, t):
        return get_sampler(self.config)(self, params, rng, tau, xt, t)