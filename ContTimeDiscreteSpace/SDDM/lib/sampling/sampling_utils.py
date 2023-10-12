import functools
import jax
import jax.numpy as jnp
import optax
import lib.utils.utils as utils
import time

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
    print("fwd_rate", type(fwd_rate), fwd_rate.shape)
    print("torch.exp(log_weight)", type(jnp.exp(log_weight)), jnp.exp(log_weight).shape)
    print("h", type(tau), tau)
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
    start = time.time()
    if xt_target is None:
        xt_target = xt
    ll_all, ll_xt = cls.backwd_model.get_logprob(params, xt, t, xt_target)
    log_weight = ll_all - jnp.expand_dims(ll_xt, axis=-1)
    fwd_rate = cls.fwd_model.rate(xt, t)
    #print("log prob dim: ll_all, ll_xt", ll_all.shape)
    xt_onehot = jax.nn.one_hot(xt_target, cls.config.vocab_size)
    posterior = tau * jnp.exp(log_weight) * fwd_rate
    posterior = posterior * (1 - xt_onehot)

    flips = jax.random.poisson(rng, lam=posterior)
    choices = jnp.expand_dims(
        jnp.arange(cls.config.vocab_size, dtype=jnp.int32), axis=list(range(xt.ndim))
    )
    if not cls.config.get("is_ordinal", True):
        tot_flips = jnp.sum(flips, axis=-1, keepdims=True)
        flip_mask = (tot_flips <= 1).astype(jnp.int32)
        flips = flips * flip_mask
    diff = choices - jnp.expand_dims(xt, axis=-1)
    avg_offset = jnp.sum(flips * diff, axis=-1)
    new_y = xt + avg_offset
    new_y = jnp.clip(new_y, a_min=0, a_max=cls.config.vocab_size - 1)
    end = time.time()
    print("sample time", end - start)
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


def get_sampler(config):
    """Get generic categorical samplers."""
    if config.get("sampler_type", "lbjf") == "lbjf":
        fn_sampler = lbjf_sample_step
    elif config.sampler_type == "tau_leaping":
        fn_sampler = tau_leaping_step
    elif config.sampler_type == "exact":
        fn_sampler = exact_sampling
    else:
        raise ValueError("Unknown sampler type %s" % config.sampler_type)
    return fn_sampler
