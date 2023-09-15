from typing import Any
from flax import linen as nn
import jax
import jax.numpy as jnp
import lib.models.model_utils as model_utils
import forward_model
import lib.utils.utils as utils

def get_logprob_with_logits(cls, xt, t, logits, xt_target=None):
    """Get lobprob with logits."""

    if xt_target is None:
        xt_target = xt
        xt_onehot = jax.nn.one_hot(xt_target, cls.config.vocab_size)
    if cls.config.get('logit_type', 'direct') == 'direct':
        log_prob = nn.log_softmax(logits, axis=-1)
    else:
        qt0 = cls.fwd_model.transition(t)
    if cls.config.logit_type == 'reverse_prob':
        p0t = jax.nn.softmax(logits, axis=-1)
        qt0 = jnp.expand_dims(qt0, axis=list(range(1, xt.ndim - 1)))
        prob_all = p0t @ qt0
        log_prob = jnp.log(prob_all + 1e-35)
    elif cls.config.logit_type == 'reverse_logscale':
        log_p0t = nn.log_softmax(logits, axis=-1)
        log_qt0 = jnp.where(qt0 <= 1e-35, -1e9, jnp.log(qt0))
        log_qt0 = jnp.expand_dims(log_qt0, axis=list(range(1, xt.ndim)))
        log_p0t = jnp.expand_dims(log_p0t, axis=-1)
        log_prob = jax.nn.logsumexp(log_p0t + log_qt0, axis=-2)
    else:
        raise ValueError('Unknown logit_type: %s' % cls.config.logit_type)
    log_xt = jnp.sum(log_prob * xt_onehot, axis=-1)
    return log_prob, log_xt


class CondFactorizedBackwardModel:
    """Conditional factorized backward model."""

    def __init__(self, config, net):
        self.config = config
        self.fwd_model = forward_model.get_fwd_model(self.config)
        self.net = net

    def get_logits(self, params, xt, t):
        return self.net.apply({'params': params}, x=xt, t=t)

    def get_logprob(self, params, xt, t, xt_target=None):
        """Get backwd ratio."""

        logits = self.get_logits(params, xt, t)
        return get_logprob_with_logits(self, xt, t, logits, xt_target)

    def get_ratio(self, params, xt, t, xt_target=None):
        log_prob, log_xt = self.get_logprob(params, xt, t, xt_target)
        log_xtneg = jnp.sum(log_prob, axis=-1) - log_xt
        return jnp.exp(log_xtneg - log_xt)

    def calc_loss(self, xt, t, ll_all, ll_xt):
        """Calc loss.

        Args:
            xt: bsize x dim(s)
            t: bsize
            ll_all: bsize x dim(s) x vocab_size
            ll_xt: bsize x dim(s)
        Returns:
            loss: bsize x dim(s)
        """
        if self.config.loss_type == 'rm':
            loss = -ll_xt
        elif self.config.loss_type == 'mle':
            loss = -((self.config.vocab_size - 1) * ll_xt +
                    jnp.sum(utils.log1mexp(ll_all), axis=-1) -
                    utils.log1mexp(ll_xt))
        elif self.config.loss_type == 'elbo':
            xt_onehot = jax.nn.one_hot(xt, self.config.vocab_size)
            b = jnp.expand_dims(jnp.arange(xt.shape[0]), tuple(range(1, xt.ndim)))
            qt0_x2y = self.fwd_model.transition(t)
            qt0_y2x = jnp.transpose(qt0_x2y, (0, 2, 1))
            qt0_y2x = qt0_y2x[b, xt]
            ll_xt = jnp.expand_dims(ll_xt, axis=-1)
            backwd = jnp.exp(ll_all - ll_xt) * qt0_y2x
            first_term = jnp.sum(backwd * (1 - xt_onehot), axis=-1)

            qt0_x2y = qt0_x2y[b, xt]
            fwd = (ll_xt - ll_all) * qt0_x2y
            second_term = jnp.sum(fwd * (1 - xt_onehot), axis=-1)
            loss = first_term - second_term
        else:
            raise ValueError('Unknown loss_type: %s' % self.config.loss_type)
        weight = model_utils.get_lambda_t(t)
        weight = jnp.expand_dims(weight, axis=list(range(1, loss.ndim)))
        loss = loss * weight
        return loss

    def loss(self, params, rng, x0, xt, t):
        """Calc loss."""
        del rng
        logits = self.get_logits(params, xt, t)
        loss = 0.0
        ce_coeff = self.config.get('ce_coeff', 0.0)
        if self.config.loss_type == 'x0ce':
            ce_coeff = 1.0
        if ce_coeff > 0:
            x0_onehot = jax.nn.one_hot(x0, self.config.vocab_size)
            ll = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.sum(ll * x0_onehot, axis=-1) * ce_coeff
        if ce_coeff < 1:
            ll_all, log_xt = get_logprob_with_logits(self, xt, t, logits)
            loss = loss + self.calc_loss(xt, t, ll_all, log_xt) * (1 - ce_coeff)
        loss = jnp.sum(loss) / xt.shape[0]
        aux = {'loss': loss}
        return loss, aux
