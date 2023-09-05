import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sddm.common import torch_utils
from sddm.model import torch_forward_model


class BackwardModel(object):
    """Backward model."""

    def __init__(self, config):
        self.config = config
        self.net = None

    def make_init_params(self, global_rng):
        if isinstance(self.config.discrete_dim, int):
            input_shape = (1, self.config.discrete_dim)
        else:
            input_shape = [1] + list(self.config.discrete_dim)
        init_kwargs = dict(
            x=torch.zeros(input_shape, dtype=torch.int32),
            t=torch.zeros((1,), dtype=torch.float32),
        )
        # wahrscheinlich falsch
        return self.net(**init_kwargs).state_dict()

    def get_lambda_t(self, t):
        """Get lambda schedule."""
        if self.config.get("lambda_t", "const") == "const":
            return torch.ones(t.shape, dtype=torch.float32)
        elif self.config.lambda_t == "grow_linear":
            return 0.5 + t
        elif self.config.lambda_t == "decay_linear":
            return 1.5 - t
        elif self.config.lambda_t == "decay_convex":
            return (0.1 + t) ** -0.5
        else:
            raise ValueError("Unknown lambda_t: %s" % self.config.lambda_t)


def get_logprob_with_logits(cls, xt, t, logits, xt_target=None):
    """Get logprob with logits."""

    if xt_target is None:
        xt_target = xt
    xt_onehot = F.one_hot(xt_target, cls.config.vocab_size)
    if cls.config.get("logit_type", "direct") == "direct":
        log_prob = F.log_softmax(logits, dim=-1)
    else:
        qt0 = cls.fwd_model.transition(t)
        if cls.config.logit_type == "reverse_prob":
            p0t = F.softmax(logits, dim=-1)
            #

            qt0 = torch_utils.expand_dims(qt0, axis=list(range(1, xt.dim() - 1)))
            prob_all = p0t @ qt0
            log_prob = torch.log(prob_all + 1e-35)
        elif cls.config.logit_type == "reverse_logscale":
            log_p0t = F.log_softmax(logits, dim=-1)
            log_qt0 = torch.where(qt0 <= 1e-35, -1e9, torch.log(qt0))
            log_qt0 = torch_utils.expand_dims(log_qt0, axis=list(range(1, xt.dim())))
            log_p0t = log_p0t.unsqueeze(-1)

            log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2)
        else:
            raise ValueError("Unknown logit_type: %s" % cls.config.logit_type)
    log_xt = torch.sum(log_prob * xt_onehot, dim=-1)
    return log_prob, log_xt


class CondFactorizedBackwardModel(BackwardModel):
    """Conditional factorized backward model."""

    def __init__(self, config):
        super(CondFactorizedBackwardModel, self).__init__(config)
        self.fwd_model = torch_forward_model.get_fwd_model(self.config)

    def get_logits(self, params, xt, t):
        return self.net(x=xt, t=t)

    def get_logprob(self, params, xt, t, xt_target=None):
        """Get backwd ratio."""

        logits = self.get_logits(params, xt, t)
        return get_logprob_with_logits(self, xt, t, logits, xt_target)

    def get_ratio(self, params, xt, t, xt_target=None):
        log_prob, log_xt = self.get_logprob(params, xt, t, xt_target)
        log_xtneg = torch.sum(log_prob, dim=-1) - log_xt
        return torch.exp(log_xtneg - log_xt)

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
        if self.config.loss_type == "rm":
            loss = -ll_xt
        elif self.config.loss_type == "mle":
            loss = -(
                (self.config.vocab_size - 1) * ll_xt
                + torch.sum(torch_utils.log1mexp(ll_all), dim=-1)
                - torch_utils.log1mexp(ll_xt)
            )
        elif self.config.loss_type == "elbo":
            xt_onehot = F.one_hot(xt, num_classes=self.config.vocab_size)
            b = torch_utils.expand_dims(
                torch.arange(xt.shape[0]), tuple(range(1, xt.dim()))
            )
            qt0_x2y = self.fwd_model.transition(t)
            qt0_y2x = qt0_x2y.permute(0, 2, 1)
            qt0_y2x = qt0_y2x[b, xt]
            ll_xt = ll_xt.unsqueeze(-1)
            backwd = torch.exp(ll_all - ll_xt) * qt0_y2x
            first_term = torch.sum(backwd * (1 - xt_onehot), dim=-1)

            qt0_x2y = qt0_x2y[b, xt]
            fwd = (ll_xt - ll_all) * qt0_x2y
            second_term = torch.sum(fwd * (1 - xt_onehot), dim=-1)
            loss = first_term - second_term
        else:
            raise ValueError("Unknown loss_type: %s" % self.config.loss_type)
        weight = self.get_lambda_t(t)
        weight = torch_utils.expand_dims(weight, axis=list(range(1, loss.dim())))
        loss = loss * weight
        return loss

    def loss(self, params, rng, x0, xt, t):
        """Calc loss."""
        del rng
        logits = self.get_logits(params, xt, t)
        loss = 0.0
        ce_coeff = self.config.get("ce_coeff", 0.0)
        if self.config.loss_type == "x0ce":
            ce_coeff = 1.0
        if ce_coeff > 0:
            x0_onehot = F.one_hot(x0, self.config.vocab_size)
            ll = F.log_softmax(logits, dim=-1)
            loss = -torch.sum(ll * x0_onehot, dim=-1) * ce_coeff
        if ce_coeff < 1:
            ll_all, log_xt = get_logprob_with_logits(self, xt, t, logits)
            loss = loss + self.calc_loss(xt, t, ll_all, log_xt) * (1 - ce_coeff)
        loss = torch.sum(loss) / xt.shape[0]
        aux = {"loss": loss}
        return loss, aux
