import functools
import torch
import torch.nn.functional as F
import numpy as np
from sddm.common import torch_utils
from sddm.model import torch_backward_model
from sddm.model import torch_ebm
from sddm.model import torch_forward_model
from sddm.model import torch_hollow_model
from sddm.model import torch_tauldr_model


# DiffusionModel got
def lbjf_corrector_step(cls, params, rng, tau, xt, t, xt_target=None):
    """Categorical simulation with lbjf."""
    if xt_target is None:
        xt_target = xt
    ll_all, ll_xt = cls.backwd_model.get_logprob(params, xt, t, xt_target)
    log_weight = ll_all - ll_xt.unsqueeze(-1)
    fwd_rate = cls.fwd_model.rate(xt, t)

    xt_onehot = F.one_hot(xt_target, cls.config.vocab_size)
    posterior = tau * (torch.exp(log_weight) * fwd_rate + fwd_rate)
    off_diag = torch.sum(posterior * (1 - xt_onehot), dim=-1, keepdim=True)
    diag = torch.clip(1.0 - off_diag, min=0)
    posterior = posterior * (1 - xt_onehot) + diag * xt_onehot
    posterior = posterior / torch.sum(posterior, dim=-1, keepdim=True)
    log_posterior = torch.log(posterior + 1e-35)
    # fix?
    cat_dist = torch.distributions.categorical.Categorical(logits=log_posterior)
    new_y = cat_dist.sample()
    # new_y = torch.multinomial(log_posterior, num_samples=1, replacement=True)
    return new_y  # .squeeze()


class DiffusionModel(object):
    """Model interface."""

    def build_backwd_model(self, config):
        raise NotImplementedError

    def __init__(self, config):
        self.config = config
        self.optimizer = torch_utils.build_optimizer(config)
        self.backwd_model = None
        self.fwd_model = None
        self.backwd_model = self.build_backwd_model(config)
        self.fwd_model = torch_forward_model.get_fwd_model(config)

    # geht das überhaupt mit
    def init_state(self, model_key):
        state = torch_utils.init_host_state(
            self.backwd_model.make_init_params(model_key), self.optimizer
        )
        return state

    def init_state(self, model_key):
        init_params = self.backwd_model.make_init_params(model_key)
        state = torch_utils.init_host_state(init_params, self.optimizer)
        return state

    def build_loss_func(self, rng, x0):
        rng, loss_rng = torch.manual_seed(rng), torch.manual_seed(rng)
        xt, t = self.fwd_model.sample_xt(x0, self.config.time_duration, rng)
        loss_fn = functools.partial(
            self.backwd_model.loss, rng=loss_rng, x0=x0, xt=xt, t=t
        )
        return loss_fn
    
    # same to training step:
    def step_fn(self, state, rng, batch):
        """Single gradient update step."""
        params, opt_state = state.params, state.opt_state
        loss_fn = self.build_loss_func(rng, batch)
        loss, aux = loss_fn(params)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_params = [
            p - lr * p.grad
            for p, lr in zip(params, self.optimizer.param_groups[0]["lr"])
        ]

        ema_params = torch_utils.apply_ema(
            decay=torch.where(state.step == 0, 0.0, self.config.ema_decay),
            avg=state.ema_params,
            new=params,
        )

        new_state = state.replace(
            step=state.step + 1, params=new_params, ema_params=ema_params
        )

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
        rng, prior_rng = torch.manual_seed(rng), torch.manual_seed(rng)
        if num_samples is None:
            num_samples = self.config.plot_samples // (torch.cuda.device_count() or 1)
        x_start = self.sample_from_prior(prior_rng, num_samples, conditioner)
        ones = torch.ones((num_samples,), dtype=torch.float32)
        tau = 1.0 / self.config.sampling_steps

        def sample_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = torch.manual_seed(rng)
            new_y = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            return new_y

        def sample_with_correct_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = torch.manual_seed(rng)
            xt = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            scale = self.config.get("corrector_scale", 1.0)

            for cstep in range(self.config.get("corrector_steps", 0)):
                c_rng = torch.manual_seed(local_rng)
                xt = self.corrector_step(state.ema_params, c_rng, tau * scale, xt, t)

            return xt

        cf = self.config.get("corrector_frac", 0.0)
        corrector_steps = int(cf * self.config.sampling_steps)
        x0 = x_start
        for step in range(0, self.config.sampling_steps - corrector_steps):
            x0 = sample_body_fn(step, x0)

        if corrector_steps > 0:
            for step in range(
                self.config.sampling_steps - corrector_steps, self.config.sampling_steps
            ):
                x0 = sample_with_correct_body_fn(step, x0)

        return x0


def binary_sample_step(cls, params, rng, tau, xt, t, xt_target=None):
    if xt_target is None:
        xt_target = xt
    ratio = cls.backwd_model.get_ratio(params, xt, t, xt_target)
    cur_rate = cls.fwd_model.rate_const * ratio
    nu_x = torch.sigmoid(cur_rate)
    flip_rate = nu_x * torch.exp(-tau * cur_rate / nu_x)
    flip = torch.bernoulli(flip_rate)
    new_y = (1 - xt_target) * flip + xt_target * (1 - flip)
    return new_y


def lbjf_sample_step(cls, params, rng, tau, xt, t, xt_target=None):
    """Categorical simulation with lbjf."""
    if xt_target is None:
        xt_target = xt
    ll_all, ll_xt = cls.backwd_model.get_logprob(params, xt, t, xt_target)
    log_weight = ll_all - ll_xt.unsqueeze(-1)
    fwd_rate = cls.fwd_model.rate(xt, t)

    xt_onehot = F.one_hot(xt_target, cls.config.vocab_size)
    posterior = tau * torch.exp(log_weight) * fwd_rate
    off_diag = torch.sum(posterior * (1 - xt_onehot), dim=-1, keepdim=True)
    diag = torch.clamp(1.0 - off_diag, min=0)
    posterior = posterior * (1 - xt_onehot) + diag * xt_onehot
    posterior = posterior / torch.sum(posterior, dim=-1, keepdim=True)
    log_posterior = torch.log(posterior + 1e-35)
    cat_dist = torch.distributions.categorical.Categorical(logits=log_posterior)
    new_y = cat_dist.sample()
    # new_y = torch.multinomial(torch.exp(log_posterior), num_samples=1)
    return new_y  # .squeeze()


def tau_leaping_step(cls, params, rng, tau, xt, t, xt_target=None):
    """Categorical simulation with tau leaping."""
    if xt_target is None:
        xt_target = xt
    ll_all, ll_xt = cls.backwd_model.get_logprob(params, xt, t, xt_target)
    log_weight = ll_all - ll_xt.unsqueeze(-1)
    fwd_rate = cls.fwd_model.rate(xt, t)

    xt_onehot = F.one_hot(xt_target, cls.config.vocab_size)
    posterior = tau * torch.exp(log_weight) * fwd_rate
    posterior = posterior * (1 - xt_onehot)

    flips = torch.distributions.poisson.Poisson(posterior).sample()
    choices = torch_utils.expand_dims(
        torch.arange(cls.config.vocab_size, dtype=torch.int32), list(range(xt.dim()))
    )

    if not cls.config.get("is_ordinal", True):
        tot_flips = torch.sum(flips, dim=-1, keepdim=True)
        flip_mask = (tot_flips <= 1).to(torch.int32)
        flips = flips * flip_mask
    diff = choices - xt.unsqueeze(-1)
    avg_offset = torch.sum(flips * diff, dim=-1)
    new_y = xt + avg_offset
    new_y = torch.clamp(new_y, min=0, max=cls.config.vocab_size - 1)
    return new_y


def exact_sampling(cls, params, rng, tau, xt, t, xt_target=None):
    """Exact categorical simulation."""
    # model.forward() quasi
    logits = cls.backwd_model.get_logits(params, xt, t)
    log_p0t = F.log_softmax(logits, dim=-1)
    
    t_eps = t - tau
    q_teps_0 = cls.fwd_model.transition(t_eps)
    q_teps_0 = torch_utils.expand_dims(q_teps_0, axis=list(range(xt.dim())))
    q_t_teps = cls.fwd_model.transit_between(t_eps, t)
    q_t_teps = q_t_teps.permute(0, 2, 1)

    b = torch_utils.expand_dims(torch.arange(xt.shape[0]), list(range(1, xt.dim())))
    q_t_teps = q_t_teps[b, xt.unsqueeze(-1)]

    qt0 = q_teps_0 * q_t_teps
    log_qt0 = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0))
    log_p0t = log_p0t.unsqueeze(-1)
    log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2)
    cat_dist = torch.distributions.categorical.Categorical(logits=log_prob)
    new_y = cat_dist.sample()
    # new_y = torch.multinomial(torch.exp(log_prob), num_samples=1)
    return new_y  # .squeeze()


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


class BinaryDiffusionModel(DiffusionModel):
    """Binary Model interface."""

    def build_backwd_model(self, config):
        if config.model_type == "ebm":
            backwd_model = torch_ebm.BinaryScoreModel(config)
        elif config.model_type == "hollow":
            backwd_model = torch_hollow_model.HollowModel(config)
        else:
            raise ValueError("Unknown model type %s" % config.model_type)
        return backwd_model

    def sample_step(self, params, rng, tau, xt, t):
        if self.config.get("sampler_type", "binary") == "binary":
            return binary_sample_step(self, params, rng, tau, xt, t)
        else:
            return get_sampler(self.config)(self, params, rng, tau, xt, t)


class CategoricalDiffusionModel(DiffusionModel):
    """Categorical Model interface."""

    def build_backwd_model(self, config):
        if config.model_type == "ebm":
            backwd_model = torch_ebm.CategoricalScoreModel(config)
        elif config.model_type == "hollow":
            backwd_model = torch_hollow_model.HollowModel(config)
        elif config.model_type == "tauldr":
            backwd_model = torch_tauldr_model.TauLDRBackward(config)
        else:
            raise ValueError("Unknown model type %s" % config.model_type)
        return backwd_model

    def sample_step(self, params, rng, tau, xt, t):
        return get_sampler(self.config)(self, params, rng, tau, xt, t)


# DiffusionModel => gets forwardmodel, and backwardmodel and in Backwardmodel the Unet/../ is defined
#
class DiffusionModelPaul(object):
    """Model interface."""

    def build_backwd_model(self, config):
        raise NotImplementedError

    def __init__(self, config):
        self.config = config
        self.optimizer = torch_utils.build_optimizer(config)
        self.backwd_model = None
        self.fwd_model = None
        self.backwd_model = self.build_backwd_model(config)
        self.fwd_model = torch_forward_model.get_fwd_model(config)

    def calc_loss(self, rng, x0):
        rng, loss_rng = torch.manual_seed(rng), torch.manual_seed(rng)
        xt, t = self.fwd_model.sample_xt(x0, self.config.time_duration, rng)
        l, _ = self.backwd_model.loss(rng=rng, x0=x0, xt=xt, t=t)
        return l

    def step_fn(self, state, rng, batch):
        """Single gradient update step."""
        params, opt_state = state.params, state.opt_state
        loss_fn = self.build_loss_func(rng, batch)
        loss, aux = loss_fn(params)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_params = [
            p - lr * p.grad
            for p, lr in zip(params, self.optimizer.param_groups[0]["lr"])
        ]

        ema_params = torch_utils.apply_ema(
            decay=torch.where(state.step == 0, 0.0, self.config.ema_decay),
            avg=state.ema_params,
            new=params,
        )

        new_state = state.replace(
            step=state.step + 1, params=new_params, ema_params=ema_params
        )

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
        rng, prior_rng = torch.manual_seed(rng), torch.manual_seed(rng)
        if num_samples is None:
            num_samples = self.config.plot_samples // (torch.cuda.device_count() or 1)
        x_start = self.sample_from_prior(prior_rng, num_samples, conditioner)
        ones = torch.ones((num_samples,), dtype=torch.float32)
        tau = 1.0 / self.config.sampling_steps

        def sample_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = torch.manual_seed(rng)
            new_y = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            return new_y

        def sample_with_correct_body_fn(step, xt):
            t = ones * tau * (self.config.sampling_steps - step)
            local_rng = torch.manual_seed(rng)
            xt = self.sample_step(state.ema_params, local_rng, tau, xt, t)
            scale = self.config.get("corrector_scale", 1.0)

            for cstep in range(self.config.get("corrector_steps", 0)):
                c_rng = torch.manual_seed(local_rng)
                xt = self.corrector_step(state.ema_params, c_rng, tau * scale, xt, t)

            return xt

        cf = self.config.get("corrector_frac", 0.0)
        corrector_steps = int(cf * self.config.sampling_steps)
        x0 = x_start
        for step in range(0, self.config.sampling_steps - corrector_steps):
            x0 = sample_body_fn(step, x0)

        if corrector_steps > 0:
            for step in range(
                self.config.sampling_steps - corrector_steps, self.config.sampling_steps
            ):
                x0 = sample_with_correct_body_fn(step, x0)

        return x0


class CategoricalDiffusionModelPaul(DiffusionModelPaul):
    """Categorical Model interface."""

    def build_backwd_model(self, config):
        if config.model_type == "ebm":
            backwd_model = torch_ebm.CategoricalScoreModel(config)
        elif config.model_type == "hollow":
            backwd_model = torch_hollow_model.HollowModel(config)
        elif config.model_type == "tauldr":
            backwd_model = torch_tauldr_model.TauLDRBackward(config)
        else:
            raise ValueError("Unknown model type %s" % config.model_type)
        return backwd_model

    def sample_step(self, params, rng, tau, xt, t):
        return get_sampler(self.config)(self, params, rng, tau, xt, t)
