import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import lib.sampling.sampling_utils as sampling_utils
import lib.utils.utils as utils
import time


def expand_dims(x, axis):
    # if axis == 0:
    #   x =x.unsqueeze(0)

    for i in axis:
        x = x.unsqueeze(i)
    return x


def get_logprob_with_logits(cfg, model, xt, t, logits, xt_target=None):
    """Get logprob with logits."""

    if xt_target is None:
        xt_target = xt
    xt_onehot = F.one_hot(xt_target.long(), cfg.data.S)
    if cfg.loss.logit_type == "direct":
        log_prob = F.log_softmax(logits, dim=-1)
    else:
        qt0 = model.transition(t)
        if cfg.loss.logit_type == "reverse_prob":
            p0t = F.softmax(logits, dim=-1)
            qt0 = utils.expand_dims(qt0, axis=list(range(1, xt.dim() - 1)))
            prob_all = p0t @ qt0
            log_prob = torch.log(prob_all + 1e-35)

            # check
        elif cfg.loss.logit_type == "reverse_logscale":
            log_p0t = F.log_softmax(logits, dim=-1)
            log_qt0 = torch.where(qt0 <= 1e-35, -1e9, torch.log(qt0))
            log_qt0 = utils.expand_dims(log_qt0, axis=list(range(1, xt.dim())))
            log_p0t = log_p0t.unsqueeze(-1)
            log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2)
            # check
        else:
            raise ValueError("Unknown logit_type: %s" % cfg.loss.logit_type)
    log_xt = torch.sum(log_prob * xt_onehot, dim=-1)  # log probability of true class

    return log_prob, log_xt


def get_initial_samples(N, D, device, S, initial_dist, initial_dist_std=None):
    if initial_dist == "uniform":
        x = torch.randint(low=0, high=S, size=(N, D), device=device)
    elif initial_dist == "gaussian":
        target = np.exp(
            -((np.arange(1, S + 1) - S // 2) ** 2) / (2 * initial_dist_std**2)
        )
        target = target / np.sum(target)

        cat = torch.distributions.categorical.Categorical(torch.from_numpy(target))
        x = cat.sample((N * D,)).view(N, D)
        x = x.to(device)
    else:
        raise NotImplementedError("Unrecognized initial dist " + initial_dist)
    return x


@sampling_utils.register_sampler
class TauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = 1.0
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio
        self.is_ordinal = cfg.sampler.is_ordinal

    def sample(self, model, N, num_intermediates=None):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(1.0, self.min_t, self.num_steps), np.array([0]))
            )
            t = 1.0

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                qt0 = model.transition(t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device))  # (N, S, S)
                # p_theta(x_0|x_t) ?

                p0t = F.softmax(
                    model(x, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S) (not log_softmax)

                x_0max = torch.max(p0t, dim=2)[1]

                qt0_denom = (
                    qt0[
                        torch.arange(N, device=device).repeat_interleave(
                            self.D * self.S
                        ),
                        torch.arange(self.S, device=device).repeat(N * self.D),
                        x.long().flatten().repeat_interleave(self.S),
                    ].view(N, self.D, self.S)
                    + self.eps_ratio
                )

                # First S is x0 second S is x tilde
                qt0_numer = qt0  # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S)
                reverse_rates = forward_rates * inner_sum  # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0

                diffs = torch.arange(self.S, device=device).view(1, 1, self.S) - x.view(
                    N, self.D, 1
                )  # choices -
                poisson_dist = torch.distributions.poisson.Poisson(
                    reverse_rates * h
                )  # posterior: p_{t-eps|t}, B, D; S
                jump_nums = (
                    poisson_dist.sample()
                )  # how many jumps in interval [t-eps, t]

                if not self.is_ordinal:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)

                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=self.S - 1)

                x = x_new

            p_0gt = F.softmax(
                model(x, self.min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int)


@sampling_utils.register_sampler
class LBJFSampling:
    def __init__(self, cfg):
        self.cfg = cfg
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps

    def sample(self, model, N, num_intermediates=None):
        t = 1.0
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(1.0, self.min_t, self.num_steps), np.array([0]))
            )

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]
                # p_theta(x_0|x_t) 

                logits = model(x, t * torch.ones((N,), device=device))

                ll_all, ll_xt = get_logprob_with_logits(
                    cfg=self.cfg,
                    model=model,
                    xt=x,
                    t=t * torch.ones((N,), device=device),
                    logits=logits,
                )

                log_weight = ll_all - ll_xt.unsqueeze(-1)  # B, D, S - B, D, 1
                # fwd_rate same as forward_rates from Campbell implementation
                fwd_rate = model.rate_mat(
                    x, t * torch.ones((N,), device=device)
                )  # B, D, S

                xt_onehot = F.one_hot(x, self.S)

                posterior = h * torch.exp(log_weight) * fwd_rate  # eq.17 c != x^d_t
                # posterior * (1 - xt_onehot) same as reverse_rates[..] = 0.0
                off_diag = torch.sum(
                    posterior * (1 - xt_onehot), axis=-1, keepdims=True
                )
                diag = torch.clip(1.0 - off_diag, min=0, max=float("inf"))
                posterior = posterior * (1 - xt_onehot) + diag * xt_onehot  # eq.17

                posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
                log_posterior = torch.log(posterior + 1e-35).view(-1, self.S)
                x = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )

            return x.detach().cpu().numpy().astype(int)


@sampling_utils.register_sampler
class TauLeapingScoreBasedPaper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.is_ordinal = cfg.sampler.is_ordinal

    def sample(self, model, N, num_intermediates=None):
        t = 1.0
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(1.0, self.min_t, self.num_steps), np.array([0]))
            )

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                logits = model(x, t * torch.ones((N,), device=device))

                ll_all, ll_xt = get_logprob_with_logits(
                    cfg=self.cfg,
                    model=model,
                    xt=x,
                    t=t * torch.ones((N,), device=device),
                    logits=logits,
                )

                log_weight = ll_all - ll_xt.unsqueeze(-1)  # B, D, S - B, D, 1
                # fwd_rate same as forward_rates from Campbell implementation
                fwd_rate = model.rate_mat(
                    x.long(), t * torch.ones((N,), device=device)
                )  # B, D, S

                posterior = h * torch.exp(log_weight) * fwd_rate  # eq.17 c != x^d_t
                xt_onehot = F.one_hot(x.long(), self.S)
                posterior = posterior * (1 - xt_onehot)

                flips = torch.distributions.poisson.Poisson(
                    posterior
                ).sample()  # B, D most 0
                choices = utils.expand_dims(
                    torch.arange(self.S, device=device, dtype=torch.int32),
                    axis=list(range(x.ndim)),
                )  # 1,1, S
                if not self.is_ordinal:
                    tot_flips = torch.sum(flips, axis=-1, keepdims=True)
                    flip_mask = (tot_flips <= 1) * 1
                    flips = flips * flip_mask
                diff = choices - x.unsqueeze(-1)

                avg_offset = torch.sum(
                    flips * diff, axis=-1
                )  # B, D, S with entries -(S - 1) to S-1

                x = x + avg_offset
                x = torch.clip(x, min=0, max=self.S - 1)

            return x.detach().cpu().numpy().astype(int)
