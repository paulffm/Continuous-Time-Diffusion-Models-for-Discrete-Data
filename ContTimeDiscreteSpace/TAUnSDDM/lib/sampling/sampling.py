import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.integrate
import math
from tqdm import tqdm
import lib.sampling.sampling_utils as sampling_utils
import lib.utils.utils as utils
import time
from lib.models.model_utils import get_logprob_with_logits

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


# hier müsste ich für dna sequenzen noch von den werten 0 bis 3 mappen auf A, C, G, T für initial samples
# mit diesem x gehe ich ja dann in model(x, t) => dort one-hot
# Problem: benutze x zum shapen von  qt0_denom, forward rates usw.
# Möglichkeit: könnte schauen, ob SDDM mit 3 Dim input umgehen kann =>
#   =>  deiniere transformer neu => one hot außerhalb


@sampling_utils.register_sampler
class TauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = 1.0
        C, H, W = self.cfg.data.shape
        self.D = C * H * W
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio
        #self.is_ordinal = cfg.sampler.is_ordinal

    def sample(self, model, N, num_intermediates):
        # in init
        # x^{1:D}_{t - h} = x^{1:D}_{t} + sum_{i} P_{i} (\tilde{x^{1:D}_{i} - x^{1:D}_{t})
        #  x^{1:D}_{t - h} = x^{1:D}_{t} + sum_{d} sum_{s\x^{d}_{t}} P_{ds} (s - x^{d}_{t})
        # Pds changes in in dim d zu während time spanne t-h
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
            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                qt0 = model.transition(t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device))  # (N, S, S)
                # p_theta(x_0|x_t) ?

                p0t = F.softmax(
                    model(x, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S) (not log_softmax)

                x_0max = torch.max(p0t, dim=2)[1]
                # if t in save_ts:
                #   x_hist.append(x.clone().detach().cpu().numpy())
                #    x0_hist.append(x_0max.clone().detach().cpu().numpy())

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
                # ll_all = torch.where(ll_all < 1e-35, -1e9, torch.log(ll_all)) => berechnen log
                # reverse_rates = log_weight * forward_rate hier schon
                # h = tau
                diffs = torch.arange(self.S, device=device).view(1, 1, self.S) - x.view(
                    N, self.D, 1
                )  # choices -
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()  # wv flips
                """
                #print("jump_nums", jump_nums, jump_nums.shape)
                #if not self.is_ordinal:
                    tot_jumps = torch.sum(jump_nums, axis=-1, keepdims=True)
                    print("tot_jumps", tot_jumps, tot_jumps.shape)
                    jump_mask = (tot_jumps <= 1).astype(torch.int32)
                    print("jump_mask", jump_mask, jump_mask.shape)
                    jump_nums = jump_nums * jump_mask
                    print("jump_nums", jump_nums, jump_nums.shape)
                """
                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=self.S - 1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(
                model(x, self.min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int)  # , x_hist, x0_hist


@sampling_utils.register_sampler
class PCTauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0

        C, H, W = self.cfg.data.shape
        D = C * H * W
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        device = model.device

        initial_dist = scfg.initial_dist
        initial_dist_std = 200#model.Q_sigma

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist, initial_dist_std)

            h = 1.0 / num_steps  # approximately
            ts = np.linspace(1.0, min_t + h, num_steps)
            save_ts = ts[np.linspace(0, len(ts) - 2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)
                    rate = model.rate(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)

                    p0t = F.softmax(
                        model(in_x, in_t * torch.ones((N,), device=device)), dim=2
                    )  # (N, D, S)

                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = (
                        qt0[
                            torch.arange(N, device=device).repeat_interleave(D * S),
                            torch.arange(S, device=device).repeat(N * D),
                            in_x.long().flatten().repeat_interleave(S),
                        ].view(N, D, S)
                        + eps_ratio
                    )

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0  # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D * S),
                        torch.arange(S, device=device).repeat(N * D),
                        in_x.long().flatten().repeat_interleave(S),
                    ].view(N, D, S)

                    reverse_rates = forward_rates * (
                        (p0t / qt0_denom) @ qt0_numer
                    )  # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten(),
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D * S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N * D),
                    ].view(N, D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1, 1, S) - in_x.view(
                        N, D, 1
                    )
                    poisson_dist = torch.distributions.poisson.Poisson(
                        in_reverse_rates * in_h
                    )
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    unclip_x_new = in_x + overall_jump
                    x_new = torch.clamp(unclip_x_new, min=0, max=S - 1)

                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.detach().cpu().numpy())
                    x0_hist.append(x_0max.detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)

                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t - h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten(),
                        ] = 0.0
                        x = take_poisson_step(
                            x, corrector_rate, corrector_step_size_multiplier * h
                        )

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(
                model(x, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int)#, x_hist, x0_hist


@sampling_utils.register_sampler
class ConditionalTauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        reject_multiple_jumps = scfg.reject_multiple_jumps
        initial_dist = scfg.initial_dist
        if initial_dist == "gaussian":
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, sample_D, device, S, initial_dist, initial_dist_std
            )

            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts) - 2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                qt0 = model.transition(t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device))  # (N, S, S)

                model_input = torch.concat((conditioner, x), dim=1)
                p0t = F.softmax(
                    model(model_input, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                p0t = p0t[:, condition_dim:, :]

                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                qt0_denom = (
                    qt0[
                        torch.arange(N, device=device).repeat_interleave(sample_D * S),
                        torch.arange(S, device=device).repeat(N * sample_D),
                        x.long().flatten().repeat_interleave(S),
                    ].view(N, sample_D, S)
                    + eps_ratio
                )

                # First S is x0 second S is x tilde

                qt0_numer = qt0  # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(sample_D * S),
                    torch.arange(S, device=device).repeat(N * sample_D),
                    x.long().flatten().repeat_interleave(S),
                ].view(N, sample_D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S)

                reverse_rates = forward_rates * inner_sum  # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(sample_D),
                    torch.arange(sample_D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1, 1, S) - x.view(
                    N, sample_D, 1
                )
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()

                if reject_multiple_jumps:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    masked_jump_nums = jump_nums * jump_num_sum_mask.view(
                        N, sample_D, 1
                    )
                    adj_diffs = masked_jump_nums * diffs
                else:
                    adj_diffs = jump_nums * diffs

                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S - 1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(
                model(model_input, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist


@sampling_utils.register_sampler
class ConditionalPCTauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        reject_multiple_jumps = scfg.reject_multiple_jumps
        eps_ratio = scfg.eps_ratio

        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time

        initial_dist = scfg.initial_dist
        if initial_dist == "gaussian":
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, sample_D, device, S, initial_dist, initial_dist_std
            )

            h = 1.0 / num_steps  # approximately
            ts = np.linspace(1.0, min_t + h, num_steps)
            save_ts = ts[np.linspace(0, len(ts) - 2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)
                    rate = model.rate(
                        in_t * torch.ones((N,), device=device)
                    )  # (N, S, S)

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(
                        model(model_input, in_t * torch.ones((N,), device=device)),
                        dim=2,
                    )  # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]

                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = (
                        qt0[
                            torch.arange(N, device=device).repeat_interleave(
                                sample_D * S
                            ),
                            torch.arange(S, device=device).repeat(N * sample_D),
                            x.long().flatten().repeat_interleave(S),
                        ].view(N, sample_D, S)
                        + eps_ratio
                    )

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0  # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D * S),
                        torch.arange(S, device=device).repeat(N * sample_D),
                        in_x.long().flatten().repeat_interleave(S),
                    ].view(N, sample_D, S)

                    reverse_rates = forward_rates * (
                        (p0t / qt0_denom) @ qt0_numer
                    )  # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten(),
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D * S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N * sample_D),
                    ].view(N, sample_D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1, 1, S) - in_x.view(
                        N, sample_D, 1
                    )
                    poisson_dist = torch.distributions.poisson.Poisson(
                        in_reverse_rates * in_h
                    )
                    jump_nums = poisson_dist.sample()

                    if reject_multiple_jumps:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        masked_jump_nums = jump_nums * jump_num_sum_mask.view(
                            N, sample_D, 1
                        )
                        adj_diffs = masked_jump_nums * diffs
                    else:
                        adj_diffs = jump_nums * diffs

                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = in_x + overall_jump
                    x_new = torch.clamp(xp, min=0, max=S - 1)
                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)
                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t - h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten(),
                        ] = 0.0
                        x = take_poisson_step(
                            x, corrector_rate, corrector_step_size_multiplier * h
                        )

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(
                model(model_input, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist


@sampling_utils.register_sampler
class ExactSampling:
    def __init__(self, cfg):
        self.cfg = cfg
        C, H, W = self.cfg.data.shape
        self.D = C * H * W
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        eps_ratio = cfg.sampler.eps_ratio
        self.initial_dist = cfg.sampler.initial_dist

    def sample(self, model, N, num_intermediates):
        start_sample = time.time()

        t = 1.0
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            xt = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(1.0, self.min_t, self.num_steps), np.array([0]))
            )
            # save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                # p_theta(x_0|x_t) ?
                # in HollowModel: get_logits = model()
                # hier x shape von (B, D) => aber in model.forward() wird x umgewandelt zu B, C, h, W für image

                # Entweder in B, D space oder in: hier kann B, D rein, und zwar mit (batch_size, 'ACTG')
                log_p0t = F.log_softmax(
                    model(xt, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                
                t_eps = t - h  # tau

                q_teps_0 = model.transition(
                    t_eps * torch.ones((N,), device=device)
                )  # (N, S, S)
                q_teps_0 = utils.expand_dims(q_teps_0, axis=list(range(1, xt.ndim)))

                q_t_teps = model.transit_between(
                    t_eps * torch.ones((N,), device=device),
                    t * torch.ones((N,), device=device),
                )  # (N, S, S
                q_t_teps = q_t_teps.permute(0, 2, 1)

                b = utils.expand_dims(
                    torch.arange(xt.shape[0]), axis=list(range(1, xt.ndim))
                )
                q_t_teps = q_t_teps[b, xt.long()].unsqueeze(-2)

                start_opt = time.time()
                # extreme slow: mabye p0t = softmax(model(x, t)) and then pt0 @ qt0

                qt0 = q_teps_0 * q_t_teps
                log_qt0 = torch.log(qt0)
                #log_qt0 = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0))

                end_opt = time.time()
                print("sampling operations time", end_opt - start_opt)
                
                log_p0t = log_p0t.unsqueeze(-1)
                log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2)
                cat_dist = torch.distributions.categorical.Categorical(logits=log_prob)
                xt = cat_dist.sample()

            return xt.detach().cpu().numpy().astype(int)


def lbjf_corrector_step(cfg, model, xt, t, h, N, device, xt_target=None):
    """Categorical simulation with lbjf."""
    if xt_target is None:
        xt_target = xt

    logits = model(xt, t * torch.ones((N,), device=device))
    ll_all, ll_xt = get_logprob_with_logits(
        cfg=cfg, model=model, xt=xt, t=t, logits=logits
    )
    log_weight = ll_all - utils.expand_dims(ll_xt, axis=-1)
    fwd_rate = model.rate(t)

    xt_onehot = F.one_hot(xt_target, cfg.data.S)
    posterior = h * (torch.exp(log_weight) * fwd_rate + fwd_rate)
    off_diag = torch.sum(posterior * (1 - xt_onehot), axis=-1, keepdims=True)
    diag = torch.clip(1.0 - off_diag, a_min=0)
    posterior = posterior * (1 - xt_onehot) + diag * xt_onehot
    posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
    log_posterior = torch.log(posterior + 1e-35)
    new_y = torch.distributions.categorical.Categorical(log_posterior).sample()
    return new_y


@sampling_utils.register_sampler
class LBJFSampling:
    def __init__(self, cfg):
        self.cfg = cfg
        C, H, W = self.cfg.data.shape
        self.D = C * H * W
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps

    def sample(self, model, N, num_intermediates):
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
                # p_theta(x_0|x_t) ?

                # stellt sich frage:
                # Entweder in B, D space oder in: hier kann B, D rein, und zwar mit (batch_size, 'ACTG')
                logits = model(x, t * torch.ones((N,), device=device))

                ll_all, ll_xt = get_logprob_with_logits(
                    cfg=self.cfg, model=model, xt=x, t=t * torch.ones((N,)), logits=logits
                )

                log_weight = ll_all - ll_xt.unsqueeze(-1) # B, D, S - B, D, 1
                fwd_rate = model.rate_mat(x, t * torch.ones((N,))) # B, D, S?

                xt_onehot = F.one_hot(x, self.S)

                posterior = h * torch.exp(log_weight) * fwd_rate # eq.17 c != x^d_t
                off_diag = torch.sum(
                    posterior * (1 - xt_onehot), axis=-1, keepdims=True
                )
                diag = torch.clip(1.0 - off_diag, min=0, max=float("inf"))
                posterior = posterior * (1 - xt_onehot) + diag * xt_onehot # eq.17

                posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
                log_posterior = torch.log(posterior + 1e-35)
                x = torch.distributions.categorical.Categorical(log_posterior).sample()

                """
                if t <= self.corrector_entry_time:
                    print("corrector")
                    for _ in range(self.num_corrector_steps):
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)
                        logits = model(x, t * torch.ones((N,), device=device))
                        ll_all, ll_xt = get_logprob_with_logits(
                            cfg=self.cfg, model=model, xt=x, t=t, logits=logits
                        )
                        log_weight = ll_all - ll_xt.unsqueeze(-1)
                        fwd_rate = model.rate_mat(x, t * torch.ones((N,)))

                        xt_onehot = F.one_hot(x, self.S)
                        posterior = h * (torch.exp(log_weight) * fwd_rate + fwd_rate)
                        off_diag = torch.sum(
                            posterior * (1 - xt_onehot), axis=-1, keepdims=True
                        )
                        diag = torch.clip(1.0 - off_diag, min=0, max=float("inf"))
                        posterior = posterior * (1 - xt_onehot) + diag * xt_onehot
                        posterior = posterior / torch.sum(
                            posterior, axis=-1, keepdims=True
                        )
                        log_posterior = torch.log(posterior + 1e-35)
                        x = torch.distributions.categorical.Categorical(
                            log_posterior
                        ).sample()
                """
            return x.detach().cpu().numpy().astype(int)


@sampling_utils.register_sampler
class ExactSamplingNotLog:
    def __init__(self, cfg):
        self.cfg = cfg
        C, H, W = self.cfg.data.shape
        self.D = C * H * W
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        eps_ratio = cfg.sampler.eps_ratio
        self.initial_dist = cfg.sampler.initial_dist

    def sample(self, model, N, num_intermediates):
        start_sample = time.time()

        t = 1.0
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            xt = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(1.0, self.min_t, self.num_steps), np.array([0]))
            )
            # save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                # p_theta(x_0|x_t) ?
                # in HollowModel: get_logits = model()
                # hier x shape von (B, D) => aber in model.forward() wird x umgewandelt zu B, C, h, W für image

                # Entweder in B, D space oder in: hier kann B, D rein, und zwar mit (batch_size, 'ACTG')
                p0t = F.softmax(
                    model(xt, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                
                t_eps = t - h  # tau

                q_teps_0 = model.transition(
                    t_eps * torch.ones((N,), device=device)
                )  # (N, S, S)
                q_teps_0 = utils.expand_dims(q_teps_0, axis=list(range(1, xt.ndim)))
                print("q_teps_0", q_teps_0.shape)
                q_t_teps = model.transit_between(
                    t_eps * torch.ones((N,), device=device),
                    t * torch.ones((N,), device=device),
                )  # (N, S, S
                q_t_teps = q_t_teps.permute(0, 2, 1)

                b = utils.expand_dims(
                    torch.arange(xt.shape[0]), axis=list(range(1, xt.ndim))
                )
                q_t_teps = q_t_teps[b, xt.long()].unsqueeze(-2)
                print("q_t_teps", q_t_teps.shape)
                print("p0t", p0t.shape)
                start_opt = time.time()
                # extreme slow: mabye p0t = softmax(model(x, t)) and then pt0 @ qt0
                qt0 = q_teps_0 * q_t_teps # N, D, S, S
                print("qt0", qt0.shape)
                
                #log_qt0 = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0))
                p0t = p0t.unsqueeze(-1)
                print("p0t", p0t.shape)
                #prob = torch.logsumexp(p0t + qt0, dim=-2)

                """
                p0t = torch.exp(log_p0t)
                qt0 = torch.exp(log_qt0)
                prob = torch.sum(p0t * qt0, dim=-2)
                """
                
                prob = p0t @ qt0
                print("prob", prob, prob.shape)
                end_opt = time.time()
                print("sampling operations time", end_opt - start_opt)

                cat_dist = torch.distributions.categorical.Categorical(prob)
                xt = cat_dist.sample()

            return xt.detach().cpu().numpy().astype(int)