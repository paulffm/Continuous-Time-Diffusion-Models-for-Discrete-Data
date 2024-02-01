import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import lib.sampling.sampling_utils as sampling_utils
import lib.utils.utils as utils
import time
from functools import partial
from lib.models.model_utils import get_logprob_with_logits
import time


# Sampling observations:
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


def get_reverse_rates(model, logits, x, t_ones, cfg, N, D, S):
    if cfg.loss.name == "CTElbo" or cfg.loss.name == "TauLDRNLL":
        device = model.device
        qt0 = model.transition(t_ones)  # (N, S, S)
        rate = model.rate(t_ones)  # (N, S, S)

        p0t = F.softmax(logits, dim=2)  # (N, D, S) (not log_softmax)

        qt0_denom = (
            qt0[
                torch.arange(N, device=device).repeat_interleave(D * S),
                torch.arange(S, device=device).repeat(N * D),
                x.long().flatten().repeat_interleave(S),
            ].view(N, D, S)
            + cfg.sampler.eps_ratio
        )

        # First S is x0 second S is x tilde
        qt0_numer = qt0  # (N, S, S)

        forward_rates = rate[
            torch.arange(N, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(N * D),
            x.long().flatten().repeat_interleave(S),
        ].view(N, D, S)

        ratio = (p0t / qt0_denom) @ qt0_numer  # (N, D, S)

        reverse_rates = forward_rates * ratio  # (N, D, S)

    else:
        ll_all, ll_xt = get_logprob_with_logits(
            cfg=cfg,
            model=model,
            xt=x,
            t=t_ones,
            logits=logits,
        )

        log_weight = ll_all - ll_xt.unsqueeze(-1)  # B, D, S - B, D, 1
        fwd_rate = model.rate_mat(x.long(), t_ones)  # B, D, S
        ratio = torch.exp(log_weight)
        reverse_rates = ratio * fwd_rate

        # B, D, S
    return reverse_rates, ratio


@sampling_utils.register_sampler
class TauL:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
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
        self.loss_name = cfg.loss.name

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            change_dim = []
            change_jumps = []
            change_mjumps = []
            # new_tensor = torch.empty((self.num_steps, N, self.D), device=device)

            time_list = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                start_time = time.time()
                # new_tensor[idx] = x
                h = ts[idx] - ts[idx + 1]
                t_ones = t * torch.ones((N,), device=device)  # (N, S, S)
                logits = model(x, t_ones)

                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                xt_onehot = F.one_hot(x.long(), self.S)
                reverse_rates = reverse_rates * (1 - xt_onehot)
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

                    # proportion of jumps
                    # jump_num_sum = torch.sum(jump_nums, dim=2)
                    # wv dim changen
                    # change_dim.append(torch.mean(((jump_num_sum > 0) * 1).to(dtype=float)).item())
                    # in wv dim sind multiple jumps
                    # change_jumps.append(torch.mean(((jump_num_sum > 1) * 1).to(dtype=float)).item())
                    # wv prop von changes sind multiple changes
                    # changes = torch.sum(((jump_num_sum > 0) * 1).to(dtype=float))
                    # changes_rej = torch.sum(((jump_num_sum > 1) * 1).to(dtype=float))
                    # change_mjumps.append((changes_rej/changes).item())

                choices = utils.expand_dims(
                    torch.arange(self.S, device=device, dtype=torch.int32),
                    axis=list(range(x.ndim)),
                )
                diff = choices - x.unsqueeze(-1)
                adj_diffs = jump_nums * diff
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump

                x_new = torch.clamp(xp, min=0, max=self.S - 1)

                x = x_new
                if t <= self.corrector_entry_time:
                    print("corrector")
                    for _ in range(self.num_corrector_steps):
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)

                        t_h_ones = (t) * torch.ones((N,), device=device)
                        rate = model.rate(t_h_ones)

                        logits = model(x_new, t_h_ones)  #
                        reverse_rates, _ = get_reverse_rates(
                            model, logits, x_new, t_h_ones, self.cfg, N, self.D, self.S
                        )
                        reverse_rates[
                            torch.arange(N, device=device).repeat_interleave(self.D),
                            torch.arange(self.D, device=device).repeat(N),
                            x_new.long().flatten(),
                        ] = 0.0

                        transpose_forward_rates = rate[
                            torch.arange(N, device=device).repeat_interleave(
                                self.D * self.S
                            ),
                            x_new.long().flatten().repeat_interleave(self.S),
                            torch.arange(self.S, device=device).repeat(N * self.D),
                        ].view(N, self.D, self.S)

                        corrector_rates = (
                            transpose_forward_rates + reverse_rates
                        )  # (N, D, S)
                        corrector_rates[
                            torch.arange(N, device=device).repeat_interleave(self.D),
                            torch.arange(self.D, device=device).repeat(N),
                            x_new.long().flatten(),
                        ] = 0.0
                        poisson_dist = torch.distributions.poisson.Poisson(
                            corrector_rates * h
                        )  # posterior: p_{t-eps|t}, B, D; S

                        jump_nums = (
                            poisson_dist.sample()
                        )  # how many jumps in interval [t-eps, t]

                        if not self.is_ordinal:
                            jump_num_sum = torch.sum(jump_nums, dim=2)
                            jump_num_sum_mask = jump_num_sum <= 1
                            jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)
                        choices = utils.expand_dims(
                            torch.arange(self.S, device=device, dtype=torch.int32),
                            axis=list(range(x_new.ndim)),
                        )
                        diff = choices - x_new.unsqueeze(-1)
                        adj_diffs = jump_nums * diff
                        overall_jump = torch.sum(adj_diffs, dim=2)
                        xp = x_new + overall_jump

                        x_new = torch.clamp(xp, min=0, max=self.S - 1)
                        x = x_new

            if self.loss_name == "CTElbo":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x

            return (
                x_0max.detach().cpu().numpy().astype(int),  # change_jump,
                change_dim,
            )  # , x_hist, x0_hist


@sampling_utils.register_sampler
class LBJF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio
        self.loss_name = cfg.loss.name

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            change_jump = []
            new_tensor = torch.empty((self.num_steps, N, self.D), device=device)
            for idx, t in tqdm(enumerate(ts[0:-1])):
                new_tensor[idx] = x
                h = ts[idx] - ts[idx + 1]
                t_ones = t * torch.ones((N,), device=device)
                qt0 = model.transition(t_ones)  # (N, S, S)
                rate = model.rate(t_ones)
                logits = model(x, t_ones)
                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                xt_onehot = F.one_hot(x.long(), self.S)
                post_0 = reverse_rates * (1 - xt_onehot)

                off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                reverse_rates = post_0 * h + diag * xt_onehot  # * h  # eq.17

                reverse_rates = reverse_rates / torch.sum(
                    reverse_rates, axis=-1, keepdims=True
                )
                log_posterior = torch.log(reverse_rates + 1e-35).view(-1, self.S)
                x_new = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )
                if t <= self.corrector_entry_time:
                    for _ in range(self.num_corrector_steps):
                        print("corrector")
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)
                        t_h_ones = t * torch.ones((N,), device=device)
                        logits = model(x_new, t_h_ones)  #
                        reverse_rates, _ = get_reverse_rates(
                            model, logits, x_new, t_h_ones, self.cfg, N, self.D, self.S
                        )

                        transpose_forward_rates = rate[
                            torch.arange(N, device=device).repeat_interleave(
                                self.D * self.S
                            ),
                            x_new.long().flatten().repeat_interleave(self.S),
                            torch.arange(self.S, device=device).repeat(N * self.D),
                        ].view(N, self.D, self.S)

                        corrector_rates = (
                            transpose_forward_rates + reverse_rates
                        )  # (N, D, S)
                        corrector_rates[
                            torch.arange(N, device=device).repeat_interleave(self.D),
                            torch.arange(self.D, device=device).repeat(N),
                            x_new.long().flatten(),
                        ] = 0.0
                        xt_new_onehot = F.one_hot(x_new.long(), self.S)
                        post_0 = corrector_rates * (1 - xt_new_onehot)

                        off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                        diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                        corrector_rates = post_0 * h + diag * xt_new_onehot
                        corrector_rates = corrector_rates / torch.sum(
                            corrector_rates, axis=-1, keepdims=True
                        )
                        log_posterior = torch.log(corrector_rates + 1e-35).view(
                            -1, self.S
                        )

                        x_new = (
                            torch.distributions.categorical.Categorical(
                                logits=log_posterior
                            )
                            .sample()
                            .view(N, self.D)
                        )
                # print(torch.sum(x_new != x, dim=1))
                change_jump.append((torch.sum(x_new != x) / (N * self.D)).item())
                x = x_new
            if self.loss_name == "CTElbo":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x
            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_jump
                # new_tensor.detach().cpu().numpy().astype(int),
            )  # , x_hist, x0_hist


@sampling_utils.register_sampler
class MidPointTauL:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.is_ordinal = cfg.sampler.is_ordinal
        self.device = cfg.device
        self.eps_ratio = cfg.sampler.eps_ratio
        self.loss_name = cfg.loss.name

        if cfg.data.name == "DiscreteMNIST":
            self.state_change = -torch.load(
                "SavedModels/MNIST/state_change_matrix_mnist.pth"
            )
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "Maze3S":
            self.state_change = -torch.tensor(
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], device=self.device
            )
            # self.state_change = torch.tensor([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], device=self.device)
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "BinMNIST" or cfg.data.name == "SyntheticData":
            self.state_change = -torch.tensor([[0, 1], [-1, 0]], device=self.device)

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        self.state_change = torch.tile(self.state_change, (N, 1, 1))
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )

            t = self.max_t
            change_jump = []
            change_clamp1 = []
            change_clamp2 = []
            change_dim = []
            change_dim_first = []
            change_1to2 = []
            h = (self.max_t - self.min_t) / self.num_steps
            # Fragen:
            # 1. Prediction zum  Zeitpunkt 0.5 * h +t_ones?
            # Wie summe über states? => meistens R * changes = 0
            #
            i = 0
            count = 0
            while t - 0.5 * h > self.min_t:
                t_ones = t * torch.ones((N,), device=device)  # (N, S, S)
                t_05 = t_ones - 0.5 * h
                logits = model(x, t_ones)

                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0

                # print(t, reverse_rates)
                # achtung ein verfahren definieren mit:
                # x_prime und eins mit echtem

                state_change = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                xt_onehot = F.one_hot(x.long(), self.S)

                change = torch.round(
                    0.5 * h * torch.sum((reverse_rates * state_change), dim=-1)
                ).to(dtype=torch.int)
                xp_prime = x + change  # , dim=-1)

                """
                if (change == torch.zeros((N, self.D), device=self.device)).all():
                    print("True")
                else:
                    count = count + 1
                    print("False")
                """
                x_prime = torch.clip(xp_prime, min=0, max=self.S - 1)

                # change_clamp1.append((torch.sum(xp_prime != x_prime) / (N * self.D)).item())
                change_dim_first.append((torch.sum(x != x_prime) / (N * self.D)).item())

                # ------------second-------------------
                logits_prime = model(x_prime, t_05)

                reverse_rates_prime, _ = get_reverse_rates(
                    model, logits_prime, x_prime, t_05, self.cfg, N, self.D, self.S
                )

                reverse_rates_prime[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x_prime.long().flatten(),
                ] = 0.0

                state_change_prime = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x_prime.long()
                    .flatten()
                    .repeat_interleave(self.S),  # wenn hier x_prime
                ].view(N, self.D, self.S)

                diff_prime = state_change_prime

                flips = torch.distributions.poisson.Poisson(
                    reverse_rates_prime * h
                ).sample()  # B, D most 0

                if not self.is_ordinal:
                    tot_flips = torch.sum(flips, axis=-1, keepdims=True)
                    flip_mask = (tot_flips <= 1) * 1
                    flips = flips * flip_mask
                else:
                    jump_num_sum = torch.sum(flips, axis=-1)
                    changes = torch.sum(((jump_num_sum > 0) * 1).to(dtype=float))
                    # print("1", changes)
                    changes_rej = torch.sum(((jump_num_sum > 1) * 1).to(dtype=float))
                    # proportion of jumps thate are multiple jumps
                    change_jump.append((changes_rej / changes).item())
                # diff = choices - x.unsqueeze(-1)

                avg_offset = torch.sum(
                    flips * diff_prime, axis=-1
                )  # B, D, S with entries -(S - 1) to S-1
                xp = x + avg_offset  # wenn hier x_prime

                x_new = torch.clip(xp, min=0, max=self.S - 1)
                # change_clamp2.append((torch.sum(xp != x) / (N * self.D)).item())
                change_dim.append((torch.sum(xp != x) / (N * self.D)).item())
                change_1to2.append((torch.sum(x_prime != x_new) / (N * self.D)).item())

                x = x_new
                t = t - h
                i += 1

            if self.loss_name == "CTElbo":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x

            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_jump,
                change_dim,
                change_dim_first,
                change_1to2,
            )  # change_clamp1, change_clamp2


@sampling_utils.register_sampler
class MidPointSampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.is_ordinal = cfg.sampler.is_ordinal
        self.device = cfg.device
        self.eps_ratio = cfg.sampler.eps_ratio
        self.sample_prime = True  # False # True
        self.loss_name = cfg.loss.name

        if cfg.data.name == "DiscreteMNIST":
            self.state_change = -torch.load(
                "SavedModels/MNIST/state_change_matrix_mnist.pth"
            )
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "Maze3S":
            if self.is_ordinal:
                self.state_change = -torch.tensor(
                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], device=self.device
                )
                # self.state_change = self.state_change.to(device=self.device)
            else:
                self.state_change = -torch.load(
                    "SavedModels/MAZE/state_change_matrix_maze.pth"
                )
                self.state_change = -torch.tensor(
                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], device=self.device
                )
                # self.state_change = torch.tensor([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], device=self.device)
                self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "BinMNIST" or cfg.data.name == "SyntheticData":
            self.state_change = -torch.tensor([[0, 1], [-1, 0]], device=self.device)

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        self.state_change = torch.tile(self.state_change, (N, 1, 1))
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )

            t = self.max_t
            change_jump = []
            change_clamp = []
            h = t / self.num_steps
            # Fragen:
            # 1. Prediction zum  Zeitpunkt 0.5 * h +t_ones?
            # Wie summe über states? => meistens R * changes = 0
            #
            i = 1
            count = 0
            while t - 0.5 * h > self.min_t:
                t_ones = t * torch.ones((N,), device=device)  # (N, S, S)
                t_05 = t_ones - 0.5 * h

                logits = model(x, t_ones)

                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                """
                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0
                """
                # print(t, reverse_rates)
                # achtung ein verfahren definieren mit:
                # x_prime und eins mit echtem

                xt_onehot = F.one_hot(x.long(), self.S)
                post_0 = reverse_rates * (1 - xt_onehot)

                off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                diag = torch.clip(1.0 - 0.5 * h * off_diag, min=0, max=float("inf"))
                reverse_rates = post_0 * 0.5 * h + diag * xt_onehot  # * h  # eq.17

                reverse_rates = reverse_rates / torch.sum(
                    reverse_rates, axis=-1, keepdims=True
                )
                log_posterior = torch.log(reverse_rates + 1e-35).view(-1, self.S)
                x_prime = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )

                x_prime = torch.clip(x_prime, min=0, max=self.S - 1)

                # ------------second-------------------
                logits_prime = model(x_prime, t_05)

                reverse_rates_prime, _ = get_reverse_rates(
                    model, logits_prime, x_prime, t_05, self.cfg, N, self.D, self.S
                )

                reverse_rates_prime[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x_prime.long().flatten(),
                ] = 0.0

                state_change_prime = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x_prime.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                diff_prime = state_change_prime

                flips = torch.distributions.poisson.Poisson(
                    reverse_rates_prime * h
                ).sample()  # B, D most 0

                if not self.is_ordinal:
                    tot_flips = torch.sum(flips, axis=-1, keepdims=True)
                    flip_mask = (tot_flips <= 1) * 1
                    flips = flips * flip_mask
                # diff = choices - x.unsqueeze(-1)

                avg_offset = torch.sum(
                    flips * diff_prime, axis=-1
                )  # B, D, S with entries -(S - 1) to S-1
                xp = x_prime + avg_offset

                # change_jump.append((torch.sum(xp != x_prime) / (N * self.D)).item())
                x_new = torch.clip(xp, min=0, max=self.S - 1)
                change_clamp.append((torch.sum(xp != x_new) / (N * self.D)).item())

                x = x_new
                t = t - h
                i += 1
            if self.loss_name == "CTElbo":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x

            return x_0max.detach().cpu().numpy().astype(int), change_jump


@sampling_utils.register_sampler
class PCLT:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio
        self.loss_name = cfg.loss.name
        self.is_ordinal = cfg.sampler.is_ordinal
        if self.num_corrector_steps == 0:
            self.num_corrector_steps = 1

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            change_jump = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]
                t_ones = t * torch.ones((N,), device=device)
                qt0 = model.transition(t_ones)  # (N, S, S)
                rate = model.rate(t_ones)
                logits = model(x, t_ones)
                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                xt_onehot = F.one_hot(x.long(), self.S)
                post_0 = reverse_rates * (1 - xt_onehot)

                off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                reverse_rates = post_0 * h + diag * xt_onehot  # * h  # eq.17

                reverse_rates = reverse_rates / torch.sum(
                    reverse_rates, axis=-1, keepdims=True
                )
                log_posterior = torch.log(reverse_rates + 1e-35).view(-1, self.S)
                x_new = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )
                if t <= self.corrector_entry_time:
                    print("corrector")
                    for _ in range(self.num_corrector_steps):
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)
                        t_h_ones = t * torch.ones((N,), device=device)
                        logits = model(x_new, t_h_ones)  #
                        reverse_rates, _ = get_reverse_rates(
                            model, logits, x_new, t_h_ones, self.cfg, N, self.D, self.S
                        )

                        transpose_forward_rates = rate[
                            torch.arange(N, device=device).repeat_interleave(
                                self.D * self.S
                            ),
                            x_new.long().flatten().repeat_interleave(self.S),
                            torch.arange(self.S, device=device).repeat(N * self.D),
                        ].view(N, self.D, self.S)

                        corrector_rates = (
                            reverse_rates  # + transpose_forward_rates  # (N, D, S)
                        )
                        corrector_rates[
                            torch.arange(N, device=device).repeat_interleave(self.D),
                            torch.arange(self.D, device=device).repeat(N),
                            x_new.long().flatten(),
                        ] = 0.0
                        poisson_dist = torch.distributions.poisson.Poisson(
                            corrector_rates * h
                        )  # posterior: p_{t-eps|t}, B, D; S

                        jump_nums = (
                            poisson_dist.sample()
                        )  # how many jumps in interval [t-eps, t]

                        if not self.is_ordinal:
                            jump_num_sum = torch.sum(jump_nums, dim=2)
                            jump_num_sum_mask = jump_num_sum <= 1
                            jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)
                        choices = utils.expand_dims(
                            torch.arange(self.S, device=device, dtype=torch.int32),
                            axis=list(range(x_new.ndim)),
                        )
                        diff = choices - x_new.unsqueeze(-1)
                        adj_diffs = jump_nums * diff
                        overall_jump = torch.sum(adj_diffs, dim=2)
                        xp = x + overall_jump

                        change_jump.append((torch.sum(xp != x) / (N * self.D)).item())

                        x_new = torch.clamp(xp, min=0, max=self.S - 1)
                change_jump.append((torch.sum(x_new != x) / (N * self.D)).item())

                x = x_new
            if self.loss_name == "CTElbo":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x
            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_jump,
            )  # , x_hist, x0_hist


class RK4:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.is_ordinal = cfg.sampler.is_ordinal
        self.device = cfg.device
        self.eps_ratio = cfg.sampler.eps_ratio
        self.sample_prime = True
        self.loss_name = cfg.loss.name

        if cfg.data.name == "DiscreteMNIST":
            self.state_change = -torch.load(
                "SavedModels/MNIST/state_change_matrix_mnist.pth"
            )
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "Maze3S":
            if self.is_ordinal:
                self.state_change = -torch.tensor(
                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], device=self.device
                )
                # self.state_change = self.state_change.to(device=self.device)
            else:
                self.state_change = -torch.load(
                    "SavedModels/MAZE/state_change_matrix_maze.pth"
                )
                self.state_change = -torch.tensor(
                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], device=self.device
                )
                # self.state_change = torch.tensor([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], device=self.device)
                self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "BinMNIST" or cfg.data.name == "SyntheticData":
            self.state_change = -torch.tensor([[0, 1], [-1, 0]], device=self.device)

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device
        self.state_change = torch.tile(self.state_change, (N, 1, 1))
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )

            t = self.max_t
            change_jump = []
            change_clamp = []
            h = t / self.num_steps
            # Fragen:
            # 1. Prediction zum  Zeitpunkt 0.5 * h +t_ones?
            # Wie summe über states? => meistens R * changes = 0
            #
            i = 1
            while t - 0.5 * h > self.min_t:
                t_ones = t * torch.ones((N,), device=device)  # (N, S, S)
                t_05 = t_ones - 0.5 * h

                logits = model(x, t_ones)

                reverse_rates, _ = get_reverse_rates(
                    model, logits, x, t_ones, self.cfg, N, self.D, self.S
                )

                """
                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0
                """
                print(t, reverse_rates)
                if self.sample_prime:
                    xt_onehot = F.one_hot(x.long(), self.S)
                    post_0 = reverse_rates * (1 - xt_onehot)

                    off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                    diag = torch.clip(1.0 - 0.5 * h * off_diag, min=0, max=float("inf"))
                    reverse_rates = post_0 * 0.5 * h + diag * xt_onehot  # * h  # eq.17

                    reverse_rates = reverse_rates / torch.sum(
                        reverse_rates, axis=-1, keepdims=True
                    )
                    log_posterior = torch.log(reverse_rates + 1e-35).view(-1, self.S)
                    x_prime = (
                        torch.distributions.categorical.Categorical(
                            logits=log_posterior
                        )
                        .sample()
                        .view(N, self.D)
                    )
                else:
                    state_change = self.state_change[
                        torch.arange(N, device=device).repeat_interleave(
                            self.D * self.S
                        ),
                        torch.arange(self.S, device=device).repeat(N * self.D),
                        x.long().flatten().repeat_interleave(self.S),
                    ].view(N, self.D, self.S)
                    change = torch.round(
                        torch.sum((0.5 * h * reverse_rates * state_change), dim=-1)
                    ).to(dtype=torch.int)
                    x_prime = x + change  # , dim=-1)
                    print(
                        (change == torch.zeros((N, self.D), device=self.device)).all()
                    )
                x_prime = torch.clip(x_prime, min=0, max=self.S - 1)

                # ------------second-------------------
                logits_prime = model(x_prime, t_05)

                reverse_rates_prime, _ = get_reverse_rates(
                    model, logits_prime, x_prime, t_05, self.cfg, N, self.D, self.S
                )

                reverse_rates_prime[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x_prime.long().flatten(),
                ] = 0.0

                state_change_prime = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x_prime.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                diff_prime = state_change_prime

                flips = torch.distributions.poisson.Poisson(
                    reverse_rates_prime * h
                ).sample()  # B, D most 0

                if not self.is_ordinal:
                    tot_flips = torch.sum(flips, axis=-1, keepdims=True)
                    flip_mask = (tot_flips <= 1) * 1
                    flips = flips * flip_mask
                # diff = choices - x.unsqueeze(-1)

                avg_offset = torch.sum(
                    flips * diff_prime, axis=-1
                )  # B, D, S with entries -(S - 1) to S-1
                xp = x_prime + avg_offset

                # change_jump.append((torch.sum(xp != x_prime) / (N * self.D)).item())
                x_new = torch.clip(xp, min=0, max=self.S - 1)
                change_clamp.append((torch.sum(xp != x_new) / (N * self.D)).item())

                x = x_new
                t = t - h
                print(i)
                i += 1
            if self.loss_name == "CTElbo":
                p_0gt = F.softmax(
                    model(x, self.min_t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S)
                x_0max = torch.max(p_0gt, dim=2)[1]
            else:
                x_0max = x

            return x_0max.detach().cpu().numpy().astype(int), change_jump


@sampling_utils.register_sampler
class PCTauL:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N):
        t = 1.0
        D = self.cfg.model.concat_dim
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
        initial_dist_std = 200  # model.Q_sigma

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist, initial_dist_std)

            h = 1.0 / num_steps  # approximately
            ts = np.linspace(1.0, min_t + h, num_steps)

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

            p_0gt = F.softmax(
                model(x, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int)  # , x_hist, x0_hist


@sampling_utils.register_sampler
class ConditionalTauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, conditioner):
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

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(
                model(model_input, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int)


@sampling_utils.register_sampler
class ConditionalPCTauLeaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, conditioner):
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

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(
                model(model_input, min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int)


def cat_logits(cfg, model, x, t_ones, N, D, S, only_logits=False):
    logits = model(x, t_ones)

    if only_logits:
        return logits
    else:
        ll_all, ll_xt = get_logprob_with_logits(
            cfg=cfg, model=model, xt=x, t=t_ones, logits=logits
        )
        return ll_all, ll_xt


def ebm_logits(cfg, model, x, t_ones, N, D, S, only_logits=False):
    device = model.device

    mask = torch.eye(D, device=device, dtype=torch.int32).repeat_interleave(
        N * S, 0
    )  # check
    xrep = torch.tile(x, (D * S, 1))
    candidate = torch.arange(S, device=device).repeat_interleave(N, 0)
    candidate = torch.tile(candidate.unsqueeze(1), ((D, 1)))
    xall = mask * candidate + (1 - mask) * xrep
    t_tile = torch.tile(t_ones, (D * S,))
    qall = model(xall, t_tile)  # can only be CatMLPScoreFunc or BinaryMLPScoreFunc
    logits = torch.reshape(qall, (D, S, N))
    logits = logits.permute(2, 0, 1)

    if only_logits:
        return logits
    else:
        ll_all = F.log_softmax(logits, dim=-1)
        ll_xt = ll_all[
            torch.arange(N, device=device)[:, None],
            torch.arange(D, device=device)[None, :],
            x,
        ]
        return ll_all, ll_xt


def bin_ebm_logits(cfg, model, x, t_ones, B, D, S, only_logits=False):
    device = model.device
    qxt = model(x, t_ones)

    mask = torch.eye(D, device=device).repeat_interleave(B, 0)
    xrep = torch.tile(x, (D, 1))

    xneg = (mask - xrep) * mask + (1 - mask) * xrep
    t = torch.tile(t_ones, (D,))
    qxneg = model(xneg, t)
    qxt = torch.tile(qxt, (D, 1))

    # get_logits
    # qxneg, qxt = self.get_q(params, xt, t)
    qxneg = qxneg.view(-1, B).t()
    qxt = qxt.view(-1, B).t()
    xt_onehot = F.one_hot(x, num_classes=2).to(qxt.dtype)
    qxneg, qxt = qxneg.unsqueeze(-1), qxt.unsqueeze(-1)
    logits = xt_onehot * qxt + (1 - xt_onehot) * qxneg
    print("bin")

    if only_logits:
        return logits
    else:
        ll_all, ll_xt = get_logprob_with_logits(cfg, model, x, t_ones, logits)
        return ll_all, ll_xt


@sampling_utils.register_sampler
class ExactSampling:
    def __init__(self, cfg):
        self.cfg = cfg
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        eps_ratio = cfg.sampler.eps_ratio
        self.initial_dist = cfg.sampler.initial_dist
        self.max_t = cfg.training.max_t

        if cfg.model.log_prob == "bin_ebm":
            self.get_logits = partial(bin_ebm_logits)
        elif cfg.model.log_prob == "ebm":
            self.get_logits = partial(ebm_logits)
        else:  # cfg.model.log_prob == 'cat':
            self.get_logits = partial(cat_logits)

    def sample(self, model, N):
        t = 1.0
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            xt = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            # ts[0] = 0.99999
            # save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]
            change_jump = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                # Entweder in B, D space oder in: hier kann B, D rein, und zwar mit (batch_size, 'ACTG')
                logits = self.get_logits(
                    self.cfg,
                    model,
                    xt,
                    t * torch.ones((N,), device=device),
                    N,
                    self.D,
                    self.S,
                    only_logits=True,
                )
                log_p0t = F.log_softmax(logits, dim=2)  # (N, D, S)

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
                    torch.arange(xt.shape[0], device=device),
                    axis=list(range(1, xt.ndim)),
                )
                q_t_teps = q_t_teps[b, xt.long()].unsqueeze(-2)  # N, D, 1, S

                qt0 = q_teps_0 * q_t_teps
                log_qt0 = torch.log(qt0)
                # log_qt0 = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0))

                log_p0t = log_p0t.unsqueeze(-1)
                log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2).view(-1, self.S)
                cat_dist = torch.distributions.categorical.Categorical(logits=log_prob)
                x_new = cat_dist.sample().view(N, self.D)
                change_jump.append((torch.sum(x_new != xt) / (N * self.D)).item())
                xt = x_new

            # p_0gt = F.softmax(model(xt, self.min_t * torch.ones((N,), device=device)), dim=2)  # (N, D, S)
            # x_0max = torch.max(p_0gt, dim=2)[1]
            x_0max = xt
            return x_0max.detach().cpu().numpy().astype(int), change_jump


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
    off_diag_post = posterior * (1 - xt_onehot)
    off_diag = torch.sum(off_diag_post, axis=-1, keepdims=True)
    diag = torch.clip(1.0 - off_diag, a_min=0)
    posterior = off_diag_post + diag * xt_onehot
    posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
    log_posterior = torch.log(posterior + 1e-35)
    new_y = torch.distributions.categorical.Categorical(log_posterior).sample()
    return new_y


@sampling_utils.register_sampler
class CRMLBJF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps

        if cfg.model.log_prob == "bin_ebm":
            self.get_logprob = partial(bin_ebm_logits)
        elif cfg.model.log_prob == "ebm":
            self.get_logprob = partial(ebm_logits)
        else:  # cfg.model.log_prob == 'cat':
            self.get_logprob = partial(cat_logits)

    def sample(self, model, N):
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
            ts[0] = 0.99999
            change_jump = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]
                # p_theta(x_0|x_t) ?
                t_ones = t * torch.ones((N,), device=device)

                ll_all, ll_xt = self.get_logprob(
                    self.cfg, model, x, t_ones, N, self.D, self.S
                )

                log_weight = ll_all - ll_xt.unsqueeze(-1)  # B, D, S - B, D, 1
                fwd_rate = model.rate_mat(x, t_ones)  # B, D, S?

                xt_onehot = F.one_hot(x, self.S)

                posterior = torch.exp(log_weight) * fwd_rate  # * h  # eq.17 c != x^d_t

                off_diag = torch.sum(
                    posterior * (1 - xt_onehot), axis=-1, keepdims=True
                )
                diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                posterior = posterior * (1 - xt_onehot) * h + diag * xt_onehot  # eq.17

                posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
                log_posterior = torch.log(posterior + 1e-35).view(-1, self.S)
                x_new = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )

                if t <= self.corrector_entry_time:
                    print("corrector")
                    for _ in range(self.num_corrector_steps):
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)
                        ll_all, ll_xt = self.get_logprob(
                            self.cfg, model, x_new, t_ones, N, self.D, self.S
                        )

                        log_weight = ll_all - ll_xt.unsqueeze(-1)
                        fwd_rate = model.rate_mat(x_new, t_ones)

                        xt_onehot = F.one_hot(x_new, self.S)
                        posterior = torch.exp(log_weight) * fwd_rate + fwd_rate
                        off_diag = torch.sum(
                            posterior * (1 - xt_onehot), axis=-1, keepdims=True
                        )
                        diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                        posterior = posterior * (1 - xt_onehot) * h + diag * xt_onehot
                        posterior = posterior / torch.sum(
                            posterior, axis=-1, keepdims=True
                        )
                        # log_posterior = torch.log(posterior + 1e-35)
                        log_posterior = torch.log(posterior + 1e-35).view(-1, self.S)
                        x_new = (
                            torch.distributions.categorical.Categorical(
                                logits=log_posterior
                            )
                            .sample()
                            .view(N, self.D)
                        )
                change_jump.append((torch.sum(x_new != x) / (N * self.D)).item())
                x = x_new
            # p_0gt = F.softmax(model(x, self.min_t * torch.ones((N,), device=device)), dim=2)  # (N, D, S)
            # x_0max = torch.max(p_0gt, dim=2)[1]
            x_0max = x
            return x_0max.detach().cpu().numpy().astype(int), change_jump


@sampling_utils.register_sampler
class CRMTauL:
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
        # self.max_t = cfg.training.max_t

        if cfg.model.log_prob == "bin_ebm":
            self.get_logprob = partial(bin_ebm_logits)
        elif cfg.model.log_prob == "ebm":
            self.get_logprob = partial(ebm_logits)
        else:  # cfg.model.log_prob == 'cat':
            self.get_logprob = partial(cat_logits)

    def sample(self, model, N):
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
            ts[0] = 0.99999
            change_jump = []
            change_clamp = []
            change_clamp2 = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                t_ones = t * torch.ones((N,), device=device)
                # ll_all, ll_xt = self.get_logprob(self.cfg, model, x, t_ones, N, self.D, self.S)
                logits = model(x, t_ones)

                ll_all, ll_xt = get_logprob_with_logits(
                    cfg=self.cfg,
                    model=model,
                    xt=x,
                    t=t_ones,
                    logits=logits,
                )

                log_weight = ll_all - ll_xt.unsqueeze(-1)  # B, D, S - B, D, 1
                fwd_rate = model.rate_mat(x.long(), t_ones)  # B, D, S

                xt_onehot = F.one_hot(x.long(), self.S)
                posterior = torch.exp(log_weight) * fwd_rate
                posterior = posterior * (1 - xt_onehot)  # B, D, S
                post_h = posterior * h
                flips = torch.distributions.poisson.Poisson(
                    post_h
                ).sample()  # B, D, S most 0

                choices = utils.expand_dims(
                    torch.arange(self.S, device=device, dtype=torch.int32),
                    axis=list(range(x.ndim)),
                )  # 1,1, S

                # if t < self.max_t:
                if not self.is_ordinal:
                    jump_num_sum = torch.sum(flips, dim=2)  # B,D
                    changes = torch.sum(((jump_num_sum > 0) * 1).to(dtype=float))
                    # print("1", changes)
                    changes_rej = torch.sum(((jump_num_sum > 1) * 1).to(dtype=float))
                    # change_clamp.
                    change_clamp.append(
                        torch.mean(((jump_num_sum > 1) * 1).to(dtype=float)).item()
                    )
                    # change_clamp2.append((changes_rej / changes).item())
                    tot_flips = torch.sum(flips, axis=-1, keepdims=True)
                    flip_mask = (tot_flips <= 1) * 1  # B, D, 1
                    flips = flips * flip_mask
                else:
                    # Update Rule Categorical
                    # # Regel 1: Wenn in der dritten Dimension mehrere Werte über 1 und einer ist größer als der andere, dann auf 1 und die anderen auf 0 setzen
                    max_values, _ = torch.max(flips, dim=-1, keepdim=True)
                    mask = (flips > 0) & (flips == max_values)
                    flips[mask] = 1
                    flips[~mask] = 0

                    # Regel 2: Wenn in einer Spalte ein Wert über 1, dann diesen zu 1 setzen (für jede Spalte separat)
                    flips[flips > 1] = 0

                    off_diag = torch.sum(
                        posterior * (1 - xt_onehot), axis=-1, keepdims=True
                    )
                    diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                    posterior = posterior * (1 - xt_onehot) * h + diag * xt_onehot
                    posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
                    # Regel 3
                    find = torch.sum(flips, axis=-1)  # B,D with numbers of jumps
                    idx = torch.where(find > 1)
                    if len(idx[0]) > 0:
                        sample_prob = posterior[idx]
                        x_jumps = torch.distributions.categorical.Categorical(
                            sample_prob
                        ).sample()
                        x_jumps = x_jumps.view(len(idx[0]), -1)
                        jumps = torch.zeros((x_jumps.shape[0], self.S), device=device)
                        jumps[
                            torch.arange(x_jumps.shape[0], device=device),
                            x_jumps.flatten(),
                        ] = 1
                        flips[idx] = jumps

                diff = choices - x.unsqueeze(-1)  # B, D, S

                avg_offset = torch.sum(
                    flips * diff, axis=-1
                )  # B, D, S with entries -(S - 1) to S-1
                xp = x + avg_offset

                # change_jump.append((torch.sum(xp != x) / (N * self.D)).item())
                x_new = torch.clip(xp, min=0, max=self.S - 1)
                change_jump.append((torch.sum(x_new != x) / (N * self.D)).item())

                x = x_new

            x_0max = x
            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_clamp,
            )  # , change_clamp2#, change_jump2, change_clamp


@sampling_utils.register_sampler
class CRMMidPointTau:
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
        self.device = cfg.device

        if cfg.data.name == "DiscreteMNIST":
            self.state_change = torch.load(
                "SavedModels/MNIST/state_change_matrix_mnist.pth"
            )
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "Maze3S":
            if self.is_ordinal:
                self.state_change = torch.load(
                    "SavedModels/MAZE/state_change_matrix_maze_ordinal.pth"
                )
                self.state_change = self.state_change.to(device=self.device)
            else:
                self.state_change = torch.load(
                    "SavedModels/MAZE/state_change_matrix_maze.pth"
                )
                self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "BinMNIST":
            self.state_change = torch.tensor([[0, 1], [-1, 0]], device=self.device)

        if cfg.model.log_prob == "bin_ebm":
            self.get_logprob = partial(bin_ebm_logits)
        elif cfg.model.log_prob == "ebm":
            self.get_logprob = partial(ebm_logits)
        else:  # cfg.model.log_prob == 'cat':
            self.get_logprob = partial(cat_logits)

    def sample(self, model, N):
        t = 1.0
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device
        self.state_change = torch.tile(self.state_change, (N, 1, 1))
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(1.0, self.min_t, self.num_steps), np.array([0]))
            )
            ts[0] = 0.99999
            change_jump = []
            change_clamp = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                t_ones = t * torch.ones((N,), device=device)
                t_05 = 0.5 * h + t_ones
                # ll_all, ll_xt = self.get_logprob(self.cfg, model, x, t_ones, N, self.D, self.S)
                logits = model(x, t * torch.ones((N,), device=device))

                ll_all, ll_xt = get_logprob_with_logits(
                    cfg=self.cfg,
                    model=model,
                    xt=x,
                    t=t_05,
                    device=device,
                    logits=logits,
                )

                log_weight = ll_all - ll_xt.unsqueeze(-1)  # B, D, S - B, D, 1
                fwd_rate = model.rate_mat(x.long(), t_ones)  # B, D, S?

                xt_onehot = F.one_hot(x.long(), self.S)
                reverse_rates = torch.exp(log_weight) * fwd_rate
                state_change = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                change = torch.round(
                    torch.sum((0.5 * h * reverse_rates * state_change), dim=-1)
                ).to(dtype=torch.int)
                x_prime = x + change
                x_prime = torch.clip(x_prime, min=0, max=self.S - 1)

                flips = torch.distributions.poisson.Poisson(*h).sample()  # B, D most 0
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
                xp = x + avg_offset

                change_jump.append((torch.sum(xp != x) / (N * self.D)).item())
                x_new = torch.clip(xp, min=0, max=self.S - 1)
                change_clamp.append((torch.sum(xp != x_new) / (N * self.D)).item())
                x = x_new

            return x.detach().cpu().numpy().astype(int), change_jump


@sampling_utils.register_sampler
class CRMBinary:
    def __init__(self, cfg):
        self.cfg = cfg
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps

        if cfg.model.log_prob == "bin_ebm":
            self.get_logprob = partial(bin_ebm_logits)
        elif cfg.model.log_prob == "ebm":
            self.get_logprob = partial(ebm_logits)
        else:  # cfg.model.log_prob == 'cat':
            self.get_logprob = partial(cat_logits)

    def sample(self, model, N):
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
            ts[0] = 0.99999
            change_jump = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]
                # p_theta(x_0|x_t) ?
                t_ones = t * torch.ones((N,), device=device)
                qxt = model(x, t_ones)

                mask = torch.eye(self.D, device=device).repeat_interleave(N, 0)
                xrep = torch.tile(x, (self.D, 1))

                xneg = (mask - xrep) * mask + (1 - mask) * xrep
                t = torch.tile(t_ones, (self.D,))
                qxneg = model(xneg, t)
                qxt = torch.tile(qxt, (self.D, 1))
                ratio = torch.exp(qxneg - qxt)
                ratio = ratio.reshape(-1, N).t()

                cur_rate = model.rate_const * ratio
                nu_x = torch.sigmoid(cur_rate)
                flip_rate = nu_x * torch.exp(utils.log1mexp(-h * cur_rate / nu_x))
                flip = torch.bernoulli(flip_rate)
                x_new = (1 - x) * flip + x * (1 - flip)
                change_jump.append((torch.sum(x_new != x) / (N * self.D)).item())
                x = x_new

        return x.detach().cpu().numpy().astype(int), change_jump


"""
 def adaptive_tau(self, x, rate, h, threshold=10):
        # Berechnen der erwarteten Anzahl von Ereignissen
        # Angenommen, dass 'rate' die Raten für die entsprechenden Zustandsübergänge enthält
        expected_events = rate.sum(dim=2) * h  # Summiert über alle Zustände

        # Anpassen von h basierend auf der erwarteten Anzahl von Ereignissen
        max_events = expected_events.max()
        if max_events > threshold:
            # Verringern Sie h, wenn die Anzahl der erwarteten Ereignisse zu hoch ist
            h = h * threshold / max_events

        return h
"""


@sampling_utils.register_sampler
class CTMidPointTauL:
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
        self.device = cfg.device
        self.eps_ratio = cfg.sampler.eps_ratio

        if cfg.data.name == "DiscreteMNIST":
            self.state_change = -torch.load(
                "SavedModels/MNIST/state_change_matrix_mnist.pth"
            )
            self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "Maze3S":
            if self.is_ordinal:
                self.state_change = -torch.load(
                    "SavedModels/MAZE/state_change_matrix_maze_ordinal.pth"
                )
                self.state_change = self.state_change.to(device=self.device)
            else:
                self.state_change = -torch.load(
                    "SavedModels/MAZE/state_change_matrix_maze.pth"
                )
                self.state_change = self.state_change.to(device=self.device)
        elif cfg.data.name == "BinMNIST":
            self.state_change = -torch.tensor([[0, 1], [-1, 0]], device=self.device)

    # SxS
    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device
        self.state_change = torch.tile(self.state_change, (N, 1, 1))
        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(1.0, self.min_t, self.num_steps), np.array([0]))
            )
            ts[0] = 0.99999
            change_jump = []
            change_clamp = []
            t = 1.0
            # Fragen:
            # 1. Prediction zum  Zeitpunkt 0.5 * h +t_ones?
            # Wie summe über states? => meistens R * changes = 0
            #
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]
                t_ones = t * torch.ones((N,), device=device)  # (N, S, S)
                t_05 = 0.5 * h + t_ones

                qt0 = model.transition(t_05)  # (N, S, S)
                rate = model.rate(t_05)  # (N, S, S)

                logits = model(x, t_05)
                p0t = F.softmax(logits, dim=2)
                # reverse_rates, _ = get_reverse_rates(model, logits, x, t_05, self.cfg, N, self.D, self.S)
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

                reverse_rates = forward_rates * inner_sum

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x.long().flatten(),
                ] = 0.0

                state_change = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)
                # B, D, S * B,D,S
                # B, D
                # x' = x + sum
                change = torch.round(
                    0.5 * h * torch.sum((reverse_rates * state_change), dim=-1)
                ).to(dtype=torch.int)

                x_prime = x + change  # , dim=-1)
                x_prime = torch.clip(x_prime, min=0, max=self.S - 1)

                # ------------second-------------------
                logits_prime = model(x_prime, t_ones)
                p0t_prime = F.softmax(logits_prime, dim=2)

                qt0_prime = model.transition(t_ones)
                rate_prime = model.rate(t_ones)
                qt0_numer_prime = qt0_prime  #
                # reverse_rates, _ = get_reverse_rates(model, logits, x, t_05, self.cfg, N, self.D, self.S)
                qt0_denom_prime = (
                    qt0_prime[
                        torch.arange(N, device=device).repeat_interleave(
                            self.D * self.S
                        ),
                        torch.arange(self.S, device=device).repeat(N * self.D),
                        x_prime.long().flatten().repeat_interleave(self.S),
                    ].view(N, self.D, self.S)
                    + self.eps_ratio
                )

                # First S is x0 second S is x tilde (N, S, S)

                forward_rates_prime = rate_prime[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x_prime.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                inner_sum_prime = (
                    p0t_prime / qt0_denom_prime
                ) @ qt0_numer_prime  # (N, D, S)

                reverse_rates_prime = forward_rates_prime * inner_sum_prime

                reverse_rates_prime[
                    torch.arange(N, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(N),
                    x_prime.long().flatten(),
                ] = 0.0
                # reverse_rates, _ = get_reverse_rates(model, logits, x_prime, t_ones, self.cfg, N, self.D, self.S)

                state_change_prime = self.state_change[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x_prime.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                diffs = torch.arange(self.S, device=device).view(
                    1, 1, self.S
                ) - x_prime.view(N, self.D, 1)
                diff_prime = state_change_prime

                flips = torch.distributions.poisson.Poisson(
                    reverse_rates_prime * h
                ).sample()  # B, D most 0

                if not self.is_ordinal:
                    tot_flips = torch.sum(flips, axis=-1, keepdims=True)
                    flip_mask = (tot_flips <= 1) * 1
                    flips = flips * flip_mask
                # diff = choices - x.unsqueeze(-1)

                avg_offset = torch.sum(
                    flips * diff_prime, axis=-1
                )  # B, D, S with entries -(S - 1) to S-1
                xp = x_prime + avg_offset

                change_jump.append((torch.sum(xp != x_prime) / (N * self.D)).item())
                x_new = torch.clip(xp, min=0, max=self.S - 1)
                change_clamp.append((torch.sum(xp != x_new) / (N * self.D)).item())
                x = x_new

            p_0gt = F.softmax(
                model(x, self.min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x = torch.max(p_0gt, dim=2)[1]
            # x_0max = x

            return x.detach().cpu().numpy().astype(int), change_jump


@sampling_utils.register_sampler
class ExactELBO:
    def __init__(self, cfg):
        self.cfg = cfg
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        eps_ratio = cfg.sampler.eps_ratio
        self.initial_dist = cfg.sampler.initial_dist
        self.max_t = cfg.training.max_t

    def sample(self, model, N):
        # t = 1.0
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            xt = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )
            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            # save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]
            change_jump = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                # Entweder in B, D space oder in: hier kann B, D rein, und zwar mit (batch_size, 'ACTG')
                logits = model(xt, t * torch.ones((N,), device=device))
                p0t = F.softmax(logits, dim=2)
                # log_p0t = F.log_softmax(logits, dim=2)  # (N, D, S)

                t_eps = t - h  # tau

                q_teps_0 = model.transition(
                    t_eps * torch.ones((N,), device=device)
                )  # (N, S, S)
                b = utils.expand_dims(
                    torch.arange(xt.shape[0], device=device),
                    axis=list(range(1, xt.ndim)),
                )
                q_teps_0_denom = q_teps_0[b, xt.long()]  # B, D, S

                # q_teps_0 (N, 1, S, S)
                # q_teps_0 = utils.expand_dims(q_teps_0, axis=list(range(1, xt.ndim)))

                q_t0 = model.transition(t * torch.ones((N,), device=device))
                q_t0_denom = q_t0[b, xt.long()]

                # q_t_denom = utils.expand_dims(q_t_denom, axis=list(range(1, xt.ndim)))

                q_t_teps = model.transit_between(
                    t_eps * torch.ones((N,), device=device),
                    t * torch.ones((N,), device=device),
                )  # (N, S, S)
                q_t_teps = q_t_teps.permute(0, 2, 1)
                q_t_teps_denom = q_t_teps.clone()

                b = utils.expand_dims(
                    torch.arange(xt.shape[0], device=device),
                    axis=list(range(1, xt.ndim)),
                )
                # print("qtteps", q_t_teps.shape)
                q_t_teps_denom = q_t_teps_denom[b, xt.long()].unsqueeze(-2)

                q_new1 = p0t / q_t_teps_denom / q_teps_0_denom
                q_new = q_new1 @ q_t_teps.transpose(1, 2) @ q_teps_0

                # qt0 = (q_t_denom * q_t_teps_denom)

                # log_qt0 = torch.log(qt0)
                # log_qt0 = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0)) # N, D, S, S

                # log_p0t = log_p0t.unsqueeze(-1)
                # log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2).view(-1, self.S) # B*D, S
                # print(log_prob.shape)
                log_prob = torch.log(q_new)
                # print(log_prob.shape)
                cat_dist = torch.distributions.categorical.Categorical(logits=log_prob)
                x_new = cat_dist.sample().view(N, self.D)
                change_jump.append((torch.sum(x_new != xt) / (N * self.D)).item())
                xt = x_new

            return xt.detach().cpu().numpy().astype(int), change_jump


@sampling_utils.register_sampler
class ElboTauL:
    def __init__(self, cfg):
        self.cfg = cfg
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = 0  # cfg.sampler.eps_ratio
        self.is_ordinal = cfg.sampler.is_ordinal
        self.max_t = cfg.training.max_t

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(
                N, self.D, device, self.S, self.initial_dist, initial_dist_std
            )

            # tau = 1 / num_steps
            ts = np.concatenate(
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            # ts[0] = 0.99999
            change_jump = []
            change_clamp = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                qt0 = model.transition(t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device))  # (N, S, S)
                # p_theta(x_0|x_t) ?

                p0t = F.softmax(
                    model(x, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S) (not log_softmax)

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
                )

                poisson_dist = torch.distributions.poisson.Poisson(
                    reverse_rates * h
                )  # posterior: p_{t-eps|t}, B, D; S
                jump_nums = (
                    poisson_dist.sample()
                )  # how many jumps in interval [t-eps, t]

                if not self.is_ordinal:
                    """
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    print(torch.mean(jump_num_sum * 1))
                    jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)
                    """
                    jump_num_sum = torch.sum(jump_nums, axis=-1, keepdims=True)
                    flip_mask = (jump_num_sum <= 1) * 1
                    jump_nums = jump_nums * flip_mask
                    """

                    jump_nums = torch.clamp(jump_nums, max=1)
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    print(torch.mean(jump_num_sum * 1).item())
                    #print(jump_nums)
                    jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)
                    """

                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump

                change_jump.append((torch.sum(xp != x) / (N * self.D)).item())

                x_new = torch.clamp(xp, min=0, max=self.S - 1)

                change_clamp.append(torch.mean(jump_num_sum * 1).item())
                x = x_new
                # if t < 0.01:
                #    break

            p_0gt = F.softmax(
                model(x, self.min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]

            # x_0max = x
            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_clamp,
            )  # , x_hist, x0_hist


@sampling_utils.register_sampler
class ElboLBJF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio
        self.max_t = cfg.training.max_t

    def sample(self, model, N):
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
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            change_jump = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]
                # print(t)
                qt0 = model.transition(t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device))  # (N, S, S)
                # p_theta(x_0|x_t) ?

                p0t = F.softmax(
                    model(x, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S) (not log_softmax)

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

                qt0_numer = qt0  # (N, S, S)
                # forward_rates == fwd_rate
                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S) #

                xt_onehot = F.one_hot(x.long(), self.S)

                posterior = forward_rates * inner_sum  # (N, D, S)
                # post_0 = posterior * (1 - xt_onehot)

                off_diag = torch.sum(
                    posterior * (1 - xt_onehot), axis=-1, keepdims=True
                )
                diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                posterior = (
                    posterior * (1 - xt_onehot) * h + diag * xt_onehot
                )  # * h  # eq.17

                posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
                log_posterior = torch.log(posterior + 1e-35).view(-1, self.S)
                x_new = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )

                if t <= self.corrector_entry_time:
                    print("corrector")
                    for _ in range(self.num_corrector_steps):
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)
                        p0t = F.softmax(
                            model(x_new, t * torch.ones((N,), device=device)), dim=2
                        )  #

                        qt0_denom = (
                            qt0[
                                torch.arange(N, device=device).repeat_interleave(
                                    self.D * self.S
                                ),
                                torch.arange(self.S, device=device).repeat(N * self.D),
                                x_new.long().flatten().repeat_interleave(self.S),
                            ].view(N, self.D, self.S)
                            + self.eps_ratio
                        )

                        qt0_numer = qt0  # (N, S, S)
                        # forward_rates == fwd_rate
                        forward_rates = rate[
                            torch.arange(N, device=device).repeat_interleave(
                                self.D * self.S
                            ),
                            torch.arange(self.S, device=device).repeat(N * self.D),
                            x_new.long().flatten().repeat_interleave(self.S),
                        ].view(N, self.D, self.S)

                        inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S) #

                        xt_onehot = F.one_hot(x_new.long(), self.S)

                        posterior = (
                            forward_rates + forward_rates * inner_sum
                        )  # (N, D, S)
                        post_0 = posterior * (1 - xt_onehot)

                        off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                        diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                        posterior = post_0 * h + diag * xt_onehot
                        posterior = posterior / torch.sum(
                            posterior, axis=-1, keepdims=True
                        )
                        log_posterior = torch.log(posterior + 1e-35).view(-1, self.S)

                        x_new = (
                            torch.distributions.categorical.Categorical(
                                logits=log_posterior
                            )
                            .sample()
                            .view(N, self.D)
                        )
                change_jump.append((torch.sum(x_new != x) / (N * self.D)).item())
                # print(torch.sum(x_new != x, dim=1))
                x = x_new
            # p_0gt = F.softmax(model(x, self.min_t * torch.ones((N,), device=device)), dim=2)  # (N, D, S)
            # x_0max = torch.max(p_0gt, dim=2)[1]
            x_0max = x
            # p_0gt = F.softmax(model(x_0max, self.min_t * torch.ones((N,), device=device)), dim=2)  # (N, D, S)
            # x_0max = torch.max(p_0gt, dim=2)[1]
            # x_0max = x
            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_jump,
            )  # , x_hist, x0_hist


@sampling_utils.register_sampler
class ElboLBJF2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
        # C, H, W = self.cfg.data.shape
        self.D = cfg.model.concat_dim
        self.S = self.cfg.data.S
        self.num_steps = cfg.sampler.num_steps
        self.min_t = cfg.sampler.min_t
        self.initial_dist = cfg.sampler.initial_dist
        self.corrector_entry_time = cfg.sampler.corrector_entry_time
        self.num_corrector_steps = cfg.sampler.num_corrector_steps
        self.eps_ratio = cfg.sampler.eps_ratio

    def sample(self, model, N):
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
                (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
            )
            change_jump = []
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx + 1]

                qt0 = model.transition(t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device))  # (N, S, S)
                # p_theta(x_0|x_t) ?

                p0t = F.softmax(
                    model(x, t * torch.ones((N,), device=device)), dim=2
                )  # (N, D, S) (not log_softmax)

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

                qt0_numer = qt0  # (N, S, S)
                # forward_rates == fwd_rate
                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(self.D * self.S),
                    torch.arange(self.S, device=device).repeat(N * self.D),
                    x.long().flatten().repeat_interleave(self.S),
                ].view(N, self.D, self.S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S) #

                xt_onehot = F.one_hot(x.long(), self.S)

                posterior = forward_rates * inner_sum  # (N, D, S)
                post_0 = posterior * (1 - xt_onehot)

                off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                posterior = post_0 * h + diag * xt_onehot  # h  # eq.17

                posterior = posterior / torch.sum(posterior, axis=-1, keepdims=True)
                log_posterior = torch.log(posterior + 1e-35).view(-1, self.S)
                x_new = (
                    torch.distributions.categorical.Categorical(logits=log_posterior)
                    .sample()
                    .view(N, self.D)
                )
                x = x_new
                if t <= self.corrector_entry_time:
                    print("corrector")
                    for _ in range(self.num_corrector_steps):
                        # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)
                        p0t = F.softmax(
                            model(x_new, t * torch.ones((N,), device=device)), dim=2
                        )  #

                        qt0_denom = (
                            qt0[
                                torch.arange(N, device=device).repeat_interleave(
                                    self.D * self.S
                                ),
                                torch.arange(self.S, device=device).repeat(N * self.D),
                                x_new.long().flatten().repeat_interleave(self.S),
                            ].view(N, self.D, self.S)
                            + self.eps_ratio
                        )

                        qt0_numer = qt0  # (N, S, S)
                        # forward_rates == fwd_rate
                        forward_rates = rate[
                            torch.arange(N, device=device).repeat_interleave(
                                self.D * self.S
                            ),
                            torch.arange(self.S, device=device).repeat(N * self.D),
                            x_new.long().flatten().repeat_interleave(self.S),
                        ].view(N, self.D, self.S)

                        inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S) #

                        xt_onehot = F.one_hot(x_new.long(), self.S)

                        posterior = (
                            forward_rates + forward_rates * inner_sum
                        )  # (N, D, S)
                        post_0 = posterior * (1 - xt_onehot)

                        off_diag = torch.sum(post_0, axis=-1, keepdims=True)
                        diag = torch.clip(1.0 - h * off_diag, min=0, max=float("inf"))
                        posterior = post_0 * h**2 + diag * xt_onehot
                        posterior = posterior / torch.sum(
                            posterior, axis=-1, keepdims=True
                        )
                        log_posterior = torch.log(posterior + 1e-35).view(-1, self.S)

                        x_new = (
                            torch.distributions.categorical.Categorical(
                                logits=log_posterior
                            )
                            .sample()
                            .view(N, self.D)
                        )
                change_jump.append((torch.sum(x_new != x) / (N * self.D)).item())
                # print(torch.sum(x_new != x, dim=1))
                x = x_new
            p_0gt = F.softmax(
                model(x, self.min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            # x_0max = x
            return (
                x_0max.detach().cpu().numpy().astype(int),
                change_jump,
            )  # , x_hist, x0_hist


@sampling_utils.register_sampler
class TAULStepSize:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_t = cfg.training.max_t
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
        self.loss_name = cfg.loss.name

    def sample(self, model, N):
        initial_dist_std = self.cfg.model.Q_sigma
        device = model.device

        x = get_initial_samples(
            N, self.D, device, self.S, self.initial_dist, initial_dist_std
        )
        # tau = 1 / num_steps
        ts = np.concatenate(
            (np.linspace(self.max_t, self.min_t, self.num_steps), np.array([0]))
        )
        change_jump = []
        change_clamp = []
        change_clip = []
        # new_tensor = torch.empty((self.num_steps, N, self.D), device=device)

        for idx, t in tqdm(enumerate(ts[0:-1])):
            # new_tensor[idx] = x
            h = ts[idx] - ts[idx + 1]

            x = x.to(dtype=torch.float32).requires_grad_(True)
            t_ones = t * torch.ones((N,), device=device, dtype=x.dtype)
            logits = model(x, t_ones)
            # logits = logits.requires_grad_(True)
            # logits= logits.flatten()
            # for i, out in enumerate(logits):
            #    print(i, torch.autograd.grad(outputs=out, inputs=x, retain_graph=True))
            # print([torch.autograd.grad(outputs=out, inputs=x, retain_graph=True) for i, out in enumerate(logits)])
            # print([torch.autograd.grad(outputs=out, inputs=x, retain_graph=True)[0][i] for i, out in enumerate(logits)])

            x = x.to(dtype=float).requires_grad_(True)
            reverse_rates, _ = get_reverse_rates(
                model, logits, x, t_ones, self.cfg, N, self.D, self.S
            )
            reverse_rates = reverse_rates.flatten()
            for i, out in enumerate(reverse_rates):
                print(i, torch.autograd.grad(outputs=out, inputs=x, retain_graph=True))

            xt_onehot = F.one_hot(x.long(), self.S)
            reverse_rates = reverse_rates * (1 - xt_onehot)
            grads = torch.empty((N, self.D, self.S), device=device)
            grad2 = torch.empty((N, self.D, self.S), device=device)
            torch.sum(reverse_rates).backward()
            grad_x = x.grad
            print(grad_x)

            """
            for i in range(N):
                for j in range(self.D):
                    for k in range(self.S):
                        #q = torch.tensor(reverse_rates[i, j, k], device=device, requires_grad=True)
                        reverse_rates[i, j, k].backward()
                        grad_r_x = x.grad
                        print(grad_r_x)

                        if grad_r_x is not None:
                            print("shaoe", grad_r_x.shape)
                            grad2[:, :, k] = grad_r_x
            """
            """
            rev_sum = torch.sum(reverse_rates, dim=-1)
            for i in range(N):
                for j in range(self.D):
                        q = torch.tensor(rev_sum[i, j], device=device, requires_grad=True)
                        q.backward()
                        grad_r_x = x.grad
                        print(grad_r_x)
            """
            x = x.to(dtype=int).requires_grad_(False)
            # print("grad", grads)

            poisson_dist = torch.distributions.poisson.Poisson(
                reverse_rates * h
            )  # posterior: p_{t-eps|t}, B, D; S
            jump_nums = poisson_dist.sample()  # how many jumps in interval [t-eps, t]

            if not self.is_ordinal:
                jump_num_sum = torch.sum(jump_nums, dim=2)
                jump_num_sum_mask = jump_num_sum <= 1
                jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)
            else:
                # proportion of jumps
                jump_num_sum = torch.sum(jump_nums, dim=2)
                change_clamp.append(
                    torch.mean(((jump_num_sum > 1) * 1).to(dtype=float)).item()
                )

            choices = utils.expand_dims(
                torch.arange(self.S, device=device, dtype=torch.int32),
                axis=list(range(x.ndim)),
            )
            diff = choices - x.unsqueeze(-1)
            adj_diffs = jump_nums * diff
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = x + overall_jump

            change_jump.append((torch.sum(xp != x) / (N * self.D)).item())

            x_new = torch.clamp(xp, min=0, max=self.S - 1)
            change_clip.append((torch.sum(x_new != x) / (N * self.D)).item())

            x = x_new
            if t <= self.corrector_entry_time:
                print("corrector")
                for _ in range(self.num_corrector_steps):
                    # x = lbjf_corrector_step(self.cfg, model, x, t, h, N, device, xt_target=None)

                    t_h_ones = (t) * torch.ones((N,), device=device)
                    rate = model.rate(t_h_ones)

                    logits = model(x_new, t_h_ones)  #
                    reverse_rates, _ = get_reverse_rates(
                        model, logits, x_new, t_h_ones, self.cfg, N, self.D, self.S
                    )
                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(self.D),
                        torch.arange(self.D, device=device).repeat(N),
                        x_new.long().flatten(),
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(
                            self.D * self.S
                        ),
                        x_new.long().flatten().repeat_interleave(self.S),
                        torch.arange(self.S, device=device).repeat(N * self.D),
                    ].view(N, self.D, self.S)

                    corrector_rates = (
                        transpose_forward_rates + reverse_rates
                    )  # (N, D, S)
                    corrector_rates[
                        torch.arange(N, device=device).repeat_interleave(self.D),
                        torch.arange(self.D, device=device).repeat(N),
                        x_new.long().flatten(),
                    ] = 0.0
                    poisson_dist = torch.distributions.poisson.Poisson(
                        corrector_rates * h
                    )  # posterior: p_{t-eps|t}, B, D; S

                    jump_nums = (
                        poisson_dist.sample()
                    )  # how many jumps in interval [t-eps, t]

                    if not self.is_ordinal:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        jump_nums = jump_nums * jump_num_sum_mask.view(N, self.D, 1)
                    choices = utils.expand_dims(
                        torch.arange(self.S, device=device, dtype=torch.int32),
                        axis=list(range(x_new.ndim)),
                    )
                    diff = choices - x_new.unsqueeze(-1)
                    adj_diffs = jump_nums * diff
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = x + overall_jump

                    change_jump.append((torch.sum(xp != x) / (N * self.D)).item())

                    x_new = torch.clamp(xp, min=0, max=self.S - 1)
                    x = x_new
        if self.loss_name == "CTElbo":
            p_0gt = F.softmax(
                model(x, self.min_t * torch.ones((N,), device=device)), dim=2
            )  # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
        else:
            x_0max = x

        return (
            x_0max.detach().cpu().numpy().astype(int),  # change_jump,
            change_jump,
        )  # , x_hist, x0_hist
