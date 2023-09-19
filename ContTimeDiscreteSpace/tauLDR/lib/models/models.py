from tkinter import Image
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lib.models.model_utils as model_utils
import lib.networks.networks as networks
import lib.networks.networks_paul as networks_paul
from torchtyping import patch_typeguard, TensorType
import torch.autograd.profiler as profiler
import math
from torch.nn.parallel import DistributedDataParallel as DDP

class ImageX0PredBasePaul(nn.Module):
    def __init__(self, cfg, device, use_net: bool = True, rank=None):
        super().__init__()

        self.fix_logistic = cfg.model.fix_logistic
        ch = cfg.model.ch
        input_channels = cfg.model.input_channels
        output_channels = cfg.model.input_channels  # * cfg.data.S
        data_min_max = cfg.model.data_min_max

        self.S = cfg.data.S
        self.data_shape = cfg.data.shape
        if use_net:
            self.net = networks_paul.MiniUNetDiscrete(
                dim=ch,
                in_channel=input_channels,
                out_channel=output_channels,
                model_output="logistic_pars",
                num_classes=self.S,
                x_min_max=data_min_max,
            ).to(device)
        else:
            self.net = networks_paul.UNet(
                in_channel=input_channels,
                out_channel=output_channels,
                channel=ch,
                channel_multiplier=cfg.model.ch_mult,
                n_res_blocks=cfg.model.num_res_blocks,
                attn_resolutions=[16],
                num_heads=1,
                dropout=cfg.model.dropout,
                model_output= 'logits', #"logistic_pars",  # 'logits' or 'logistic_pars'
                num_classes=self.S,
                x_min_max=data_min_max,
                img_size=self.data_shape[2],
            ).to(device)

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space for each pixel
        """
        B, D = x.shape
        C, H, W = self.data_shape
        S = self.S
        x = x.view(B, C, H, W)

        # Output: 3 × 32 × 32 × 2 => mean and log scale of a logistic distribution
        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf
        # wenden tanh auf beides an, d3pm nur auf mu
        net_out = self.net(x, times)  # (B, 2*C, H, W)
        mu = net_out[0].unsqueeze(-1)
        log_scale = net_out[1].unsqueeze(-1)

        # The probability for a state is then the integral of this continuous distribution between
        # this state and the next when mapped onto the real line. To impart a residual inductive bias
        # on the output, the mean of the logistic distribution is taken to be tanh(xt + μ′) where xt
        # is the normalized input into the model and μ′ is mean outputted from the network.
        # The normalization operation takes the input in the range 0, . . . , 255 and maps it to [−1, 1].
        inv_scale = torch.exp(-(log_scale - 2))

        bin_width = 2.0 / self.S
        bin_centers = torch.linspace(
            start=-1.0 + bin_width / 2,
            end=1.0 - bin_width / 2,
            steps=self.S,
            device=self.device,
        ).view(1, 1, 1, 1, self.S)

        sig_in_left = (bin_centers - bin_width / 2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width / 2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(
            -sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf
        )
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1

        logits = logits.view(B, D, S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
        Compute log (exp(a) - exp(b)) for (b<a)
        From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b - a) + eps)


class ImageX0PredBase(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        self.fix_logistic = cfg.model.fix_logistic
        ch = cfg.model.ch
        num_res_blocks = cfg.model.num_res_blocks
        num_scales = cfg.model.num_scales
        ch_mult = cfg.model.ch_mult
        input_channels = cfg.model.input_channels
        output_channels = cfg.model.input_channels * cfg.data.S
        scale_count_to_put_attn = cfg.model.scale_count_to_put_attn
        data_min_max = cfg.model.data_min_max
        dropout = cfg.model.dropout
        skip_rescale = cfg.model.skip_rescale
        do_time_embed = True
        time_scale_factor = cfg.model.time_scale_factor
        time_embed_dim = cfg.model.time_embed_dim

        tmp_net = networks.UNet(
            ch,
            num_res_blocks,
            num_scales,
            ch_mult,
            input_channels,
            output_channels,
            scale_count_to_put_attn,
            data_min_max,
            dropout,
            skip_rescale,
            do_time_embed,
            time_scale_factor,
            time_embed_dim,
        ).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.S = cfg.data.S
        self.data_shape = cfg.data.shape

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space for each pixel
        """
        B, D = x.shape
        C, H, W = self.data_shape
        S = self.S
        x = x.view(B, C, H, W)

        # Output: 3 × 32 × 32 × 2 => mean and log scale of a logistic distribution
        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf
        # wenden tanh auf beides an, d3pm nur auf mu
        net_out = self.net(x, times)  # (B, 2*C, H, W)
        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)

        # The probability for a state is then the integral of this continuous distribution between
        # this state and the next when mapped onto the real line. To impart a residual inductive bias
        # on the output, the mean of the logistic distribution is taken to be tanh(xt + μ′) where xt
        # is the normalized input into the model and μ′ is mean outputted from the network.
        # The normalization operation takes the input in the range 0, . . . , 255 and maps it to [−1, 1].
        inv_scale = torch.exp(-(log_scale - 2))

        bin_width = 2.0 / self.S
        bin_centers = torch.linspace(
            start=-1.0 + bin_width / 2,
            end=1.0 - bin_width / 2,
            steps=self.S,
            device=self.device,
        ).view(1, 1, 1, 1, self.S)

        sig_in_left = (bin_centers - bin_width / 2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width / 2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(
            -sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf
        )
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1

        logits = logits.view(B, D, S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
        Compute log (exp(a) - exp(b)) for (b<a)
        From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b - a) + eps)


class BirthDeathForwardBase:
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.sigma_min, self.sigma_max = cfg.model.sigma_min, cfg.model.sigma_max
        self.device = device
        # base rate is a (S-1, S-1) Matrix with ones on first off diagonals and - the sum of these entries on diagonal
        base_rate = np.diag(np.ones((S - 1,)), 1)
        base_rate += np.diag(np.ones((S - 1,)), -1)
        base_rate -= np.diag(np.sum(base_rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(base_rate)

        self.base_rate = torch.from_numpy(base_rate).float().to(self.device)
        self.base_eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.base_eigvecs = torch.from_numpy(eigvecs).float().to(self.device)

    # R_t = beta_t * R_b
    # R_b = self.base_rate
    def _rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        # time dependent scalar beta_(t)
        # choice: beta_t = a * (b ** t) * logb
        return (
            self.sigma_min**2
            * (self.sigma_max / self.sigma_min) ** (2 * t)
            * math.log(self.sigma_max / self.sigma_min)
        )

    def _integral_rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        # integral of beta_t over 0 to t: a * b ** t - a
        #
        return (
            0.5 * self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t)
            - 0.5 * self.sigma_min**2
        )

    def rate(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        # R_t = beta_t * R_b
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.base_eigvals.view(1, S)
        # P_t = Q * exp(lambda * integrate_scalar) * Q^-1
        transitions = (
            self.base_eigvecs.view(1, S, S)
            @ torch.diag_embed(torch.exp(adj_eigvals))
            @ self.base_eigvecs.T.view(1, S, S)
        )

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] BirthDeathForwardBase, large negative transition values {torch.min(transitions)}"
            )

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions # [B, S, S] 


class UniformRate:
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_const = cfg.model.rate_const
        self.device = device

        rate = self.rate_const * np.ones((S, S))
        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(rate)

        self.rate_matrix = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        # above same as get_rate_matrix from ForwardModel

    # rate_mat
    def rate(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S

        return torch.tile(self.rate_matrix.view(1, S, S), (B, 1, 1))
    # func usvt from forward model
    def transition(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S
        transitions = (
            self.eigvecs.view(1, S, S)
            @ torch.diag_embed(torch.exp(self.eigvals.view(1, S) * t.view(B, 1)))
            @ self.eigvecs.T.view(1, S, S)
        )

        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}"
            )

        transitions[transitions < 1e-8] = 0.0

        return transitions
    
    def transit_between(self, t1, t2):
        return self.transition(t2 - t1)


class GaussianTargetRate:
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_sigma = cfg.model.rate_sigma
        self.Q_sigma = cfg.model.Q_sigma
        self.time_exponential = cfg.model.time_exponential
        self.time_base = cfg.model.time_base
        self.device = device

        rate = np.zeros((S, S))

        vals = np.exp(-np.arange(0, S) ** 2 / (self.rate_sigma**2))
        for i in range(S):
            for j in range(S):
                if i < S // 2:
                    if j > i and j < S - i:
                        rate[i, j] = vals[j - i - 1]
                elif i > S // 2:
                    if j < i and j > -i + S - 1:
                        rate[i, j] = vals[i - j - 1]
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * np.exp(
                        -((j + 1) ** 2 - (i + 1) ** 2 + S * (i + 1) - S * (j + 1))
                        / (2 * self.Q_sigma**2)
                    )

        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))

        eigvals, eigvecs = np.linalg.eig(rate)
        inv_eigvecs = np.linalg.inv(eigvecs)

        self.base_rate = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float().to(self.device)

    def _integral_rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        return self.time_base * (self.time_exponential**t) - self.time_base

    def _rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        return (
            self.time_base
            * math.log(self.time_exponential)
            * (self.time_exponential**t)
        )

    def rate(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)

        transitions = (
            self.eigvecs.view(1, S, S)
            @ torch.diag_embed(torch.exp(adj_eigvals))
            @ self.inv_eigvecs.view(1, S, S)
        )

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] GaussianTargetRate, large negative transition values {torch.min(transitions)}"
            )

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions
    
    def transit_between(self, t1, t2):
        return self.transition(t2 - t1)


class SequenceTransformer(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        num_layers = cfg.model.num_layers
        d_model = cfg.model.d_model
        num_heads = cfg.model.num_heads
        dim_feedforward = cfg.model.dim_feedforward
        dropout = cfg.model.dropout
        num_output_FFresiduals = cfg.model.num_output_FFresiduals
        time_scale_factor = cfg.model.time_scale_factor
        temb_dim = cfg.model.temb_dim
        use_one_hot_input = cfg.model.use_one_hot_input
        self.S = cfg.data.S

        assert len(cfg.data.shape) == 1
        max_len = cfg.data.shape[0]

        tmp_net = networks.TransformerEncoder(
            num_layers,
            d_model,
            num_heads,
            dim_feedforward,
            dropout,
            num_output_FFresiduals,
            time_scale_factor,
            self.S,
            max_len,
            temb_dim,
            use_one_hot_input,
            device,
        ).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.data_shape = cfg.data.shape

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """
        B, D = x.shape
        S = self.S

        logits = self.net(x.long(), times.long())  # (B, D, S)

        return logits


class ResidualMLP(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        self.S = cfg.data.S
        num_layers = cfg.model.num_layers
        d_model = cfg.model.d_model  # int
        hidden_dim = cfg.model.hidden_dim
        time_scale_factor = cfg.model.time_scale_factor
        temb_dim = cfg.model.temb_dim

        assert len(cfg.data.shape) == 1
        D = cfg.data.shape[0]

        tmp_net = networks.ResidualMLP(
            num_layers, d_model, hidden_dim, D, self.S, time_scale_factor, temb_dim
        ).to(device)

        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.data_shape = cfg.data.shape

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """

        logits = self.net(x, times)  # (B, D, S)

        return logits

def HTransformer():
    pass
    # self.net = bidirectional transformer


# Based on https://github.com/yang-song/score_sde_pytorch/blob/ef5cb679a4897a40d20e94d8d0e2124c3a48fb8c/models/ema.py
class EMA:
    def __init__(self, cfg):
        self.decay = cfg.model.ema_decay
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.shadow_params = []
        self.collected_params = []
        self.num_updates = 0

    def init_ema(self):
        self.shadow_params = [
            p.clone().detach() for p in self.parameters() if p.requires_grad
        ]

    def update_ema(self):
        if len(self.shadow_params) == 0:
            raise ValueError("Shadow params not initialized before first ema update!")

        decay = self.decay
        self.num_updates += 1
        decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in self.parameters() if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def state_dict(self):
        sd = nn.Module.state_dict(self)
        sd["ema_decay"] = self.decay
        sd["ema_num_updates"] = self.num_updates
        sd["ema_shadow_params"] = self.shadow_params

        return sd

    def move_shadow_params_to_model_params(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def move_model_params_to_collected_params(self):
        self.collected_params = [param.clone() for param in self.parameters()]

    def move_collected_params_to_model_params(self):
        for c_param, param in zip(self.collected_params, self.parameters()):
            param.data.copy_(c_param.data)

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = nn.Module.load_state_dict(
            self, state_dict, strict=False
        )

        # print("state dict keys")
        # for key in state_dict.keys():
        #     print(key)

        if len(missing_keys) > 0:
            print("Missing keys: ", missing_keys)
            raise ValueError
        if not (
            len(unexpected_keys) == 3
            and "ema_decay" in unexpected_keys
            and "ema_num_updates" in unexpected_keys
            and "ema_shadow_params" in unexpected_keys
        ):
            print("Unexpected keys: ", unexpected_keys)
            raise ValueError

        self.decay = state_dict["ema_decay"]
        self.num_updates = state_dict["ema_num_updates"]
        self.shadow_params = state_dict["ema_shadow_params"]

    def train(self, mode=True):
        if self.training == mode:
            print(
                "Dont call model.train() with the same mode twice! Otherwise EMA parameters may overwrite original parameters"
            )
            print("Current model training mode: ", self.training)
            print("Requested training mode: ", mode)
            raise ValueError

        nn.Module.train(self, mode)
        if mode:
            if len(self.collected_params) > 0:
                self.move_collected_params_to_model_params()
            else:
                print("model.train(True) called but no ema collected parameters!")
        else:
            self.move_model_params_to_collected_params()
            self.move_shadow_params_to_model_params()


# make sure EMA inherited first so it can override the state dict functions
# for CIFAR10
@model_utils.register_model
class UniformRateSequenceTransformerEMA2(EMA, SequenceTransformer, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        SequenceTransformer.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()

@model_utils.register_model
class GaussianTargetRateImageX0PredEMA(EMA, ImageX0PredBase, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()

@model_utils.register_model
class GaussianTargetRateImageX0PredEMAPaul(EMA, ImageX0PredBasePaul, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBasePaul.__init__(self, cfg, device, use_net=False, rank=rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()


# make sure EMA inherited first so it can override the state dict functions
@model_utils.register_model
class UniformRateSequenceTransformerEMA(EMA, SequenceTransformer, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        SequenceTransformer.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()


# make sure EMA inherited first so it can override the state dict functions
@model_utils.register_model
class BirthDeathRateSequenceTransformerEMA(
    EMA, SequenceTransformer, BirthDeathForwardBase
):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        SequenceTransformer.__init__(self, cfg, device, rank)
        BirthDeathForwardBase.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class GaussianRateResidualMLP(ResidualMLP, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        ResidualMLP.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)
