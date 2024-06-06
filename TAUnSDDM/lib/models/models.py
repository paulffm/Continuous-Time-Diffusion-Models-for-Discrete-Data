import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.networks import ddsm_networks
import lib.models.model_utils as model_utils
from torchtyping import TensorType
import torch.autograd.profiler as profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.networks import hollow_networks, ebm_networks, tau_networks, unet, dit, u_vit
from lib.models.forward_model import (
    UniformRate,
    UniformVariantRate,
    GaussianTargetRate,
    BirthDeathForwardBase,
)
from lib.datasets.sudoku import define_relative_encoding


def log_minus_exp(a, b, eps=1e-6):
    """
    Compute log (exp(a) - exp(b)) for (b<a)
    From https://arxiv.org/pdf/2107.03006.pdf
    """
    return a + torch.log1p(-torch.exp(b - a) + eps)


def sample_logistic(net_out, B, C, D, S, fix_logistic, device):
    """
    net_out: Output of neural network with shape B, 2*C, H,W
    B: Batch Size
    C: Number of channel
    D: Dimension = C*H*W
    S: Number of States

    """
    mu = net_out[0].unsqueeze(-1)  # B, C, H, W, 1
    log_scale = net_out[1].unsqueeze(-1)  # B, C, H, W, 1

    # if self.padding:
    #    mu = mu[:, :, :-1, :-1, :]
    #    log_scale = log_scale[:, :, :-1, :-1, :]

    # The probability for a state is then the integral of this continuous distribution between
    # this state and the next when mapped onto the real line. To impart a residual inductive bias
    # on the output, the mean of the logistic distribution is taken to be tanh(xt + μ′) where xt
    # is the normalized input into the model and μ′ is mean outputted from the network.
    # The normalization operation takes the input in the range 0, . . . , 255 and maps it to [−1, 1].
    inv_scale = torch.exp(-(log_scale - 2))

    bin_width = 2.0 / S
    bin_centers = torch.linspace(
        start=-1.0 + bin_width / 2,
        end=1.0 - bin_width / 2,
        steps=S,
        device=device,
    ).view(1, 1, 1, 1, S)

    sig_in_left = (bin_centers - bin_width / 2 - mu) * inv_scale
    bin_left_logcdf = F.logsigmoid(sig_in_left)
    sig_in_right = (bin_centers + bin_width / 2 - mu) * inv_scale
    bin_right_logcdf = F.logsigmoid(sig_in_right)

    logits_1 = log_minus_exp(bin_right_logcdf, bin_left_logcdf)
    logits_2 = log_minus_exp(
        -sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf
    )
    if fix_logistic:
        logits = torch.min(logits_1, logits_2)
    else:
        logits = logits_1
    logits = logits.view(B, D, S)  # shape before B, C, H, W, S

    return logits


class UViTModel(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()


        self.S = cfg.data.S

        # assert len(cfg.data.shape) == 1

        tmp_net = u_vit.UViT(
            img_size=cfg.data.image_size,
            num_states=cfg.data.S,
            patch_size=cfg.model.patch_size,
            in_chans=cfg.model.input_channel,
            embed_dim=cfg.model.hidden_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            mlp_time_embed=True,
            num_classes=-1,
            use_checkpoint=False,
            conv=True,
            skip=True,
            model_output=cfg.model.model_output,
        ).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.data_shape = cfg.data.shape

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"], label: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """
        B, D = x.shape

        logits = self.net(x, times)  
        logits = logits.view(B, D, self.S) # (B, D, S)

        return logits


class LogDiT(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.fix_logistic = cfg.model.fix_logistic
        self.S = cfg.data.S
        self.data_shape = cfg.data.shape
        net = dit.DiT(
            input_size=cfg.data.image_size,  # 28
            patch_size=cfg.model.patch_size,  # 2
            in_channels=cfg.model.input_channel,
            hidden_size=cfg.model.hidden_dim,  # 1152
            depth=cfg.model.depth,  # 28
            num_heads=cfg.model.num_heads,  # 16
            mlp_ratio=cfg.model.mlp_ratio,  # 4.0,
            class_dropout_prob=cfg.model.dropout,  # 0.1
            num_classes=self.S,
            logits_pars_out=cfg.model.model_output,
            x_min_max=cfg.model.data_min_max,
        )  # logistic_pars output)

        if cfg.distributed:
            self.net = DDP(net, device_ids=[rank])
        else:
            self.net = net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"], y: TensorType["B"] = None
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space for each pixel
        """
        if len(x.shape) == 2:
            B, D = x.shape
            C, H, W = self.data_shape
            x = x.view(B, C, H, W)
        else:
            B, C, H, W = x.shape

        net_out = self.net(x, times, y)  # (B, 2*C, H, W)
        logits = sample_logistic(
            net_out, B, C, D, self.S, self.fix_logistic, self.device
        )
        return logits


class ImageX0PredBasePaul(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()
        self.cfg = cfg
        self.fix_logistic = cfg.model.fix_logistic
        self.data_shape = cfg.data.shape
        self.S = cfg.data.S
        self.padding = cfg.model.padding
        if self.padding:
            img_size = cfg.data.image_size + 1
        else:
            img_size = cfg.data.image_size

        net = unet.UNet(
            in_channel=cfg.model.input_channels,
            out_channel=cfg.model.input_channels,
            channel=cfg.model.ch,
            channel_multiplier=cfg.model.ch_mult,
            n_res_blocks=cfg.model.num_res_blocks,
            attn_resolutions=cfg.model.attn_resolutions,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            model_output=cfg.model.model_output,  # c or 'logistic_pars'
            num_classes=cfg.data.S,
            x_min_max=cfg.model.data_min_max,
            img_size=img_size,
        ).to(device)

        if cfg.distributed:
            self.net = DDP(net, device_ids=[rank])
        else:
            self.net = net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space for each pixel
        """
        if len(x.shape) == 2:
            B, D = x.shape
            C, H, W = self.data_shape
            x = x.view(B, C, H, W)
        else:
            B, C, H, W = x.shape

        if self.padding:
            x = nn.ReplicationPad2d((0, 1, 0, 1))(x.float())

        # Output: 3 × 32 × 32 × 2 => mean and log scale of a logistic distribution
        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf

        net_out = self.net(x, times)  # (B, 2*C, H, W)
        if self.cfg.model.model_output == "logits":
            logits = net_out

        else:
            mu = net_out[0].unsqueeze(-1)
            log_scale = net_out[1].unsqueeze(-1)
            # if self.padding:
            #    mu = mu[:, :, :-1, :-1, :]
            #    log_scale = log_scale[:, :, :-1, :-1, :]

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

        if self.padding:
            logits = logits[:, :, :-1, :-1, :]
            logits = logits.reshape(B, D, self.S)
            # logits = logits.view(B, D, self.S)
        else:
            # logits.view(B, C, H, W, self.S)# d3pm
            logits = logits.view(B, D, self.S)

        return logits  # .view(B, D, self.S) # d3pm

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

        tmp_net = tau_networks.UNet(
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

        # logits = logits.view(B, D, S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
        Compute log (exp(a) - exp(b)) for (b<a)
        From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b - a) + eps)


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
        use_cat = cfg.model.use_cat

        # assert len(cfg.data.shape) == 1
        max_len = cfg.data.shape[0]

        tmp_net = tau_networks.TransformerEncoder(
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
            use_cat,
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

        tmp_net = tau_networks.ResidualMLP(
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


class HollowTransformer(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()
        if cfg.model.nets == "bidir_transformer2":
            tmp_net = hollow_networks.BidirectionalTransformer2(
                cfg, readout_dim=None
            ).to(device)
        # elif cfg.model.nets == "visual":
        #    tmp_net = hollow_networks.BiVisualTransformer(
        #        cfg, readout_dim=None
        #    ).to(device)
        else:
            tmp_net = hollow_networks.BidirectionalTransformer(
                cfg, readout_dim=None
            ).to(device)

        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """

        logits = self.net(x, times)  # (B, D, S)

        return logits


class HollowTransformerLogistics(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()
        self.S = cfg.data.S
        self.data_shape = cfg.data.shape
        self.fix_logistic = cfg.model.fix_logistic
        self.device = cfg.device
        if cfg.model.nets == "bidir_transformer2":
            tmp_net = hollow_networks.BidirectionalTransformer2(cfg, readout_dim=2).to(
                device
            )
        # elif cfg.model.nets == "visual":
        #    tmp_net = hollow_networks.BiVisualTransformer(
        #        cfg, readout_dim=None
        #    ).to(device)
        else:
            tmp_net = hollow_networks.BidirectionalTransformer(cfg, readout_dim=2).to(
                device
            )

        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """
        print("SHAPE", x.shape)
        B, D = x.shape
        C, H, W = self.data_shape
        x = x.view(B, C, H, W)

        net_out = self.net(x, times)  # (B, D, 2)
        net_out = net_out.view(B, 2 * C, H, W)

        mu = net_out[0].unsqueeze(-1)
        log_scale = net_out[1].unsqueeze(-1)

        # if self.padding:
        #    mu = mu[:, :, :-1, :-1, :]
        #    log_scale = log_scale[:, :, :-1, :-1, :]

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

        logits = logits.view(B, D, self.S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
        Compute log (exp(a) - exp(b)) for (b<a)
        From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b - a) + eps)

        return logits


class MaskedModel(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        tmp_net = hollow_networks.EnumerativeTransformer(cfg).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """

        logits = self.net(x, times)  # (B, D, S)

        return logits


class BertMLPRes(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        tmp_net = hollow_networks.BertEnumTransformer(cfg).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """

        logits = self.net(x, times)  # (B, D, S)

        return logits


class SudokuScoreNet(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()
        encoding = define_relative_encoding()
        tmp_net = ddsm_networks.SudokuScoreNet(cfg, encoding).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """
        B, D = x.shape
        x = x.view(-1, 9, 9, 9)
        logits = self.net(x, times)  # (B, D, S)
        logits = logits.view(B, 81, 9)
        return logits


class ProteinScoreNet(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        tmp_net = ddsm_networks.ProteinScoreNet(cfg).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """
        x = x.view(-1, 15 * 15 * 1)
        logits = self.net(x, times)  # (B, D, S)

        logits = logits.view(-1, 1, 15, 15, 3)
        return logits


class BinaryEBM(nn.Module):
    def __init__(self, cfg, device, encoding, rank=None):
        super().__init__()

        tmp_net = ebm_networks.BinaryTransformerScoreFunc(cfg).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """

        logits = self.net(x, times)  # (B, D, S)

        return logits


# Based on https://github.com/yang-song/score_sde_pytorch/blob/ef5cb679a4897a40d20e94d8d0e2124c3a48fb8c/models/ema.py
class EMA:
    def __init__(self, cfg):
        self.decay = cfg.model.ema_decay
        self.device = cfg.device
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
                param = param.to(self.device)
                s_param = s_param.to(self.device)
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
        print("ema state dict function")
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


##############################################################################################################################################################

# make sure EMA inherited first so it can override the state dict functions
# for CIFAR10

@model_utils.register_model
class GaussianUViTEMA(
    EMA, UViTModel, GaussianTargetRate
):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        UViTModel.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()

@model_utils.register_model
class GaussianLogDiTEMA(EMA, LogDiT, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        LogDiT.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniformRateImageX0PredEMA(EMA, ImageX0PredBasePaul, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBasePaul.__init__(self, cfg, device, rank=rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniVarHollowEMA(EMA, HollowTransformer, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        HollowTransformer.__init__(self, cfg, device, rank)
        UniformVariantRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniVarHollowEMALogistics(EMA, HollowTransformerLogistics, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        HollowTransformerLogistics.__init__(self, cfg, device, rank)
        UniformVariantRate.__init__(self, cfg, device)

        self.init_ema()


# hollow
@model_utils.register_model
class UniformMaskedEMA(EMA, MaskedModel, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        MaskedModel.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniVarMaskedEMA(EMA, MaskedModel, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        MaskedModel.__init__(self, cfg, device, rank)
        UniformVariantRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniformHollowEMA(EMA, HollowTransformer, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        HollowTransformer.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniVarScoreNetEMA(EMA, SudokuScoreNet, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        SudokuScoreNet.__init__(self, cfg, device, rank)
        UniformVariantRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniVarProteinScoreNetEMA(EMA, ProteinScoreNet, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ProteinScoreNet.__init__(self, cfg, device, rank)
        UniformVariantRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniProteinD3PM(EMA, ProteinScoreNet):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ProteinScoreNet.__init__(self, cfg, device, rank)

        self.init_ema()


@model_utils.register_model
class GaussianTargetRateImageX0PredEMAPaul(
    EMA, ImageX0PredBasePaul, GaussianTargetRate
):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBasePaul.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class GaussianHollowEMA(EMA, HollowTransformer, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        HollowTransformer.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class GaussianTargetRateImageX0PredEMA(EMA, ImageX0PredBase, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()


# Maze, MNIST
@model_utils.register_model
class UniformRateUnetEMA(EMA, ImageX0PredBasePaul, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBasePaul.__init__(self, cfg, device, rank=rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniVarUnetEMA(EMA, ImageX0PredBasePaul, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBasePaul.__init__(self, cfg, device, rank=rank)
        UniformVariantRate.__init__(self, cfg, device)

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


@model_utils.register_model
class UniformRateResMLP(ResidualMLP, UniformRate):
    def __init__(self, cfg, device, rank=None):
        # EMA.__init__(self, cfg)
        ResidualMLP.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)


@model_utils.register_model
class UniVarBertEMA(EMA, BertMLPRes, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        BertMLPRes.__init__(self, cfg, device, rank)
        UniformVariantRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniBertD3PM(EMA, BertMLPRes):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        BertMLPRes.__init__(self, cfg, device, rank)

        self.init_ema()


@model_utils.register_model
class UniformBertEMA(EMA, BertMLPRes, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        BertMLPRes.__init__(self, cfg, device, rank)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniVarBinaryEBMEMA(EMA, BinaryEBM, UniformVariantRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        BinaryEBM.__init__(self, cfg, device, rank)
        UniformVariantRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class UniformBDTEMA(EMA, hollow_networks.BidirectionalTransformer, UniformRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        hollow_networks.BidirectionalTransformer.__init__(
            self, cfg, readout_dim=None
        )  # .to(device)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()
