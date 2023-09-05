import torch
import torch.nn as nn
import math
from sddm.model import torch_nets
from sddm.model import torch_backward_model
import functorch


class BidirectionalTransformer(nn.Module):
    def __init__(self, config, readout_dim=None):
        super(BidirectionalTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.temb_scale = config.time_scale_factor

        if self.config.net_arch == "bidir_transformer":
            self.module_l2r = torch_nets.UniDirectionalTransformer(self.config, "l2r")
            self.module_r2l = torch_nets.UniDirectionalTransformer(self.config, "r2l")
        elif self.config.net_arch == "bidir_combiner_transformer":
            self.module_l2r = torch_nets.CombinerAxial(self.config, "l2r")
            self.module_r2l = torch_nets.CombinerAxial(self.config, "r2l")
        else:
            raise ValueError("Unknown net_arch: %s" % self.config.net_arch)

        if readout_dim is None:
            readout_dim = self.config.vocab_size

        self.readout_dim = readout_dim

        if self.config.bidir_readout == "concat":
            self.readout_module = torch_nets.ConcatReadout(
                self.config, readout_dim=readout_dim
            )
        elif self.config.bidir_readout == "res_concat":
            self.readout_module = torch_nets.ConcatResidualReadout(
                self.config, readout_dim=readout_dim
            )
        elif self.config.bidir_readout == "attention":
            self.readout_module = torch_nets.AttentionReadout(
                self.config, readout_dim=readout_dim
            )
        else:
            raise ValueError("Unknown bidir_readout: %s" % self.config.bidir_readout)

    def forward(self, x, t):
        temb = torch_nets.transformer_timestep_embedding(
            t * self.temb_scale, self.config.embed_dim
        )
        x_embed = self.embedding(x)
        input_shape = list(x_embed.shape)[:-1]
        x_embed = x_embed.view(x_embed.shape[0], -1, x_embed.shape[-1])

        l2r_embed = self.module_l2r(x_embed, temb)
        r2l_embed = self.module_r2l(x_embed, temb)

        self.readout_module(l2r_embed, r2l_embed, temb)

        logits = logits.view(input_shape + [self.readout_dim])
        return logits


# still inefficient
class EnumerativeTransformer(nn.Module):
    def __init__(self, config):
        super(EnumerativeTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.temb_scale = config.time_scale_factor
        self.transformer = torch_nets.MaskedTransformer(self.config)

    def forward(self, x, t):
        temb = torch_nets.transformer_timestep_embedding(
            t * self.temb_scale, self.config.embed_dim
        )
        x_embed = self.embedding(x)
        logits = self.enumerative_transformer(x_embed, temb)
        return logits

    def enumerative_transformer(self, x, temb):
        x_shape = x.shape
        x = x.view(x.shape[0], -1)

        def masked_logits(pos):
            x_masked = x.clone()
            x_masked[:, pos] = self.config.vocab_size
            logits = self.transformer(x_masked, temb, pos)
            logits = logits.squeeze(1)
            return logits

        prefix_cond = self.config.get("conditional_dim", 0)
        logits = functorch.vmap(masked_logits, out_dims=1)(
            torch.arange(prefix_cond, x.shape[1])
        )
        # logits = torch.stack([masked_logits(pos) for pos in range(prefix_cond, x.shape[1])], dim=1)
        if prefix_cond:
            dummy_logits = torch.zeros(
                [x.shape[0], prefix_cond] + list(logits.shape[2:]), dtype=torch.float32
            )
            logits = torch.cat([dummy_logits, logits], dim=1)
        logits = logits.view(x_shape + (self.config.vocab_size,))
        return logits


class PrefixConditionalBidirTransformer(nn.Module):
    def __init__(self, config):
        super(PrefixConditionalBidirTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        if self.config.net_arch == "bidir_transformer":
            self.module_l2r = torch_nets.UniDirectionalTransformer(self.config, "l2r")
            self.module_r2l = torch_nets.UniDirectionalTransformer(self.config, "r2l")
        elif self.config.net_arch == "bidir_combiner_transformer":
            self.module_l2r = torch_nets.CombinerAxial(self.config, "l2r")
            self.module_r2l = torch_nets.CombinerAxial(self.config, "r2l")
        else:
            raise ValueError("Unknown net_arch: %s" % self.config.net_arch)

        if self.config.bidir_readout == "concat":
            self.readout_module = torch_nets.ConcatReadout(self.config)
        elif self.config.bidir_readout == "res_concat":
            self.readout_module = torch_nets.ConcatResidualReadout(self.config)
        elif self.config.bidir_readout == "attention":
            self.readout_module = torch_nets.AttentionReadout(self.config)
        else:
            raise ValueError("Unknown bidir_readout: %s" % self.config.bidir_readout)

    def forward(self, x, t):
        temb = torch_nets.transformer_timestep_embedding(
            t * self.config.time_scale_factor, self.config.embed_dim
        )
        conditioner, x = torch.split(x, [self.config.conditional_dim], dim=1)
        x = self.embedding(x)

        l2r_embed = self.module_l2r(x, temb, conditioner)[:, -x.shape[1] :]
        r2l_embed = self.module_r2l(x, temb, conditioner)[:, : x.shape[1]]

        logits = self.readout_module(l2r_embed, r2l_embed, temb)
        dummy_logits = torch.zeros(
            [x.shape[0], self.config.conditional_dim] + list(logits.shape[2:]),
            dtype=torch.float32,
        )
        logits = torch.cat([dummy_logits, logits], dim=1)
        assert logits.shape[1] == self.config.conditional_dim + x.shape[1]
        return logits


class HollowModel(torch_backward_model.CondFactorizedBackwardModel):
    def __init__(self, config):
        super(HollowModel, self).__init__(config)
        if "bidir" in config.net_arch and "transformer" in config.net_arch:
            self.net = BidirectionalTransformer(config)
        elif config.net_arch == "enum_transformer":
            self.net = EnumerativeTransformer(config)
        else:
            raise ValueError("Unknown net arch: %s" % config.net_arch)


class PrefixCondHollowModel(HollowModel):
    def __init__(self, config):
        super(PrefixCondHollowModel, self).__init__(config)
        if "bidir" in config.net_arch and "transformer" in config.net_arch:
            self.net = PrefixConditionalBidirTransformer(config)
        elif config.net_arch == "enum_transformer":
            self.net = EnumerativeTransformer(config)
        else:
            raise ValueError("Unknown net arch: %s" % config.net_arch)

    def loss(self, params, rng, x0, xt, t):
        del x0, rng
        ll_all, log_xt = self.get_logprob(params, xt, t)
        ll_all = ll_all[:, self.config.conditional_dim :]
        log_xt = log_xt[:, self.config.conditional_dim :]
        loss = self.calc_loss(xt, t, ll_all, log_xt)
        loss = torch.sum(loss) / xt.shape[0]
        aux = {"loss": loss}
        return loss, aux
