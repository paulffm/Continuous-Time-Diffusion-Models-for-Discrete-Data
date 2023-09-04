import torch
import torch.nn as nn
import math
from sddm.model import torch_nets
from sddm.model import torch_backward_model

class BidirectionalTransformer(nn.Module):
    def __init__(self, config):
        super(BidirectionalTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.temb_scale = config.time_scale_factor

    def forward(self, x, t):
        temb = self.transformer_timestep_embedding(t * self.temb_scale, self.config.embed_dim)
        x_embed = self.embedding(x)
        logits = self.bidir_transformer(x_embed, temb)
        return logits

    def transformer_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        assert embedding_dim % 2 == 0
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -math.log(max_positions) / (half_dim - 1))
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def bidir_transformer(self, x, temb, readout_dim=None):
        if readout_dim is None:
            readout_dim = self.config.vocab_size
        input_shape = list(x.shape)[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        if self.config.net_arch == 'bidir_transformer':
            module = torch_nets.UniDirectionalTransformer
        elif self.config.net_arch == 'bidir_combiner_transformer':
            module = torch_nets.CombinerAxial
        else:
            raise ValueError('Unknown net_arch: %s' % self.config.net_arch)
        l2r_embed = module(self.config, 'l2r')(x, temb)
        r2l_embed = module(self.config, 'r2l')(x, temb)
        if self.config.bidir_readout == 'concat':
            readout_module = torch_nets.ConcatReadout
        elif self.config.bidir_readout == 'res_concat':
            readout_module = torch_nets.ConcatResidualReadout
        elif self.config.bidir_readout == 'attention':
            readout_module = torch_nets.AttentionReadout
        else:
            raise ValueError('Unknown bidir_readout: %s' % self.config.bidir_readout)
        logits = readout_module(self.config, readout_dim=readout_dim)(
            l2r_embed, r2l_embed, temb)
        logits = logits.view(input_shape + [readout_dim])
        return logits

class EnumerativeTransformer(nn.Module):
    def __init__(self, config):
        super(EnumerativeTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.temb_scale = config.time_scale_factor

    def forward(self, x, t):
        temb = self.transformer_timestep_embedding(t * self.temb_scale, self.config.embed_dim)
        x_embed = self.embedding(x)
        logits = self.enumerative_transformer(x_embed, temb)
        return logits

    def transformer_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        assert embedding_dim % 2 == 0
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -math.log(max_positions) / (half_dim - 1))
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def enumerative_transformer(self, x, temb):
        transformer = torch_nets.MaskedTransformer(self.config)
        x_shape = x.shape
        x = x.view(x.shape[0], -1)

        def masked_logits(pos):
            x_masked = x.clone()
            x_masked[:, pos] = self.config.vocab_size
            logits = transformer(x_masked, temb, pos)
            logits = logits.squeeze(1)
            return logits

        prefix_cond = self.config.get('conditional_dim', 0)
        logits = torch.stack([masked_logits(pos) for pos in range(prefix_cond, x.shape[1])], dim=1)
        if prefix_cond:
            dummy_logits = torch.zeros(
                x.shape[0], prefix_cond, *logits.shape[2:], dtype=torch.float32)
            logits = torch.cat([dummy_logits, logits], dim=1)
        logits = logits.view(x_shape + (self.config.vocab_size,))
        return logits

def prefix_conditional_forward(x, t, config, net_fn):
    embedding = nn.Embedding(config.vocab_size, config.embed_dim)
    temb = torch_nets.transformer_timestep_embedding(
        t * config.time_scale_factor, config.embed_dim)
    conditioner, x = torch.split(x, [config.conditional_dim], dim=1)
    x_embed = embedding(x)
    logits = net_fn(x_embed, temb, conditioner)
    dummy_logits = torch.zeros(
        x.shape[0], config.conditional_dim, *logits.shape[2:], dtype=torch.float32)
    logits = torch.cat([dummy_logits, logits], dim=1)
    assert logits.shape[1] == config.conditional_dim + x.shape[1]
    return logits

class PrefixConditionalBidirTransformer(nn.Module):
    def __init__(self, config):
        super(PrefixConditionalBidirTransformer, self).__init__()
        self.config = config

    def logits_fn(self, x, temb, conditioner):
        if self.config.net_arch == 'bidir_transformer':
            module = torch_nets.UniDirectionalTransformer
        elif self.config.net_arch == 'bidir_combiner_transformer':
            module = torch_nets.CombinerAxial
        else:
            raise ValueError('Unknown net_arch: %s' % self.config.net_arch)
        l2r_embed = module(self.config, 'l2r')(x, temb, conditioner)[:, -x.shape[1]:]
        r2l_embed = module(self.config, 'r2l')(x, temb, conditioner)[:, :x.shape[1]]
        if self.config.bidir_readout == 'concat':
            readout_module = torch_nets.ConcatReadout
        elif self.config.bidir_readout == 'res_concat':
            readout_module = torch_nets.ConcatResidualReadout
        elif self.config.bidir_readout == 'attn':
            readout_module = torch_nets.AttentionReadout
        else:
            raise ValueError('Unknown bidir_readout: %s' % self.config.bidir_readout)
        logits = readout_module(self.config)(l2r_embed, r2l_embed, temb)
        return logits

    def forward(self, x, t):
        config = self.config
        embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        temb = torch_nets.transformer_timestep_embedding(
            t * config.time_scale_factor, config.embed_dim)
        conditioner, x = torch.split(x, [config.conditional_dim], dim=1)
        x_embed = embedding(x)
        logits = self.logits_fn(x_embed, temb, conditioner)
        dummy_logits = torch.zeros(
            x.shape[0], config.conditional_dim, *logits.shape[2:], dtype=torch.float32)
        logits = torch.cat([dummy_logits, logits], dim=1)
        assert logits.shape[1] == config.conditional_dim + x.shape[1]
        return logits

class HollowModel(torch_backward_model.CondFactorizedBackwardModel):
    def __init__(self, config):
        super(HollowModel, self).__init__(config)
        if 'bidir' in config.net_arch and 'transformer' in config.net_arch:
            self.net = BidirectionalTransformer(config)
        elif config.net_arch == 'enum_transformer':
            self.net = EnumerativeTransformer(config)
        else:
            raise ValueError('Unknown net arch: %s' % config.net_arch)

class PrefixCondHollowModel(HollowModel):
    def __init__(self, config):
        super(PrefixCondHollowModel, self).__init__(config)
        if 'bidir' in config.net_arch and 'transformer' in config.net_arch:
            self.net = PrefixConditionalBidirTransformer(config)
        elif config.net_arch == 'enum_transformer':
            self.net = EnumerativeTransformer(config)
        else:
            raise ValueError('Unknown net arch: %s' % config.net_arch)

    def loss(self, params, rng, x0, xt, t):
        del x0, rng
        ll_all, log_xt = self.get_logprob(params, xt, t)
        ll_all = ll_all[:, self.config.conditional_dim:]
        log_xt = log_xt[:, self.config.conditional_dim:]
        loss = self.calc_loss(xt, t, ll_all, log_xt)
        loss = torch.sum(loss) / xt.shape[0]
        aux = {'loss': loss}
        return loss, aux