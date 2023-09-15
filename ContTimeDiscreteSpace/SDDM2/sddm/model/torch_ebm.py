import torch
import torch.nn as nn
import torch.nn.functional as F
from sddm.model import torch_backward_model
from sddm.model import torch_forward_model
from sddm.model import torch_nets


class BinaryMLPScoreFunc(nn.Module):
    def __init__(self, num_layers, hidden_size, time_scale_factor=1000.0):
        super(BinaryMLPScoreFunc, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_scale_factor = time_scale_factor
        self.transformer_timestep_embedding = torch_nets.transformer_timestep_embedding

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.final_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x, t):
        temb = self.transformer_timestep_embedding(
            t * self.time_scale_factor, self.hidden_size
        )
        x = x.float()
        for layer in self.layers:
            x = layer(x) + temb
            x = F.elu(x)
        x = self.final_layer(x)
        return x


class BinaryTransformerScoreFunc(nn.Module):
    def __init__(self, config):
        super(BinaryTransformerScoreFunc, self).__init__()
        self.config = config
        self.transformer_timestep_embedding = torch_nets.transformer_timestep_embedding
        self.masked_transformer = torch_nets.MaskedTransformer(config)

    def forward(self, x, t):
        temb = self.transformer_timestep_embedding(
            t * self.config.time_scale_factor, self.config.embed_dim
        )
        x = x.view(x.size(0), -1).long()
        cls_token = (
            torch.ones((x.size(0), 1), dtype=torch.long) * self.config.vocab_size
        )
        x = torch.cat([cls_token, x], dim=1)
        score = self.masked_transformer(x, temb, 0)[..., 0]
        return score


class CatMLPScoreFunc(nn.Module):
    def __init__(
        self,
        vocab_size,
        cat_embed_size,
        num_layers,
        hidden_size,
        time_scale_factor=1000.0,
    ):
        super(CatMLPScoreFunc, self).__init__()
        self.vocab_size = vocab_size
        self.cat_embed_size = cat_embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_scale_factor = time_scale_factor
        self.transformer_timestep_embedding = torch_nets.transformer_timestep_embedding

        self.embed = nn.Embedding(self.vocab_size, self.cat_embed_size)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.final_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x, t):
        temb = self.transformer_timestep_embedding(
            t * self.time_scale_factor, self.hidden_size
        )
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x) + temb
            x = F.silu(x)
        x = self.final_layer(x)
        return x


class BinaryScoreModel(torch_backward_model.BackwardModel):
    def __init__(self, config):
        super(BinaryScoreModel, self).__init__()
        self.config = config
        if config.net_arch == "mlp":
            self.net = BinaryMLPScoreFunc(
                num_layers=config.num_layers,
                hidden_size=config.embed_dim,
                time_scale_factor=config.time_scale_factor,
            )
        elif config.net_arch == "transformer":
            self.net = BinaryTransformerScoreFunc(config)
        else:
            raise ValueError("Unknown net arch: %s" % config.net_arch)
        self.fwd_model = torch_forward_model.get_fwd_model(self.config)

    def get_q(self, params, xt, t):
        bsize = xt.shape[0]
        ddim = self.config.discrete_dim
        qxt = self.net(xt, t)
        #
        mask = torch.eye(ddim, device=xt.device).repeat_interleave(bsize, 0)
        xrep = torch.tile(xt, (ddim, 1))

        xneg = (mask - xrep) * mask + (1 - mask) * xrep
        t = torch.tile(t, (ddim,))
        qxneg = self.net(xneg, t)
        qxt = torch.tile(qxt, (ddim, 1))
        return qxneg, qxt

    def get_logits(self, params, xt, t):
        bsize = xt.shape[0]
        qxneg, qxt = self.get_q(params, xt, t)
        qxneg = qxneg.view(-1, bsize).t()
        qxt = qxt.view(-1, bsize).t()
        xt_onehot = F.one_hot(xt, num_classes=2).to(qxt.dtype)
        qxneg, qxt = qxneg.unsqueeze(-1), qxt.unsqueeze(-1)
        logits = xt_onehot * qxt + (1 - xt_onehot) * qxneg
        return logits

    def get_ratio(self, params, xt, t, xt_target=None):
        qxneg, qxt = self.get_q(params, xt, t)
        bsize = xt.shape[0]
        ratio = torch.exp(qxneg - qxt)
        return ratio.view(-1, bsize).t()

    def get_logprob(self, params, xt, t, xt_target=None):
        logits = self.get_logits(params, xt, t)
        return torch_backward_model.get_logprob_with_logits(self, xt, t, logits)

    def loss(self, params, rng, x0, xt, t):
        _, ll_xt = self.get_logprob(params, xt, t)
        loss = -ll_xt
        loss = loss.sum() / xt.shape[0]
        aux = {"loss": loss}
        return loss, aux


class CategoricalScoreModel(torch_backward_model.BackwardModel):
    def __init__(self, config):
        super(CategoricalScoreModel, self).__init__()
        self.config = config
        if config.net_arch == "mlp":
            if config.vocab_size == 2:
                self.net = BinaryMLPScoreFunc(
                    num_layers=config.num_layers,
                    hidden_size=config.embed_dim,
                    time_scale_factor=config.time_scale_factor,
                )
            else:
                self.net = CatMLPScoreFunc(
                    vocab_size=config.vocab_size,
                    cat_embed_size=config.cat_embed_size,
                    num_layers=config.num_layers,
                    hidden_size=config.embed_dim,
                    time_scale_factor=config.time_scale_factor,
                )
        else:
            raise ValueError("Unknown net arch: %s" % config.net_arch)

    def get_logits(self, params, xt, t):
        assert xt.ndim == 2
        bsize = xt.shape[0]
        ddim = self.config.discrete_dim
        vocab_size = self.config.vocab_size
        mask = torch.eye(ddim, dtype=torch.int32).repeat_interleave(
            bsize * vocab_size, 0
        )
        xrep = torch.tile(xt, (ddim * vocab_size, 1))
        candidate = torch.arange(vocab_size).repeat_interleave(bsize, 0)
        candidate = torch.tile(candidate.unsqueeze(1), ((ddim, 1)))
        xall = mask * candidate + (1 - mask) * xrep
        t = torch.tile(t, (ddim * vocab_size,))
        qall = self.net(x=xall, t=t)
        logits = torch.reshape(qall, (ddim, vocab_size, bsize))
        logits = logits.permute(2, 0, 1)
        return logits

    def get_logprob(self, params, xt, t):
        bsize = xt.shape[0]
        ddim = self.config.discrete_dim
        logits = self.get_logits(params, xt, t)
        ll_all = F.log_softmax(logits, dim=-1)
        ll_xt = ll_all[torch.arange(bsize)[:, None], torch.arange(ddim)[None, :], xt]
        return ll_all, ll_xt

    def loss(self, params, rng, x0, xt, t):
        del x0, rng
        _, ll_xt = self.get_logprob(params, xt, t)
        loss = -ll_xt.sum(dim=-1)
        loss = loss.mean()
        aux = {"loss": loss.item()}
        return loss, aux
