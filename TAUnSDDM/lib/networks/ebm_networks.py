
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.networks.network_utils import transformer_timestep_embedding
from lib.networks.hollow_networks import MaskedTransformer

class BinaryMLPScoreFunc(nn.Module):
    def __init__(self, cfg):
        super(BinaryMLPScoreFunc, self).__init__()
        self.num_layers = cfg.model.num_layers
        self.hidden_size = cfg.model.mlp_dim
        self.time_scale_factor = cfg.model.time_scale_factor


        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.final_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x, t):
        temb = transformer_timestep_embedding(
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
        self.masked_transformer = MaskedTransformer(config)
        self.embed_dim = config.model.embed_dim
        self.time_scale_factor = self.config.model.time_scale_factor
        self.S = config.data.S
        self.device = config.device

    def forward(self, x, t):


        temb = transformer_timestep_embedding(
            t * self.time_scale_factor, self.embed_dim
        )
        x = x.view(x.size(0), -1).long()
        cls_token = (
            torch.ones((x.size(0), 1), device=self.device, dtype=torch.long) * self.S
        )

        x = torch.cat([cls_token, x], dim=1)
        score = self.masked_transformer(x, temb, 0)[..., 0]
        return score


class CatMLPScoreFunc(nn.Module):
    def __init__(self, cfg
    ):
        super(CatMLPScoreFunc, self).__init__()
        self.S = cfg.data.S
        self.cat_embed_size = cfg.model.embed_dim
        self.num_layers = cfg.model.num_layers
        self.hidden_size = cfg.model.mlp_dim
        self.time_scale_factor = cfg.model.time_scale_factor

        self.embed = nn.Embedding(self.S, self.cat_embed_size)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.final_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x, t):
        temb = transformer_timestep_embedding(
            t * self.time_scale_factor, self.hidden_size
        )
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x) + temb
            x = F.silu(x)
        x = self.final_layer(x)
        return x
