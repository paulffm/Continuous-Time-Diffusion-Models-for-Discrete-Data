import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools
import torch
import torch.nn as nn
from sddm.common import torch_utils
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, features, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.features = features
        self.activation = activation

        layers = []
        for i in range(len(features) - 1):
            layers.append(nn.Linear(features[i], features[i + 1]))
            layers.append(self.activation())
        layers.append(nn.Linear(features[-2], features[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def apply_film(film_params, x):
    film_params = film_params.unsqueeze(1)
    assert film_params.dim() == 3 and x.dim() == 3
    a, b = torch.chunk(film_params, 2, dim=-1)
    x = a * x + b
    return x


class ConcatReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ConcatReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        out_dim = self.readout_dim if self.readout_dim != 0 else self.config.vocab_size
        self.predictor = MLP([2 * self.config.embed_dim, out_dim], activation=nn.GELU())

    def forward(self, l2r_embed, r2l_embed, _):
        state = torch.cat([l2r_embed, r2l_embed], dim=-1)
        #out_dim = self.readout_dim if self.readout_dim != 0 else self.config.vocab_size
        #predictor = MLP([2 * self.config.embed_dim, out_dim], activation=nn.GELU())
        logits = self.predictor(state)
        return logits


class ResidualReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ResidualReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.embed_dim = config.embed_dim  # To be set during forward

        self.out_dim = self.readout_dim if self.readout_dim != 0 else self.config.vocab_size

        
        #self.dense_in = nn.Linear(temb.shape[1], 2 * embed_dim)
        self.mlp = MLP([self.config.mlp_dim, 4 * self.embed_dim], activation=nn.GELU())

        #self.lnorm = nn.LayerNorm(embed_dim)

        #self.dense_out  = nn.Linear(embed_dim, self.out_dim)

    # es fehlt temb.shape und embed_dim in config , dann könnte alles in __init_

    def forward(self, x, temb):
        embed_dim = x.shape[-1]
        temb = self.mlp(temb)
        for _ in range(self.config.num_output_ffresiduals):
            film_params = nn.Linear(temb.shape[1], 2 * embed_dim)(temb)
            z = MLP([self.config.mlp_dim, embed_dim], activation=nn.GELU())(x)
            x = nn.LayerNorm(embed_dim)(x + z)
            x = apply_film(film_params, x)
        logits = nn.Linear(embed_dim, self.out_dim)(x)
        return logits
    
class ResidualReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ResidualReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.embed_dim = config.embed_dim
        
        self.out_dim = self.readout_dim if self.readout_dim != 0 else self.config.vocab_size
        
        self.mlp = MLP([self.config.mlp_dim, 4 * self.embed_dim], activation=nn.GELU())
        
        # Vorbereitung der Linearen Layer im Konstruktor
        self.film_params_layer = nn.Linear(self.embed_dim, 2 * self.embed_dim)  # adjust the input size
        self.z_layer = MLP([self.config.mlp_dim, self.embed_dim], activation=nn.GELU())
        self.norm_layer = nn.LayerNorm(self.embed_dim)
        self.logits_layer = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, x, temb):
        temb = self.mlp(temb)
        for _ in range(self.config.num_output_ffresiduals):
            film_params = self.film_params_layer(temb)
            z = self.z_layer(x)
            x = self.norm_layer(x + z)
            x = apply_film(film_params, x)  # ensure the apply_film function is properly defined
        logits = self.logits_layer(x)
        return logits


class ConcatResidualReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ConcatResidualReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.model = ResidualReadout(self.config, readout_dim=self.readout_dim)

    def forward(self, l2r_embed, r2l_embed, temb):
        state = torch.cat([l2r_embed, r2l_embed], dim=-1)
        #return ResidualReadout(self.config, readout_dim=self.readout_dim)(state, temb)
        return self.model(state, temb)


def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert embedding_dim % 2 == 0
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class TransformerMlpBlock(nn.Module):
    def __init__(
        self,
        mlp_dim,
        out_dim=None,
        dtype=torch.float32,
        dropout_rate=0.0,
        dropout_deterministic=False,
        kernel_init=None,
        bias_init=None,
    ):
        super(TransformerMlpBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.dropout_deterministic = dropout_deterministic
        self.kernel_init = (
            kernel_init if kernel_init is not None else nn.init.xavier_uniform_
        )
        self.bias_init = bias_init if bias_init is not None else nn.init.normal_

        layers = []
        layers.append(nn.Linear(mlp_dim, mlp_dim, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(
            nn.Linear(
                mlp_dim,
                self.out_dim if self.out_dim is not None else mlp_dim,
                bias=False,
            )
        )
        layers.append(nn.Dropout(p=dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)



def cross_attention(config, l2r_embed, r2l_embed, temb):
    seq_len = l2r_embed.shape[1]
    temb = temb.unsqueeze(1)
    head_dim = config.qkv_dim // config.num_heads
    dense = functools.partial(
        nn.Linear,
        in_features=config.qkv_dim,
        out_features=config.num_heads * head_dim,
        bias=False,
    )
    query = dense()(l2r_embed + r2l_embed)
    all_embed = torch.cat([temb, l2r_embed, r2l_embed], dim=1)
    key = dense()(all_embed)
    val = dense()(all_embed)
    query = query / torch.sqrt(torch.tensor(query.shape[-1], dtype=config.dtype))
    logits = torch.einsum("bqhd,bkhd->bhqk", query, key)

    idx = torch.arange(seq_len, dtype=torch.int32)
    # fehler hier
    att_l2r_mask = torch.ge(torch.unsqueeze(idx, 1), torch.unsqueeze(idx, 0))
    att_r2l_mask = torch.le(torch.unsqueeze(idx, 1), torch.unsqueeze(idx, 0))
    # fehler
    att_t = torch.ones((1, seq_len, 1))
    joint_mask = torch.cat([att_t, att_l2r_mask, att_r2l_mask], dim=-1)
    joint_mask = joint_mask.unsqueeze(0)
    attn_weights = torch.where(joint_mask, logits, torch.finfo(config.dtype).min)
    attn_weights = F.softmax(attn_weights, dim=-1)
    x = torch.einsum("bhqk,bkhd->bqhd", attn_weights, val)
    x = nn.Linear(config.qkv_dim, config.embed_dim)(x)
    return x


class AttentionReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(AttentionReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.model = ResidualReadout(self.config, self.readout_dim)

    def forward(self, l2r_embed, r2l_embed, temb):
        inputs = l2r_embed + r2l_embed
        if self.config.transformer_norm_type == "prenorm":
            l2r_embed = nn.LayerNorm(l2r_embed.shape[-1])(l2r_embed)
            r2l_embed = nn.LayerNorm(r2l_embed.shape[-1])(r2l_embed)
            x = cross_attention(self.config, l2r_embed, r2l_embed, temb)
            x = x + inputs
        elif self.config.transformer_norm_type == "postnorm":
            x = cross_attention(self.config, l2r_embed, r2l_embed, temb)
            x = x + inputs
            x = nn.LayerNorm(inputs.shape[-1])(x)
        else:
            raise ValueError("unknown norm type %s" % self.config.transformer_norm_type)
        #return ResidualReadout(self.config, self.readout_dim)(x, temb)
        return self.model(x, temb)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.multi_head_attn = nn.MultiheadAttention(
            config.qkv_dim, config.num_heads, dropout=config.attention_dropout_rate
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.trans_mlp_block = TransformerMlpBlock(
            config.mlp_dim,
            dtype=config.dtype,
            dropout_rate=config.dropout_rate,
            dropout_deterministic=config.dropout_deterministic,
        )

    def forward(self, inputs, masks):
        assert inputs.ndim == 3

        if self.config.transformer_norm_type == "prenorm":
            # sa
            x = nn.LayerNorm(inputs.shape[-1])(inputs)
            x = self.multi_head_attn(x, x, x, attn_mask=masks)
            x = self.dropout(x)
            x = x + inputs

            # ff
            z = nn.LayerNorm(x.shape[-1])(x)
            z = self.trans_mlp_block(z)
            z = x + z
        else:
            # sa
            x = self.multi_head_attn(inputs, inputs, inputs, attn_mask=masks)
            x = self.dropout(x)
            x = x + inputs
            x = nn.LayerNorm(inputs.shape[-1])(x)
            # ff
            z = self.trans_mlp_block(x)
            z = x + z
            z = nn.LayerNorm(z.shape[-1])(z)

        return z


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate)
        self.transformer_block = TransformerBlock(config)



    def forward(self, x, temb, conditioner=None):
        assert x.ndim == 3 and temb.ndim == 2
        config = self.config
        temb = temb.unsqueeze(1)
        if conditioner is None:
            conditioner = temb
        else:
            conditioner = torch.cat([conditioner, temb], dim=1)
        x = torch.cat([conditioner, x], dim=1)
        # könnte noch in init
        pos_embed = nn.Parameter(
            torch.empty(1, x.size(1), x.size(2)).uniform_(), requires_grad=True
        )
        x = x + pos_embed
        x = self.dropout(x)
        for layer_idx in range(config.num_layers):
            x = self.transformer_block(x, masks=None)
        x = x[:, 1:]
        return x


class MaskedTransformer(nn.Module):
    """Masked transformer."""

    def __init__(self, config):
        super(MaskedTransformer, self).__init__()
        self.config = config
        self.trans_encoder = TransformerEncoder(config)

        if config.readout == "mlp":
            self.model = MLP([2 * config.embed_dim, config.vocab_size], activation=nn.functional.gelu)
        elif config.readout == "resnet":
            self.model = ResidualReadout(config)
        else:
            raise ValueError("Unknown readout type %s" % config.readout)

    def forward(self, x, temb, pos):
        config = self.config
        embed = self.trans_encoder(x, temb)
        embed = embed[:, pos].unsqueeze(1)
        if config.readout == "mlp":
            logits = self.model(embed)
        elif config.readout == "resnet":
            logits = self.model(embed, temb)

        return logits


class UniDirectionalTransformer(nn.Module):
    """Transformer in one direction."""

    def __init__(self, config, direction):
        super(UniDirectionalTransformer, self).__init__()
        self.config = config
        self.direction = direction
        self.dropout = nn.Dropout(config.dropout_rate)
        self.trans_block = TransformerBlock(config)

    def forward(self, x, temb, conditioner=None):
        assert x.ndim == 3 and temb.ndim == 2
        temb = temb.unsqueeze(1)
        if conditioner is None:
            conditioner = temb
        else:
            conditioner = torch.cat([conditioner, temb], dim=1)
        config = self.config
        cond_dim = conditioner.size(1)
        concat_dim = x.size(1) + cond_dim - 1
        pos_idx = torch_utils.expand_dims(torch.arange(concat_dim, dtype=torch.int32), axis=0)

        # wahrscheinlich falsch
        if self.direction == "l2r":
            x = torch.cat([conditioner, x[:, :-1]], dim=1)
            mask = pos_idx >= pos_idx.transpose(1, 0)
        else:
            x = torch.cat([x[:, 1:], conditioner], dim=1)
            mask = pos_idx <= pos_idx.transpose(1, 0)
        pos_embed = nn.Parameter(init.xavier_uniform_(
            torch.empty(1, concat_dim, x.size(2), dtype=config.dtype)
        ))
        #
        x = x + pos_embed
        x = self.dropout(x)
        for layer_idx in range(config.num_layers):
            x = self.trans_block(x, masks=mask)
        return x
