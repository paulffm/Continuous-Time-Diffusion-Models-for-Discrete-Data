import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools
import torch
import torch.nn as nn
from lib.utils import utils
import torch.nn.functional as F
import torch.nn.init as init
import functorch
from torchtyping import TensorType

class MLP(nn.Module):
    def __init__(self, features, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.features = features
        self.activation = activation

        layers = []
        for i in range(len(features) - 1):
            layers.append(nn.Linear(features[i], features[i + 1]))

            if i != len(features) - 1:
                layers.append(self.activation)

        # layers.append(nn.Linear(features[-1], features[-1]))
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
        out_dim = self.readout_dim if self.readout_dim != 0 else self.config.data.S
        self.predictor = MLP(
            [2 * self.config.embed_dim, config.mlp_dim, out_dim], activation=nn.GELU()
        )

    def forward(self, l2r_embed, r2l_embed, _):
        state = torch.cat([l2r_embed, r2l_embed], dim=-1)
        # out_dim = self.readout_dim if self.readout_dim != 0 else self.config.vocab_size
        # predictor = MLP([2 * self.config.embed_dim, out_dim], activation=nn.GELU())
        logits = self.predictor(state)
        return logits


class ResidualReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ResidualReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.embed_dim = config.embed_dim

        self.out_dim = self.readout_dim if self.readout_dim != 0 else self.config.data.S

        self.mlp = MLP(
            [self.embed_dim, self.config.mlp_dim, 4 * self.embed_dim],
            activation=nn.GELU(),
        )  #
        self.input_layer = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.film_params_layer = nn.Linear(4 * self.embed_dim, 4 * self.embed_dim)
        self.z_layer = MLP(
            [2 * self.embed_dim, self.config.mlp_dim, 2 * self.embed_dim],
            activation=nn.GELU(),
        )
        self.norm_layer = nn.LayerNorm(2 * self.embed_dim)
        self.logits_layer = nn.Linear(2 * self.embed_dim, self.out_dim)

    def forward(self, x, temb):  # x=state => shape von l2r_embed, r2l_embed
        temb = self.mlp(temb)  # B, 4 E

        for _ in range(self.config.num_output_ffresiduals):
            film_params = self.film_params_layer(temb)  # B, 2E
            z = self.z_layer(x)  # B, D, E
            x = self.norm_layer(x + z)
            x = apply_film(
                film_params, x
            )  # ensure the apply_film function is properly defined
        logits = self.logits_layer(x)
        return logits  # B, D, S


class ConcatResidualReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ConcatResidualReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.embed_dim = config.embed_dim

        self.out_dim = self.readout_dim if self.readout_dim != 0 else self.config.data.S

        self.mlp = MLP(
            [self.embed_dim, self.config.mlp_dim, 4 * self.embed_dim],
            activation=nn.GELU(),
        )  #

        self.film_params_layer = nn.Linear(4 * self.embed_dim, 4 * self.embed_dim)
        self.z_layer = MLP(
            [2 * self.embed_dim, self.config.mlp_dim, 2 * self.embed_dim],
            activation=nn.GELU(),
        )
        self.norm_layer = nn.LayerNorm(2 * self.embed_dim)
        self.logits_layer = nn.Linear(2 * self.embed_dim, self.out_dim)
        # self.model = ResidualReadout(self.config, readout_dim=self.readout_dim)

    def forward(
        self, l2r_embed: torch.Tensor, r2l_embed: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        assert (
            l2r_embed.size()[:-1] == r2l_embed.size()[:-1]
        ), "Embeddings must have matching sizes except in the last dimension"
        x = torch.cat([l2r_embed, r2l_embed], dim=-1)  # B, D, 2E

        temb = self.mlp(temb)  # B, 4 E

        for _ in range(self.config.num_output_ffresiduals):
            film_params = self.film_params_layer(temb)  # B, 2E
            z = self.z_layer(x)  # B, D, 2E
            x = self.norm_layer(x + z)
            x = apply_film(
                film_params, x
            )  # ensure the apply_film function is properly defined
        logits = self.logits_layer(x)
        return logits


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


# probably wrong
class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.config = config
        self.head_dim = config.qkv_dim // config.num_heads
        self.dense = nn.Linear(
            config.qkv_dim, config.num_heads * self.head_dim, bias=False
        )

        self.out_linear = nn.Linear(config.qkv_dim, config.embed_dim)

    def forward(self, l2r_embed, r2l_embed, temb):
        seq_len = l2r_embed.shape[1]
        temb = temb.unsqueeze(1)

        query = self.dense(l2r_embed + r2l_embed)
        all_embed = torch.cat([temb, l2r_embed, r2l_embed], dim=1)
        key = self.dense(all_embed)
        val = self.dense(all_embed)

        query = query / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
        logits = torch.einsum("bqhd,bkhd->bhqk", query, key)

        idx = torch.arange(seq_len, dtype=torch.int32)
        att_l2r_mask = torch.ge(idx.unsqueeze(-1), idx.unsqueeze(-2))
        att_r2l_mask = torch.le(idx.unsqueeze(-1), idx.unsqueeze(-2))
        att_t = torch.ones((1, seq_len, 1))
        joint_mask = torch.cat([att_t, att_l2r_mask, att_r2l_mask], dim=-1).unsqueeze(0)
        attn_weights = torch.where(joint_mask, logits, torch.finfo(torch.float32).min)
        attn_weights = F.softmax(attn_weights, dim=-1)

        x = torch.einsum("bhqk,bkhd->bqhd", attn_weights, val)
        x = self.out_linear(x)
        return x


class AttentionReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(AttentionReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.cross_attention = CrossAttention(config)
        self.model = ResidualReadout(self.config, self.readout_dim)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)  # l2r_embed.shape[-1]

    def forward(self, l2r_embed, r2l_embed, temb):
        inputs = l2r_embed + r2l_embed
        if self.config.transformer_norm_type == "prenorm":
            l2r_embed = self.ln1(l2r_embed)
            r2l_embed = self.ln2(r2l_embed)
            x = self.cross_attention(l2r_embed, r2l_embed, temb)
            x = x + inputs
        elif self.config.transformer_norm_type == "postnorm":
            x = self.cross_attention(l2r_embed, r2l_embed, temb)
            x = x + inputs
            x = self.ln1(x)  # adjust based on your requirements
        else:
            raise ValueError("unknown norm type %s" % self.config.transformer_norm_type)
        return self.model(x, temb)  # B, D, S


class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SelfAttentionBlock, self).__init__()
        self.config = config
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,  # make sure to set these parameters according to your needs
            num_heads=config.num_heads,
            dropout=config.attention_dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        # ToDo: LayerNorm
        self.norm = nn.LayerNorm(config.embed_dim)  # adjust the normalization dimension

    # input shape == output shape
    def forward(self, inputs, masks):
        # print("inputs SA", inputs.shape)
        if self.config.transformer_norm_type == "prenorm":
            x = self.norm(inputs)
            x, _ = self.self_attention(
                x, x, x, attn_mask=masks
            )  # adjust the input as needed
            x = self.dropout(x)
            x = x + inputs
        elif self.config.transformer_norm_type == "postnorm":  # used in _sa_block
            x, _ = self.self_attention(inputs, inputs, inputs, attn_mask=masks)
            x = self.dropout(x)
            x = x + inputs  # not in _sa_block
            x = self.norm(x)  # not in _sa_block
        assert inputs.shape == x.shape
        return x


class TransformerMlpBlock(nn.Module):  # directly used in FFResidual in TAU
    def __init__(
        self,
        mlp_dim,
        embed_dim,
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

        if dropout_deterministic:
            torch.backends.cudnn.deterministic = True

        self.kernel_init = (
            kernel_init if kernel_init is not None else nn.init.xavier_uniform_
        )
        bias_init = bias_init if bias_init is not None else nn.init.normal_

        self.fc1 = nn.Linear(
            embed_dim, mlp_dim, bias=bias_init
        )  # mlp_dim => d_model in TAU
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(
            mlp_dim, self.out_dim if self.out_dim is not None else embed_dim, bias=False
        )
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Apply initializations
        self.kernel_init(self.fc1.weight)
        self.kernel_init(self.fc2.weight)

    def forward(self, inputs):
        inputs = inputs.to(self.dtype)
        x = self.fc1(inputs)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, config):
        super(FeedForwardBlock, self).__init__()
        self.config = config
        self.mlp = TransformerMlpBlock(
            mlp_dim=config.mlp_dim,  # make sure to pass the necessary parameters
            dropout_rate=config.dropout_rate,
            embed_dim=config.embed_dim,
            out_dim=None,  # config.out_dim => muss gleich embed_dim sein oder zwischen z = z+x noch linear layer
        )
        # ToDO: dim in nn.LayerNorm
        self.norm = nn.LayerNorm(config.embed_dim)  # adjust the normalization dimension

    def forward(self, x):
        if self.config.transformer_norm_type == "prenorm":
            z = self.norm(x)
            z = self.mlp(z)
            z = x + z
        elif self.config.transformer_norm_type == "postnorm":  # used in ff_
            z = self.mlp(x)
            z = x + z
            z = self.norm(z)
        return z


class TransformerBlock(nn.Module):  #
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.self_attention_block = SelfAttentionBlock(config)

        self.feed_forward_block = FeedForwardBlock(config)

    def forward(self, inputs, masks):
        # inputs = B, D + 1, E
        x = self.self_attention_block(inputs, masks)
        # x shape?
        z = self.feed_forward_block(x)
        # z shape?
        return z


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate)
        self.transformer_block = TransformerBlock(config)
        # self.pos_embed = nn.Parameter(torch.nn.init.xavier_uniform_(
        #    torch.empty(1, seq_len, feature_dim)),requires_grad=True)

    def forward(self, x, temb, conditioner=None):
        assert x.ndim == 3 and temb.ndim == 2
        temb = temb.unsqueeze(1)
        if conditioner is None:
            conditioner = temb
        else:
            conditioner = torch.cat([conditioner, temb], dim=1)  # B, 1, E
        x = torch.cat([conditioner, x], dim=1)

        # ToDO: Positional Embedding

        # x = x + self.pos_embed
        x = self.dropout(x)
        for layer_idx in range(self.config.num_layers):
            x = self.transformer_block(x, masks=None)
        x = x[:, 1:]
        return x


class MaskedTransformer(nn.Module):
    """Masked transformer."""

    def __init__(self, config):
        super(MaskedTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size + 1, config.embed_dim)
        self.trans_encoder = TransformerEncoder(config)

        if config.readout == "mlp":
            self.model = MLP(
                [2 * config.embed_dim, config.mlp_dim, self.config.data.S],
                activation=nn.functional.gelu,
            )
        elif config.readout == "resnet":
            self.model = ResidualReadout(config)
        else:
            raise ValueError("Unknown readout type %s" % config.readout)

    def forward(self, x, temb, pos):
        x = self.embedding(x)
        embed = self.trans_encoder(x, temb)
        embed = embed[:, pos].unsqueeze(1)
        if self.config.readout == "mlp":
            logits = self.model(embed)
        elif self.config.readout == "resnet":
            logits = self.model(embed, temb)

        return logits


# concat_dim in config, embed_dim in config = D + 1, dytpe
class UniDirectionalTransformer(nn.Module):
    """Transformer in one direction."""

    def __init__(self, config, direction):
        super(UniDirectionalTransformer, self).__init__()
        self.config = config
        self.direction = direction
        self.dropout = nn.Dropout(config.dropout_rate)
        self.trans_block = TransformerBlock(config)
        # concat_dim in
        #self.pos_embed = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, config.concat_dim, config.embed_dim, dtype=torch.float32)))
        self.pos_embed = PositionalEncoding('cpu', config.embed_dim, config.dropout_rate, config.concat_dim)

    def forward(self, x, temb, conditioner=None):
        assert x.ndim == 3 and temb.ndim == 2  # B, D, E and B, E
        temb = temb.unsqueeze(1)
        if conditioner is None:
            conditioner = temb  # B, 1, E
        else:
            conditioner = torch.cat([conditioner, temb], dim=1)  # B, 2, E
        # eventuell
        cond_dim = conditioner.size(1)  # 2, 1
        concat_dim = x.size(1) + cond_dim - 1  # D

        if self.direction == "l2r":
            x = torch.cat(
                [conditioner, x[:, :-1]], dim=1
            )  # x[:, :-1] B, D-1, E; condtioner B, 1, E => B, D, E
            mask = torch.triu(torch.ones((concat_dim, concat_dim), dtype=torch.bool), diagonal=1) # right mask


            #mask = torch.tril(torch.ones((concat_dim, concat_dim), dtype=torch.bool), diagonal=-1)
        else:
            x = torch.cat([x[:, 1:], conditioner], dim=1)
            mask = torch.tril(torch.ones((concat_dim, concat_dim), dtype=torch.bool), diagonal=-1) # right mask

            #mask = torch.triu(torch.ones((concat_dim, concat_dim), dtype=torch.bool), diagonal=1)

        # equivalent to Positional encoding: yes d_model =x.size(2) = config.embed_dim
        #x = x + self.pos_embed
        # if use PositionalEncoding
        x = self.pos_embed(x)

        x = self.dropout(x)
        for layer_idx in range(self.config.num_layers):
            x = self.trans_block(x, masks=mask)
        return x

def normalize_input(x, S):
    x = x/S # (0, 1)
    x = x*2 - 1 # (-1, 1)
    return x

class BidirectionalTransformer(nn.Module):
    def __init__(self, config, readout_dim=None):
        super(BidirectionalTransformer, self).__init__()
        self.config = config
        self.S = config.data.S
        self.embed_dim = config.embed_dim
        self.mlp_dim = config.mlp_dim
        self.embedding = nn.Embedding(
            self.S, config.embed_dim
        )  # B, D with values to  B, D, E
        self.temb_scale = config.model.time_scale_factor
        self.use_one_hot = config.model.use_one_hot

        if self.config.net_arch == "bidir_transformer":
            self.module_l2r = UniDirectionalTransformer(self.config, "l2r")
            self.module_r2l = UniDirectionalTransformer(self.config, "r2l")
        # elif self.config.net_arch == "bidir_combiner_transformer":
        #    self.module_l2r = torch_nets.CombinerAxial(self.config, "l2r")
        #    self.module_r2l = torch_nets.CombinerAxial(self.config, "r2l")
        else:
            raise ValueError("Unknown net_arch: %s" % self.config.net_arch)

        if readout_dim is None:
            readout_dim = self.S

        self.readout_dim = readout_dim

        if self.config.bidir_readout == "concat":
            self.readout_module = ConcatReadout(self.config, readout_dim=readout_dim)
        elif self.config.bidir_readout == "res_concat":
            self.readout_module = ConcatResidualReadout(
                self.config, readout_dim=readout_dim
            )
        elif self.config.bidir_readout == "attention":
            self.readout_module = AttentionReadout(self.config, readout_dim=readout_dim)
        else:
            raise ValueError("Unknown bidir_readout: %s" % self.config.bidir_readout)

        if self.config.use_one_hot_input:
            self.input_embedding = nn.Linear(self.S, config.embed_dim)
        else:
            self.input_embedding = nn.Embedding(self.S, config.embed_dim)

        # macht hier keinen sinn, da ich explizit B, E brauche für torch.cat
        # self.temb_net = nn.Sequential(nn.Linear(config.embed_dim, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, 4*temb_dim))
        self.temb_net = nn.Sequential(
            nn.Linear(int(self.embed_dim / 2), self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.embed_dim),
        )

    def forward(self, x, t):
        temb = self.temb_net(transformer_timestep_embedding(t * self.temb_scale, int(self.embed_dim / 2)))  # B, E
        # temb = transformer_timestep_embedding(t * self.temb_scale, self.embed_dim)

        # way to use disrupt ordinality?
        if self.use_one_hot:
            x_one_hot = nn.functional.one_hot(x, num_classes=self.S)
            x_embed = self.input_embedding(x_one_hot)

        else:
            x = normalize_input(x, self.S)
            x_embed = self.embedding(x)

        input_shape = list(x_embed.shape)[:-1]
        x_embed = x_embed.view(x_embed.shape[0], -1, x_embed.shape[-1])  # B, D, E
        # print("x_embed", x_embed.shape)
        l2r_embed = self.module_l2r(x_embed, temb)
        r2l_embed = self.module_r2l(x_embed, temb)  # output shape?
        # print("l2r_embed", l2r_embed.shape)
        # print("r2l_embed", r2l_embed.shape)
        logits = self.readout_module(l2r_embed, r2l_embed, temb)  # resnet output shape?

        logits = logits.view(input_shape + [self.readout_dim])  # B, D, S
        return logits


# still inefficient
class EnumerativeTransformer(nn.Module):
    def __init__(self, config):
        super(EnumerativeTransformer, self).__init__()
        self.config = config
        self.S = config.data.S
        self.embedding = nn.Embedding(self.S, config.embed_dim)
        self.temb_scale = config.time_scale_factor
        self.transformer = MaskedTransformer(self.config)

    def forward(self, x, t):
        temb = transformer_timestep_embedding(
            t * self.temb_scale, self.config.embed_dim
        )
        x = x.view(x.shape[0], -1)

        prefix_cond = self.config.get("conditional_dim", 0)
        positions = torch.arange(prefix_cond, x.shape[1])
        logits_list = []

        for pos in positions:
            x_masked = x.clone()
            x_masked[:, pos] = self.S
            logit = self.transformer(x_masked, temb, pos)
            logit = logit.squeeze(1)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1)
        # End of the loop

        if prefix_cond:
            dummy_logits = torch.zeros(
                [x.shape[0], prefix_cond] + list(logits.shape[2:]), dtype=torch.float32
            )
            logits = torch.cat([dummy_logits, logits], dim=1)
        logits = logits.view(x.shape + (self.S,))

        return logits


class PrefixConditionalBidirTransformer(nn.Module):
    def __init__(self, config):
        super(PrefixConditionalBidirTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.data.S, config.embed_dim)

        if self.config.net_arch == "bidir_transformer":
            self.module_l2r = UniDirectionalTransformer(self.config, "l2r")
            self.module_r2l = UniDirectionalTransformer(self.config, "r2l")
        # elif self.config.net_arch == "bidir_combiner_transformer":
        #    self.module_l2r = CombinerAxial(self.config, "l2r")
        #    self.module_r2l = CombinerAxial(self.config, "r2l")
        else:
            raise ValueError("Unknown net_arch: %s" % self.config.net_arch)

        if self.config.bidir_readout == "concat":
            self.readout_module = ConcatReadout(self.config)
        elif self.config.bidir_readout == "res_concat":
            self.readout_module = ConcatResidualReadout(self.config)
        elif self.config.bidir_readout == "attention":
            self.readout_module = AttentionReadout(self.config)
        else:
            raise ValueError("Unknown bidir_readout: %s" % self.config.bidir_readout)

    def forward(self, x, t):
        temb = transformer_timestep_embedding(
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


class PositionalEncoding(nn.Module):

    def __init__(self, device, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x: TensorType["B", "L", "K"]
    ) -> TensorType["B", "L", "K"]:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, 0:x.size(1), :]
        return self.dropout(x)