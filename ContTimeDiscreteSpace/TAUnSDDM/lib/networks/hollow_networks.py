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

"""
class PositionwiseFeedForward(nn.Module):
"Implements FFN equation."

def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)
    self.activation = GELU()

def forward(self, x):
    return self.w_2(self.dropout(self.activation(self.w_1(x))))
"""


class MLP(nn.Module):
    def __init__(self, features, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.features = features
        self.activation = activation
        # E, MlP
        # Gelu
        # MLP, E

        layers = []
        for i in range(len(features) - 1):
            layers.append(nn.Linear(features[i], features[i + 1]))

            if i != len(features) - 2:  # -2
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

        # if out_dim in TransMLPBlock not None: 2*out_dim instead of self.config.model.embed_dim in MLP
        # but always none
        self.predictor = MLP(
            [2 * self.config.model.embed_dim, config.model.mlp_dim, out_dim],
            activation=nn.GELU(),
        )

    def forward(
        self,
        l2r_embed: TensorType["B", "K", "E"],
        r2l_embed: TensorType["B", "K", "E"],
        _,
    ) -> TensorType["B", "K", "S"]:
        # if read_out_dim None: O = E else O = R2
        state = torch.cat([l2r_embed, r2l_embed], dim=-1)
        logits = self.predictor(state)
        return logits


class ResidualReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ResidualReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.embed_dim = config.model.embed_dim
        self.out_dim = self.readout_dim if self.readout_dim != 0 else self.config.data.S
        self.input_layer = nn.Linear(self.embed_dim, 2 * self.embed_dim)

        # if out_dim in TransMLPBlock not None: 2*out_dim instead of self.config.model.embed_dim in MLP
        # but always none
        self.mlp = MLP(
            [self.embed_dim, self.config.model.mlp_dim, 4 * self.embed_dim],
            activation=nn.GELU(),
        )
        self.resid_layers = []
        self.film_layer = []
        for _ in range(config.model.num_output_ffresiduals):
            self.resid_layers.append(
                MLP(
                    [2 * self.embed_dim, self.config.model.mlp_dim, 2 * self.embed_dim],
                    activation=nn.GELU(),
                )
            )
            self.resid_layers.append(nn.LayerNorm(2 * self.embed_dim))
            self.film_layer.append(nn.Linear(4 * self.embed_dim, 4 * self.embed_dim))
        self.resid_layers = nn.ModuleList(self.resid_layers)
        self.film_layer = nn.ModuleList(self.film_layer)

        self.logits_layer = nn.Linear(2 * self.embed_dim, self.out_dim)

    def forward(
        self, x, temb: TensorType["B", "E"]
    ) -> TensorType["B", "K", "S"]:  # x=state => shape von l2r_embed, r2l_embed
        # x: B, D, E
        temb = self.mlp(temb)  # B, E -> # B, 4 E
        x = self.input_layer(x)
        for i in range(self.config.model.num_output_ffresiduals):
            film_params = self.film_layer[i](temb)  # B, 4E -> B, 4E
            z = self.resid_layers[2 * i](x)  # B, D, 2E -> B, D, 2E
            x = self.resid_layers[2 * i + 1](x + z)
            x = apply_film(
                film_params, x  # B, 4 E -> B, 2 E
            )  # ensure the apply_film function is properly defined
        logits = self.logits_layer(x)
        return logits  # B, D, S


class ConcatResidualReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(ConcatResidualReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.embed_dim = config.model.embed_dim

        self.out_dim = self.readout_dim if self.readout_dim != 0 else self.config.data.S

        self.mlp = MLP(
            [self.embed_dim, self.config.model.mlp_dim, 4 * self.embed_dim],
            activation=nn.GELU(),
        )
        self.resid_layers = []
        self.film_layer = []
        for _ in range(config.model.num_output_ffresiduals):
            self.resid_layers.append(
                MLP(
                    [2 * self.embed_dim, self.config.model.mlp_dim, 2 * self.embed_dim],
                    activation=nn.GELU(),
                )
            )
            self.resid_layers.append(nn.LayerNorm(2 * self.embed_dim))
            self.film_layer.append(nn.Linear(4 * self.embed_dim, 4 * self.embed_dim))
        self.resid_layers = nn.ModuleList(self.resid_layers)
        self.film_layer = nn.ModuleList(self.film_layer)

        self.logits_layer = nn.Linear(2 * self.embed_dim, self.out_dim)

    def forward(
        self,
        l2r_embed: TensorType["B", "K", "E"],
        r2l_embed: TensorType["B", "K", "E"],
        temb: TensorType["B", "E"],
    ) -> TensorType["B", "K", "O"]:
        # out_dim None O = E
        assert (
            l2r_embed.size()[:-1] == r2l_embed.size()[:-1]
        ), "Embeddings must have matching sizes except in the last dimension"
        x = torch.cat([l2r_embed, r2l_embed], dim=-1)  # B, K, 2E

        temb = self.mlp(temb)  # B, 4 E

        for i in range(self.config.model.num_output_ffresiduals):
            film_params = self.film_layer[i](temb)  # B, 4E -> B, 4E
            z = self.resid_layers[i * 2](x)  # B, K, 2E -> B, K, 2E
            x = self.resid_layers[i * 2 + 1](x + z)
            x = apply_film(
                film_params, x  # B, 4 E -> B, 2 E
            )  # ensure the apply_film function is properly defined
        logits = self.logits_layer(x)
        return logits


def transformer_timestep_embedding(
    timesteps, embedding_dim, device="cpu", max_positions=10000
):
    assert embedding_dim % 2 == 0
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)

    return emb


# probably wrong
class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.config = config
        self.device = config.device
        self.num_heads = config.model.num_heads
        self.head_dim = config.model.qkv_dim // config.model.num_heads  #
        self.dense_query = nn.Linear(
            config.model.qkv_dim, config.model.num_heads * self.head_dim, bias=False
        )
        self.dense_key = nn.Linear(config.model.qkv_dim, self.num_heads * self.head_dim)
        self.dense_val = nn.Linear(config.model.qkv_dim, self.num_heads * self.head_dim)

        self.out_linear = nn.Linear(config.model.qkv_dim, config.model.embed_dim)

    def forward(
        self,
        l2r_embed: TensorType["B", "D", "E"],
        r2l_embed: TensorType["B", "D", "E"],
        temb: TensorType["B", "E"],
    ):
        seq_len = l2r_embed.shape[1]
        temb = temb.unsqueeze(1)  # B, 1, D

        query = self.dense_query(l2r_embed + r2l_embed).view(
            l2r_embed.size(0), l2r_embed.size(1), self.num_heads, self.head_dim
        )  # B, D, E => B, D, E
        all_embed = torch.cat([temb, l2r_embed, r2l_embed], dim=1)  # B, 2D + 1, E

        key = self.dense_key(all_embed).view(
            all_embed.size(0), all_embed.size(1), self.num_heads, self.head_dim
        )
        val = self.dense_val(all_embed).view(
            all_embed.size(0), all_embed.size(1), self.num_heads, self.head_dim
        )

        query = query / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
        logits = torch.einsum("bqhd,bkhd->bhqk", query, key)
        # logits = torch.einsum("bqe,bke->bkq", query, key) # without view

        att_l2r_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        att_r2l_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool),
            diagonal=-1,
        ).unsqueeze(0)
        att_t = torch.ones((1, seq_len, 1), device=self.device)

        joint_mask = torch.cat([att_t, att_l2r_mask, att_r2l_mask], dim=-1).unsqueeze(
            0
        )  # 1, 1, seq_len, 2 * seq_len + 1
        attn_weights = torch.where(joint_mask, logits, torch.finfo(torch.float32).min)
        attn_weights = F.softmax(attn_weights, dim=-1)

        x = torch.einsum(
            "bhqk,bkhd->bqhd", attn_weights, val
        )  # B, D, self.num_heads, self.head_dim
        # x = torch.einsum("bkq,bke->bqe", attn_weights, val) # without view
        x = x.view(x.shape[0], x.shape[1], self.num_heads * self.head_dim)
        x = self.out_linear(x)
        return x


class AttentionReadout(nn.Module):
    def __init__(self, config, readout_dim=0):
        super(AttentionReadout, self).__init__()
        self.config = config
        self.readout_dim = readout_dim
        self.cross_attention = CrossAttention(config)
        self.model = ResidualReadout(self.config, self.readout_dim)
        self.ln1 = nn.LayerNorm(config.model.embed_dim)
        self.ln2 = nn.LayerNorm(config.model.embed_dim)  # l2r_embed.shape[-1]

    def forward(self, l2r_embed, r2l_embed, temb):
        inputs = l2r_embed + r2l_embed  # B, D, E
        if self.config.model.transformer_norm_type == "prenorm":
            l2r_embed = self.ln1(l2r_embed)
            r2l_embed = self.ln2(r2l_embed)
            x = self.cross_attention(l2r_embed, r2l_embed, temb)
            x = x + inputs
        elif self.config.model.transformer_norm_type == "postnorm":
            x = self.cross_attention(l2r_embed, r2l_embed, temb)
            x = x + inputs
            x = self.ln1(x)  # adjust based on your requirements
        else:
            raise ValueError(
                "unknown norm type %s" % self.config.model.transformer_norm_type
            )
        return self.model(x, temb)  # B, D, S


class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SelfAttentionBlock, self).__init__()
        self.config = config
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            dropout=config.model.attention_dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.norm = nn.LayerNorm(config.model.embed_dim)

    # input shape == output shape
    def forward(
        self, inputs: TensorType["B", "K", "E"], masks: TensorType["K", "K"]
    ) -> TensorType["B", "K", "E"]:
        # if conditioner None: K=D else K=D+n
        if self.config.model.transformer_norm_type == "prenorm":
            x = self.norm(inputs)
            x, _ = self.self_attention(x, x, x, attn_mask=masks)
            x = self.dropout(x)
            x = x + inputs
        elif self.config.model.transformer_norm_type == "postnorm":
            x, _ = self.self_attention(inputs, inputs, inputs, attn_mask=masks)
            x = self.dropout(x)
            x = x + inputs
            x = self.norm(x)
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
        # Bert: x + Norm -> Sublayer -> Dropout
        # Sublayer: Lin, Lin Dropout, GeLu
        # TMLP: Lin, Relu, Dropout, Linear, Droput
        self.fc1 = nn.Linear(
            embed_dim, mlp_dim, bias=bias_init
        )  # mlp_dim => d_model in TAU
        self.activation = nn.ReLU()  # hier GeLu?
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(
            mlp_dim, self.out_dim if self.out_dim is not None else embed_dim, bias=False
        )
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Apply initializations
        self.kernel_init(self.fc1.weight)
        self.kernel_init(self.fc2.weight)

    def forward(self, inputs: TensorType["B", "K", "E"]) -> TensorType["B", "K", "O"]:
        # if conditioner None: K =D else D+n; if out_dim None: O = E
        # in jax code they do not use out_dim; so O=E
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
            mlp_dim=config.model.mlp_dim,  # make sure to pass the necessary parameters
            dropout_rate=config.model.dropout_rate,
            embed_dim=config.model.embed_dim,
            out_dim=None,  # cfg.model.out_dim
        )
        # if cfg.model.out_dim != None and config.model.transformer_norm_type == "postnorm":
        # self.norm = nn.LayerNorm(config.model.out_dim)
        # else:
        self.norm = nn.LayerNorm(config.model.embed_dim)

    def forward(self, x: TensorType["B", "K", "E"]) -> TensorType["B", "K", "O"]:
        # if conditioner None: K=D else K=D+1; O=E since out_dim =None
        # Bert: x + Norm -> Sublayer -> Dropout
        # Sublayer: Lin, Lin Dropout, GeLu
        if self.config.model.transformer_norm_type == "prenorm":
            z = self.norm(x)
            z = self.mlp(z)
            z = x + z
        elif self.config.model.transformer_norm_type == "postnorm":  # used in ff_
            z = self.mlp(x)
            z = x + z
            z = self.norm(z)
        return z


class TransformerBlock(nn.Module):  #
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.self_attention_block = SelfAttentionBlock(
            config
        )  # do not need K; do not change dim: Output B,K, E
        self.feed_forward_block = FeedForwardBlock(
            config
        )  #  do not need K; Attention; if O !=E

    def forward(
        self, inputs: TensorType["B", "K", "E"], masks: TensorType["K", "K"]
    ) -> TensorType["B", "K", "O"]:
        # if conditioner None: k=D else K=D+n; if out_dim None: O = E => True

        # Bert: Norm -> Sublayer -> Dropout + x
        # Sublayer-MHAtt: 3Lin Layer, Att, Output Lin Layer
        # Sublayer-FF: Lin, Lin Dropout, GeLu
        # Att (Pre): Norm -> Att -> Droput + x
        # TMLP (Pre): Norm, Lin, Relu, Dropout, Linear, Dropout + x

        x = self.self_attention_block(inputs, masks)
        z = self.feed_forward_block(x)
        return z


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.trans_block_layers = []
        for _ in range(config.model.num_layers):
            self.trans_block_layers.append(TransformerBlock(config))
        self.trans_block_layers = nn.ModuleList(self.trans_block_layers)

        self.pos_embed = PositionalEncoding(
            config.device,
            config.model.embed_dim,
            config.model.dropout_rate,
            config.model.concat_dim
            + 1,  # +1 time_emb; need config.model.concat_dim + config.model.cond_dim + 1
        )
        # self.pos_embed = nn.Parameter(torch.nn.init.xavier_uniform_(
        #    torch.empty(1, seq_len, feature_dim)),requires_grad=True)

    def forward(
        self, x: TensorType["B", "D", "E"], temb: TensorType["B", "E"], conditioner=None
    ):
        assert x.ndim == 3 and temb.ndim == 2
        temb = temb.unsqueeze(1)
        if conditioner is None:
            conditioner = temb  # B, 1, E
        else:
            conditioner = torch.cat([conditioner, temb], dim=1)  # B, cond_dim + 1, E
        x = torch.cat([conditioner, x], dim=1)

        x = self.pos_embed(x)
        x = self.dropout(x)
        for trans_block_layers in self.trans_block_layers:
            x = trans_block_layers(x, masks=None)  # B, D + cond_dim +1, E
        x = x[:, 1:]  # B, D + cond_dim, E
        return x


# concat_dim in config, embed_dim in config = D + 1, dytpe
class UniDirectionalTransformer(nn.Module):
    """Transformer in one direction."""

    def __init__(self, config, direction):
        super(UniDirectionalTransformer, self).__init__()
        self.config = config
        self.device = config.device
        self.direction = direction
        self.dropout = nn.Dropout(config.model.dropout_rate)

        self.trans_block_layers = []
        for i in range(config.model.num_layers):
            self.trans_block_layers.append(TransformerBlock(config))
        self.trans_block_layers = nn.ModuleList(self.trans_block_layers)

        # concat_dim in
        # self.pos_embed = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, config.concat_dim, config.embed_dim, dtype=torch.float32)))
        # if K != D => need to initialize pos_embed with cfg.concat_dim + n
        self.pos_embed = PositionalEncoding(
            config.device,
            config.model.embed_dim,
            config.model.dropout_rate,
            config.model.concat_dim,
        )

    def forward(
        self, x: TensorType["B", "D", "E"], temb: TensorType["B", "E"], conditioner=None
    ) -> TensorType["B", "K", "O"]:
        # conditioner None: K = E else K = D + cond_dim - 1; out_dim always none in TransMLP: O = E
        assert x.ndim == 3 and temb.ndim == 2  # B, D, E and B, E
        temb = temb.unsqueeze(1)
        if conditioner is None:
            conditioner = temb  # B, 1, E; B, 1, T
        else:
            conditioner = torch.cat([conditioner, temb], dim=1)  # B, 2, E
        # eventuell
        cond_dim = conditioner.size(1)  # 1 if None, else 2
        concat_dim = (
            x.size(1) + cond_dim - 1
        )  # D if None, else D+1 # mabye if: D+1 => I need in

        if self.direction == "l2r":
            x = torch.cat(
                [conditioner, x[:, :-1]], dim=1
            )  # x[:, :-1] B, D-1, E; condtioner B, 1, E => B, D, E; => if temb not B, E => would not work
            mask = torch.triu(
                torch.ones(
                    (concat_dim, concat_dim), device=self.device, dtype=torch.bool
                ),
                diagonal=1,  # concat_dim = D; if conditioner:
            )  # right mask

        else:
            x = torch.cat([x[:, 1:], conditioner], dim=1)  # B, D-1, E + B, 1or2, E
            mask = torch.tril(
                torch.ones(
                    (concat_dim, concat_dim), device=self.device, dtype=torch.bool
                ),
                diagonal=-1,
            )  # right mask

        # if K != D => need to initialize pos_embed with cfg.concat_dim + n
        x = self.pos_embed(x)

        x = self.dropout(x)
        # x: (B, D, E) or with conditioner: D + cond_dim - 1, E) true
        # x now: (B, K, E)
        for trans_block_layers in self.trans_block_layers:
            x = trans_block_layers(x, masks=mask)

        return x  # x now: B, K,E


def normalize_input(x, S):
    x = x / (S - 1)  # (0, 1)
    x = x * 2 - 1  # (-1, 1)
    return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, config, readout_dim=None):
        super(BidirectionalTransformer, self).__init__()
        self.config = config
        self.device = config.device
        self.S = config.data.S
        self.embed_dim = config.model.embed_dim
        self.mlp_dim = config.model.mlp_dim
        self.embedding = nn.Embedding(
            self.S, config.model.embed_dim
        )  # B, D with values to  B, D, E
        self.temb_scale = config.model.time_scale_factor
        self.use_cat = config.model.use_cat
        self.use_one_hot_input = config.model.use_one_hot_input

        if self.config.model.net_arch == "bidir_transformer":
            self.module_l2r = UniDirectionalTransformer(self.config, "l2r")
            self.module_r2l = UniDirectionalTransformer(self.config, "r2l")
        else:
            raise ValueError("Unknown net_arch: %s" % self.config.model.net_arch)

        if readout_dim is None:
            readout_dim = self.S

        self.readout_dim = readout_dim

        if self.config.model.bidir_readout == "concat":
            self.readout_module = ConcatReadout(self.config, readout_dim=readout_dim)
        elif self.config.model.bidir_readout == "res_concat":
            self.readout_module = ConcatResidualReadout(
                self.config, readout_dim=readout_dim
            )
        elif self.config.model.bidir_readout == "attention":
            self.readout_module = AttentionReadout(self.config, readout_dim=readout_dim)
        else:
            raise ValueError(
                "Unknown bidir_readout: %s" % self.config.model.bidir_readout
            )

        if self.use_cat:
            if self.use_one_hot_input:
                self.input_embedding = nn.Linear(self.S, self.embed_dim)
            else:
                self.input_embedding = nn.Embedding(self.S, self.embed_dim)
        else:
            # self.input_embedding = nn.Embedding(self.S, config.embed_dim)
            # if i normalize i cant use embedding
            # transformiert die Eingabewerte durch eine gewichtete Summe (plus einem Bias),
            self.input_embedding = nn.Linear(1, self.embed_dim)

        # need explicitly B, E
        self.temb_net = nn.Sequential(
            nn.Linear(int(self.embed_dim / 2), self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.embed_dim),
        )

    def forward(
        self, x: TensorType["B", "D"], t: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        temb = self.temb_net(
            transformer_timestep_embedding(
                t * self.temb_scale, int(self.embed_dim / 2), self.device
            )
        )  # B, E
        # temb = transformer_timestep_embedding(t * self.temb_scale, self.embed_dim)
        B, D = x.shape
        # isrupt ordinality
        if self.use_cat:
            if self.use_one_hot_input:
                x_one_hot = nn.functional.one_hot(x.long(), num_classes=self.S)
                x_embed = self.input_embedding(x_one_hot.float())
            else:
                x_embed = self.input_embedding(x)
        else:
            x = normalize_input(x, self.S)
            x = x.view(B, D, 1)
            x_embed = self.input_embedding(x)

        input_shape = list(x_embed.shape)[:-1]  # (B, D)
        x_embed = x_embed.view(x_embed.shape[0], -1, x_embed.shape[-1])  # B, D, E
        l2r_embed = self.module_l2r(x_embed, temb)
        r2l_embed = self.module_r2l(x_embed, temb)

        logits = self.readout_module(l2r_embed, r2l_embed, temb)  # B, K, S

        logits = logits.view(input_shape + [self.readout_dim])  # B, D, S
        # logits = logits + x_one_hot
        return logits


class BidirectionalTransformer2(nn.Module):
    def __init__(self, config, readout_dim=None):
        super(BidirectionalTransformer2, self).__init__()
        self.config = config
        self.device = config.device
        self.S = config.data.S
        self.embed_dim = config.model.embed_dim
        self.mlp_dim = config.model.mlp_dim
        self.embedding = nn.Embedding(
            self.S, config.model.embed_dim
        )  # B, D with values to  B, D, E
        self.temb_scale = config.model.time_scale_factor
        self.use_cat = config.model.use_cat
        self.use_one_hot_input = config.model.use_one_hot_input

        if self.config.model.net_arch == "bidir_transformer":
            self.module_l2r = UniDirectionalTransformer(self.config, "l2r")
            self.module_r2l = UniDirectionalTransformer(self.config, "r2l")
        else:
            raise ValueError("Unknown net_arch: %s" % self.config.model.net_arch)

        if readout_dim is None:
            readout_dim = self.S

        self.readout_dim = readout_dim

        if self.config.model.bidir_readout == "concat":
            self.readout_module = ConcatReadout(self.config, readout_dim=readout_dim)
        elif self.config.model.bidir_readout == "res_concat":
            self.readout_module = ConcatResidualReadout(
                self.config, readout_dim=readout_dim
            )
        elif self.config.model.bidir_readout == "attention":
            self.readout_module = AttentionReadout(self.config, readout_dim=readout_dim)
        else:
            raise ValueError(
                "Unknown bidir_readout: %s" % self.config.model.bidir_readout
            )

        if self.use_cat:
            if self.use_one_hot_input:
                self.input_embedding = nn.Linear(self.S, self.embed_dim)
            else:
                self.input_embedding = nn.Embedding(self.S, self.embed_dim)
        else:
            # self.input_embedding = nn.Embedding(self.S, config.embed_dim)
            # if i normalize i cant use embedding
            # transformiert die Eingabewerte durch eine gewichtete Summe (plus einem Bias),
            self.input_embedding = nn.Linear(1, self.embed_dim)

        self.temb_net = nn.Sequential(
            nn.Linear(int(self.embed_dim / 2), self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.embed_dim),
        )

    # Difference: add one hot in the end; time_step_embedding not learned
    def forward(
        self, x: TensorType["B", "D"], t: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        temb = transformer_timestep_embedding(
            t * self.temb_scale, self.embed_dim, device=self.device
        )  # B, E
        B, D = x.shape
        if self.use_cat:
            if self.use_one_hot_input:
                x_one_hot = nn.functional.one_hot(x.long(), num_classes=self.S)
                x_embed = self.input_embedding(x_one_hot.float())
            else:
                x_embed = self.input_embedding(x)
        else:
            x = normalize_input(x, self.S)
            x = x.view(B, D, 1)
            x_embed = self.input_embedding(x)

        input_shape = list(x_embed.shape)[:-1]
        x_embed = x_embed.view(x_embed.shape[0], -1, x_embed.shape[-1])  # B, D, E

        l2r_embed = self.module_l2r(x_embed, temb)
        r2l_embed = self.module_r2l(x_embed, temb)

        logits = self.readout_module(l2r_embed, r2l_embed, temb)

        logits = logits.view(input_shape + [self.readout_dim])  # B, D, S
        # logits = logits + x_one_hot
        return logits


# very similiar to Bert: https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model + MLP or ResNet at the end


class MaskedTransformer(nn.Module):
    """Masked transformer."""

    def __init__(self, config):
        super(MaskedTransformer, self).__init__()
        self.config = config
        self.use_cat = config.model.use_cat
        self.use_one_hot_input = config.model.use_one_hot_input
        self.S = config.data.S
        self.embed_dim = config.model.embed_dim
        # self.embedding = nn.Embedding(config.data.S + 1, config.model.embed_dim)
        self.trans_encoder = TransformerEncoder(config)  # config.model.num_layers = 12

        # I can basically use any neural net
        if config.model.readout == "mlp":
            self.model = MLP(
                [2 * config.model.embed_dim, config.model.mlp_dim, self.config.data.S],
                activation=nn.functional.gelu,
            )
        elif config.model.readout == "resnet":
            self.model = ResidualReadout(
                config
            )  # config.model.num_output_ffresiduals = 2
        else:
            raise ValueError("Unknown readout type %s" % config.model.readout)

        if self.use_cat:
            if self.use_one_hot_input:
                self.input_embedding = nn.Linear(self.S + 1, self.embed_dim)
            else:
                self.input_embedding = nn.Embedding(self.S + 1, self.embed_dim)
        else:
            self.input_embedding = nn.Linear(1, self.embed_dim)

    def forward(self, x, temb, pos):
        B, D = x.shape
        if self.use_cat:
            if self.use_one_hot_input:
                x_one_hot = nn.functional.one_hot(x.long(), num_classes=self.S)
                x = self.input_embedding(x_one_hot.float())
            else:
                x = self.input_embedding(x)
        else:
            x = normalize_input(x, self.S)
            x = x.view(B, D, 1)
            x = self.input_embedding(x)

        embed = self.trans_encoder(x, temb)
        embed = embed[:, pos].unsqueeze(1)  # B, 1, E
        if self.config.model.readout == "mlp":
            logits = self.model(embed)
        elif self.config.model.readout == "resnet":
            logits = self.model(embed, temb)

        return logits  # B, 1, S


class EnumerativeTransformer(nn.Module):
    """
    First embedds input data with Transformer Network by: adding positional encoding and time embedding + general Embedding layer
    Then predicts logits by an arbitrary neural network
    """

    def __init__(self, config):
        super(EnumerativeTransformer, self).__init__()
        self.config = config
        self.device = config.device
        self.S = config.data.S
        self.embed_dim = config.model.embed_dim
        self.temb_scale = config.model.time_scale_factor

        self.transformer = MaskedTransformer(self.config)

    def forward(
        self, x: TensorType["B", "D"], t: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        temb = transformer_timestep_embedding(
            t * self.temb_scale, self.embed_dim, self.device
        )
        x = x.view(x.shape[0], -1)

        prefix_cond = self.config.model.get("conditional_dim", 0)
        positions = torch.arange(prefix_cond, x.shape[1], device=self.device)
        logits_list = []

        for pos in positions:
            x_masked = x.clone()
            x_masked[:, pos] = self.S  # B, D - cond_dim
            logit = self.transformer(x_masked, temb, pos)  # B, 1, S
            logit = logit.squeeze(1)  # B, S
            logits_list.append(logit)  # list with len D of tensors with shape B, S

        logits = torch.stack(logits_list, dim=1)  # B, D, S

        if prefix_cond:
            dummy_logits = torch.zeros(
                [x.shape[0], prefix_cond] + list(logits.shape[2:]), dtype=torch.float32
            )
            logits = torch.cat([dummy_logits, logits], dim=1)
        logits = logits.view(x.shape + (self.S,))
        return logits  # B, D, S


class BertEnumTransformer(nn.Module):
    """
    First embedds input data with Transformer Network by: adding positional encoding and time embedding + general Embedding layer
    Then predicts logits by an arbitrary neural network
    """

    def __init__(self, config):
        super(BertEnumTransformer, self).__init__()
        self.config = config
        self.use_cat = config.model.use_cat
        self.use_one_hot_input = config.model.use_one_hot_input
        self.S = config.data.S
        self.embed_dim = config.model.embed_dim
        self.device = config.device

        self.temb_scale = config.model.time_scale_factor
        # self.embedding = nn.Embedding(config.data.S, config.model.embed_dim)
        self.trans_encoder = TransformerEncoder(config)  # config.model.num_layers = 12

        # I can basically use any neural net
        if config.model.readout == "mlp":
            self.model = MLP(
                [2 * config.model.embed_dim, config.model.mlp_dim, self.config.data.S],
                activation=nn.functional.gelu,
            )
        elif config.model.readout == "resnet":
            self.model = ResidualReadout(
                config
            )  # config.model.num_output_ffresiduals = 2
        else:
            raise ValueError("Unknown readout type %s" % config.model.readout)

        if self.use_cat:
            if self.use_one_hot_input:
                self.input_embedding = nn.Linear(self.S, self.embed_dim)
            else:
                self.input_embedding = nn.Embedding(self.S, self.embed_dim)
        else:
            self.input_embedding = nn.Linear(1, self.embed_dim)

    def forward(
        self, x: TensorType["B", "D"], t: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        temb = transformer_timestep_embedding(
            t * self.temb_scale, self.embed_dim, self.device
        )
        x = x.view(x.shape[0], -1)
        B, D = x.shape

        prefix_cond = self.config.model.get("conditional_dim", 0)

        if self.use_cat:
            if self.use_one_hot_input:
                x_one_hot = nn.functional.one_hot(x.long(), num_classes=self.S)
                x = self.input_embedding(x_one_hot.float())
            else:
                x = self.input_embedding(x)
        else:
            x = normalize_input(x, self.S)
            x = x.view(B, D, 1)
            x = self.input_embedding(x)

        embed = self.trans_encoder(x, temb)

        if self.config.model.readout == "mlp":
            logits = self.model(embed)
        elif self.config.model.readout == "resnet":
            logits = self.model(embed, temb)

        if prefix_cond:
            dummy_logits = torch.zeros(
                [x.shape[0], prefix_cond] + list(logits.shape[2:]), dtype=torch.float32
            )
            logits = torch.cat([dummy_logits, logits], dim=1)
        logits = logits.view(x.shape + (self.S,))
        return logits  # B, D, S


# loss noch anpassen in HollowAux => ConditionalLoss => erben
class PrefixConditionalBidirTransformer(nn.Module):
    def __init__(self, config):
        super(PrefixConditionalBidirTransformer, self).__init__()
        self.config = config
        self.device = config.device
        self.S = config.data.S
        self.embed_dim = config.model.embed_dim
        self.mlp_dim = config.model.mlp_dim
        self.embedding = nn.Embedding(
            self.S, self.embed_dim
        )  # B, D with values to  B, D, E
        self.temb_scale = config.model.time_scale_factor
        self.use_cat = config.model.use_cat
        self.use_one_hot_input = config.model.use_one_hot_input
        self.conditional_dim = self.config.model.get(
            "conditional_dim", 0
        )  # config.conditional_dim

        if self.config.model.net_arch == "bidir_transformer":
            self.module_l2r = UniDirectionalTransformer(self.config, "l2r")
            self.module_r2l = UniDirectionalTransformer(self.config, "r2l")
        else:
            raise ValueError("Unknown net_arch: %s" % self.config.net_arch)

        if readout_dim is None:
            readout_dim = self.S

        self.readout_dim = readout_dim

        if self.config.model.bidir_readout == "concat":
            self.readout_module = ConcatReadout(self.config, readout_dim=readout_dim)
        elif self.config.model.bidir_readout == "res_concat":
            self.readout_module = ConcatResidualReadout(
                self.config, readout_dim=readout_dim
            )
        elif self.config.model.bidir_readout == "attention":
            self.readout_module = AttentionReadout(self.config, readout_dim=readout_dim)
        else:
            raise ValueError(
                "Unknown bidir_readout: %s" % self.config.model.bidir_readout
            )

        if self.use_cat:
            if self.use_one_hot_input:
                self.input_embedding = nn.Linear(self.S, self.embed_dim)
            else:
                self.input_embedding = nn.Embedding(self.S, self.embed_dim)
        else:
            self.input_embedding = nn.Linear(1, self.embed_dim)

        # macht hier keinen sinn, da ich explizit B, E brauche für torch.cat
        # self.temb_net = nn.Sequential(nn.Linear(config.embed_dim, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, 4*temb_dim))
        self.temb_net = nn.Sequential(
            nn.Linear(int(self.embed_dim / 2), self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.embed_dim),
        )

    def forward(self, x, t, conditioner=None):
        temb = self.temb_net(
            transformer_timestep_embedding(
                t * self.temb_scale, int(self.embed_dim / 2), self.device
            )
        )  # B, E
        B, D = x.shape

        if self.use_cat:
            if self.use_one_hot_input:
                x_one_hot = nn.functional.one_hot(x.long(), num_classes=self.S)
                x_embed = self.input_embedding(x_one_hot.float())
            else:
                x_embed = self.input_embedding(x)
        else:
            x = normalize_input(x, self.S)
            x = x.view(B, D, 1)
            x_embed = self.input_embedding(x)

        input_shape = list(x_embed.shape)[:-1]
        x_embed = x_embed.view(x_embed.shape[0], -1, x_embed.shape[-1])  # B, D, E

        # oder direkt conditioner mit geben => y?
        conditioner, x = torch.split(x, [self.conditional_dim], axis=1)  # ohne []

        l2r_embed = self.module_l2r(x_embed, temb, conditioner)[
            :, -x.shape[1] :
        ]  # extract last x.shape[1] dim of prediction
        r2l_embed = self.module_r2l(x_embed, temb, conditioner)[
            :, : x.shape[1]
        ]  # extract first x.shape[1] dim of prediction
        logits = self.readout_module(l2r_embed, r2l_embed, temb)
        # logits = logits.view(input_shape + [self.readout_dim])  # B, D,

        dummy_logits = torch.zeros(
            [x.shape[0], self.conditional_dim] + list(logits.shape[2:]),
            dtype=torch.float32,
        )
        logits = torch.cat([dummy_logits, logits], dim=1)
        assert logits.shape[1] == self.conditional_dim + x.shape[1]
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, device, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x: TensorType["B", "L", "K"]) -> TensorType["B", "L", "K"]:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, 0 : x.size(1), :]
        return self.dropout(x)
