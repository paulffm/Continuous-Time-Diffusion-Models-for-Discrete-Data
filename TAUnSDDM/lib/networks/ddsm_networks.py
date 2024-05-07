import numpy as np
import math
from matplotlib import pyplot as plt
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F

import sys 
sys.path.append("../")


def worker_init_fn(worker_id):
    np.random.seed(worker_id)


#############################################
############## SUDOKU MDDEL #################
#############################################

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, bias=None):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", bias)

        self.bias_proj = nn.Linear(bias.shape[-1], n_head)  # T, T, nh

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + self.bias_proj(self.bias).permute((2, 0, 1))
        att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, bias=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, bias=bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
        ))
        m = self.mlp

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_2(x))))
        return x


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, device, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2, device=device) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class SudokuScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, cfg, allenc_relative):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        embed_dim = cfg.model.embed_dim
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim, device=cfg.device),
                                   nn.Linear(embed_dim, embed_dim))

        self.linear = nn.Linear(9, 128)
        self.blocks = nn.ModuleList(Block(128, 8, bias=allenc_relative) for _ in range(20))
        self.denses = nn.ModuleList(Dense(embed_dim, 128) for _ in range(20))
        self.act = NewGELU()
        self.softplus = nn.Softplus()
        self.output = nn.Linear(128, 9)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))

        # Encoding path
        x = x.to(torch.float32)
        h = self.linear(x.view(-1, 81, 9))
        print("h", h.shape)
        for le, ld in zip(self.blocks, self.denses):
            h = le(h + ld(embed)[:, None, :])

        h = self.output(h)

        # h = h.reshape(x.size()) * torch.exp(-t[:,None,None,None]* self.softplus(self.scale)) /  ((x+1e-6)*(1-x+1e-6))
        h = h.reshape(
            x.size())  # * torch.exp(-t[:,None,None,None]* self.softplus(self.scale)) * (1/(x+1e-3)+1/(1-x+1e-3))
        h = h - h.mean(axis=-1, keepdims=True)
        print("h", h.shape)
        return h


class ProteinScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, cfg):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        embed_dim = cfg.model.embed_dim
        self.S = cfg.data.S

        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim, device=cfg.device),
                                   nn.Linear(embed_dim, embed_dim))
        n = embed_dim

        self.linear = nn.Conv1d(self.S, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1, device=cfg.device))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, self.S, kernel_size=1))

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        embed = self.act(self.embed(t / 2))
        

        # Encoding path
        # x: NLC -> NCL
        x = F.one_hot(x.long(), self.S) 
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out.float()))

        # pos encoding
        i = 1
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            #print((out == h).all())
            i +=1
            if h.shape == out.shape:
                out = h + out
            else:
                out = h


        out = self.final(out)

        out = out.permute(0, 2, 1)

        out = out - out.mean(axis=-1, keepdims=True)
        
        return out