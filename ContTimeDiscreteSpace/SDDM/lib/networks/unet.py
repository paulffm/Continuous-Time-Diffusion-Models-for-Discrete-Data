# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Linen version of unet0."""

from typing import Tuple, Optional

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
from lib.utils import utils

nonlinearity = nn.swish
Normalize = nn.normalization.GroupNorm


def get_timestep_embedding(
    timesteps, embedding_dim, max_time=1000.0, dtype=jnp.float32
):
    """Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
      timesteps: jnp.ndarray: generate embedding vectors at these timesteps
      embedding_dim: int: dimension of the embeddings to generate
      max_time: float: largest time input
      dtype: data type of the generated embeddings

    Returns:
      embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    timesteps *= 1000.0 / max_time

    half_dim = embedding_dim // 2
    emb = onp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def nearest_neighbor_upsample(x):
    batch_size, height, width, channels = x.shape
    x = x.reshape(batch_size, height, 1, width, 1, channels)
    x = jnp.broadcast_to(x, (batch_size, height, 2, width, 2, channels))
    return x.reshape(batch_size, height * 2, width * 2, channels)


class ResnetBlock(nn.Module):
    """Convolutional residual block."""

    dropout: float
    out_ch: Optional[int] = None

    @nn.compact
    def __call__(self, x, *, temb, y, deterministic):
        batch_size, _, _, channels = x.shape

        assert temb.shape[0] == batch_size and len(temb.shape) == 2
        out_ch = channels if self.out_ch is None else self.out_ch

        h = x
        h = nonlinearity(Normalize(name="norm1")(h))
        h = nn.Conv(features=out_ch, kernel_size=(3, 3), strides=(1, 1), name="conv1")(
            h
        )

        # add in timestep embedding
        h += nn.Dense(features=out_ch, name="temb_proj")(nonlinearity(temb))[
            :, None, None, :
        ]

        # add in class information
        if y is not None:
            assert y.ndim == 2 and y.shape[0] == batch_size
            h += nn.Dense(features=out_ch, name="y_proj")(y)[:, None, None, :]

        h = nonlinearity(Normalize(name="norm2")(h))
        # h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        h = nn.Conv(
            features=out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.zeros,
            name="conv2",
        )(h)

        if channels != out_ch:
            x = nn.Dense(features=out_ch, name="nin_shortcut")(x)

        assert x.shape == h.shape
        logging.info(
            "%s: x=%r temb=%r y=%r",
            self.name,
            x.shape,
            temb.shape,
            None if y is None else y.shape,
        )
        return x + h


class AttnBlock(nn.Module):
    """Self-attention residual block."""

    num_heads: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
        assert C % self.num_heads == 0
        head_dim = C // self.num_heads

        h = Normalize(name="norm")(x)

        assert h.shape == (B, H, W, C)
        h = h.reshape(B, H * W, C)
        q = nn.DenseGeneral(features=(self.num_heads, head_dim), name="q")(h)
        k = nn.DenseGeneral(features=(self.num_heads, head_dim), name="k")(h)
        v = nn.DenseGeneral(features=(self.num_heads, head_dim), name="v")(h)
        assert q.shape == k.shape == v.shape == (B, H * W, self.num_heads, head_dim)
        h = nn.dot_product_attention(query=q, key=k, value=v)
        assert h.shape == (B, H * W, self.num_heads, head_dim)
        h = nn.DenseGeneral(
            features=C,
            axis=(-2, -1),
            kernel_init=nn.initializers.zeros,
            name="proj_out",
        )(h)
        assert h.shape == (B, H * W, C)
        h = h.reshape(B, H, W, C)
        assert h.shape == x.shape
        return x + h


def normalize_data(x):
    return x / 127.5 - 1.0


def get_logits_from_logistic_pars(loc, log_scale, num_pixel_vals, eps=1e-6):
    """Computes logits for an underlying logistic distribution."""

    loc = jnp.expand_dims(loc, axis=-1)
    log_scale = jnp.expand_dims(log_scale, axis=-1)

    # Shift log_scale such that if it's zero the probs have a scale
    # that is not too wide and not too narrow either.
    inv_scale = jnp.exp(-(log_scale - 2.0))

    bin_width = 2.0 / (num_pixel_vals - 1.0)
    bin_centers = jnp.linspace(start=-1.0, stop=1.0, num=num_pixel_vals, endpoint=True)

    bin_centers = jnp.expand_dims(bin_centers, axis=tuple(range(0, loc.ndim - 1)))

    bin_centers = bin_centers - loc
    log_cdf_min = jax.nn.log_sigmoid(inv_scale * (bin_centers - 0.5 * bin_width))
    log_cdf_plus = jax.nn.log_sigmoid(inv_scale * (bin_centers + 0.5 * bin_width))

    logits = utils.log_min_exp(log_cdf_plus, log_cdf_min, eps)

    # Normalization:
    # # Option 1:
    # # Assign cdf over range (-\inf, x + 0.5] to pmf for pixel with
    # # value x = 0.
    # logits = logits.at[..., 0].set(log_cdf_plus[..., 0])
    # # Assign cdf over range (x - 0.5, \inf) to pmf for pixel with
    # # value x = 255.
    # log_one_minus_cdf_min = - jax.nn.softplus(
    #     inv_scale * (bin_centers - 0.5 * bin_width))
    # logits = logits.at[..., -1].set(log_one_minus_cdf_min[..., -1])
    # # Option 2:
    # # Alternatively normalize by reweighting all terms. This avoids
    # # sharp peaks at 0 and 255.
    # since we are outputting logits here, we don't need to do anything.
    # they will be normalized by softmax anyway.
    return logits


class UNet(nn.Module):
    """A UNet architecture."""

    shape: Tuple
    num_classes: int = 1
    ch: int = 32
    out_ch: int = 1
    ch_mult: Tuple[int] = (1, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int] = (16,)
    num_heads: int = 1
    dropout: float = 0.1
    model_output: str = "logits"  # 'logits' or 'logistic_pars'
    max_time: float = 1000.0
    num_pixel_vals: int = 256

    @nn.compact
    def __call__(self, x, t):
        assert x.dtype == jnp.int32
        train = True
        y = None
        # print("X shape input unet:", x.shape)
        x = jnp.reshape(x, (x.shape[0], self.shape[0], self.shape[1], self.shape[2]))

        x_onehot = jax.nn.one_hot(x, num_classes=self.num_pixel_vals)
        # Convert to float and scale image to [-1, 1]
        x = normalize_data(x.astype(jnp.float32))

        batch_size, height, width, _ = x.shape
        assert height == width
        assert x.dtype in (jnp.float32, jnp.float64)
        assert t.shape == (batch_size,)  # and t.dtype == jnp.int32
        num_resolutions = len(self.ch_mult)
        ch = self.ch

        # Class embedding
        assert self.num_classes >= 1
        if self.num_classes > 1:
            # logging.info("conditional: num_classes=%d", self.num_classes)
            assert y.shape == (batch_size,) and y.dtype == jnp.int32
            y = jax.nn.one_hot(y, num_classes=self.num_classes, dtype=x.dtype)
            y = nn.Dense(features=ch * 4, name="class_emb")(y)
            assert y.shape == (batch_size, ch * 4)
        else:
            # logging.info("unconditional: num_classes=%d", self.num_classes)
            y = None

        # Timestep embedding
        # logging.info("model max_time: %f", self.max_time)
        temb = get_timestep_embedding(t, ch, max_time=self.max_time)
        temb = nn.Dense(features=ch * 4, name="dense0")(temb)
        temb = nn.Dense(features=ch * 4, name="dense1")(nonlinearity(temb))
        assert temb.shape == (batch_size, ch * 4)

        # Downsampling
        hs = [
            nn.Conv(features=ch, kernel_size=(3, 3), strides=(1, 1), name="conv_in")(x)
        ]
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = ResnetBlock(
                    out_ch=ch * self.ch_mult[i_level],
                    dropout=self.dropout,
                    name=f"down_{i_level}.block_{i_block}",
                )(hs[-1], temb=temb, y=y, deterministic=not train)
                if h.shape[1] in self.attn_resolutions:
                    h = AttnBlock(
                        num_heads=self.num_heads, name=f"down_{i_level}.attn_{i_block}"
                    )(h)
                hs.append(h)
            # Downsample
            if i_level != num_resolutions - 1:
                hs.append(self._downsample(hs[-1], name=f"down_{i_level}.downsample"))

        # Middle
        h = hs[-1]
        h = ResnetBlock(dropout=self.dropout, name="mid.block_1")(
            h, temb=temb, y=y, deterministic=not train
        )
        h = AttnBlock(num_heads=self.num_heads, name="mid.attn_1")(h)
        h = ResnetBlock(dropout=self.dropout, name="mid.block_2")(
            h, temb=temb, y=y, deterministic=not train
        )

        # Upsampling
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks + 1):
                h = ResnetBlock(
                    out_ch=ch * self.ch_mult[i_level],
                    dropout=self.dropout,
                    name=f"up_{i_level}.block_{i_block}",
                )(
                    jnp.concatenate([h, hs.pop()], axis=-1),
                    temb=temb,
                    y=y,
                    deterministic=not train,
                )
                if h.shape[1] in self.attn_resolutions:
                    h = AttnBlock(
                        num_heads=self.num_heads, name=f"up_{i_level}.attn_{i_block}"
                    )(h)
            # Upsample
            if i_level != 0:
                h = self._upsample(h, name=f"up_{i_level}.upsample")
        assert not hs

        # End.
        h = nonlinearity(Normalize(name="norm_out")(h))

        if self.model_output == "logistic_pars":
            # The output represents logits or the log scale and loc of a
            # logistic distribution.
            h = nn.Conv(
                features=self.out_ch * 2,
                kernel_size=(3, 3),
                strides=(1, 1),
                kernel_init=nn.initializers.zeros,
                name="conv_out",
            )(h)
            loc, log_scale = jnp.split(h, 2, axis=-1)

            # ensure loc is between [-1, 1], just like normalized data.
            loc = jnp.tanh(loc + x)
            logits = get_logits_from_logistic_pars(loc, log_scale, self.num_pixel_vals)
            logits = jnp.reshape(logits, (logits.shape[0], -1, logits.shape[-1]))
            return logits

        elif self.model_output == "logits":
            h = nn.Conv(
                features=self.out_ch * self.num_pixel_vals,
                kernel_size=(3, 3),
                strides=(1, 1),
                kernel_init=nn.initializers.zeros,
                name="conv_out",
            )(h)
            h = jnp.reshape(h, (*x.shape[:3], self.out_ch, self.num_pixel_vals))

            h = x_onehot + h
            # print("h shape before reshape output", h.shape)
            h = jnp.reshape(h, (h.shape[0], -1, h.shape[-1]))
            # print("h shape output", h.shape)
            return h

        else:
            raise ValueError(
                f"self.model_output = {self.model_output} but must be "
                "logits or logistic_pars"
            )

    def _downsample(self, x, name):
        batch_size, height, width, channels = x.shape
        x = nn.Conv(features=channels, kernel_size=(3, 3), strides=(2, 2), name=name)(x)
        assert x.shape == (batch_size, height // 2, width // 2, channels)
        return x

    def _upsample(self, x, name):
        batch_size, height, width, channels = x.shape
        x = nearest_neighbor_upsample(x)
        x = nn.Conv(features=channels, kernel_size=(3, 3), strides=(1, 1), name=name)(x)
        assert x.shape == (batch_size, height * 2, width * 2, channels)
        return x
