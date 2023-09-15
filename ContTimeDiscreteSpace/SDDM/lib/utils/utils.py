
"""Utils."""

import functools
from typing import Any
from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


@flax.struct.dataclass
class TrainState:
  step: int
  params: Any
  opt_state: Any
  ema_params: Any


def apply_ema(decay, avg, new):
  return jax.tree_map(lambda a, b: decay * a + (1. - decay) * b, avg, new)


def copy_pytree(pytree):
  return jax.tree_map(jnp.array, pytree)

def init_host_state(params, optimizer):
  state = TrainState(
      step=0,
      params=params,
      opt_state=optimizer.init(params),
      ema_params=copy_pytree(params),
  )
  return jax.device_get(state)

def make_init_params(config, net, global_rng):
    if isinstance(config.discrete_dim, int):
        input_shape = (1, config.discrete_dim)
    else:
        input_shape = [1] + list(config.discrete_dim)
    init_kwargs = dict(
        x=jnp.zeros(input_shape, dtype=jnp.int32),
        t=jnp.zeros((1,), dtype=jnp.float32)
    )
    return net.init({'params': global_rng}, **init_kwargs)['params']

def init_state(config, model, model_key):
    #state = init_host_state(make_init_params(config, model.backwd_model, model_key), model.optimizer)

    if isinstance(config.discrete_dim, int):
        input_shape = (1, config.discrete_dim)
    else:
        input_shape = [1] + list(config.discrete_dim)
    init_kwargs = dict(
        x=jnp.zeros(input_shape, dtype=jnp.int32),
        t=jnp.zeros((1,), dtype=jnp.float32)
    )

    params = model.backwd_model.init({'params': model_key}, **init_kwargs)['params']
    state = TrainState(
      step=0,
      params=params,
      opt_state=model.optimizer.init(params),
      ema_params=copy_pytree(params),
    )

    return jax.device_get(state)




