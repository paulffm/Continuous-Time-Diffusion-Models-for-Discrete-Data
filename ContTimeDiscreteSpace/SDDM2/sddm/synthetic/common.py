"""Synthetic common."""

import io
import os
from absl import logging
from flax import jax_utils
import jax
import numpy as np
import PIL
from sddm.common import utils
from sddm.synthetic.data import utils as data_utils


def plot(xbin, fn_xbin2float, output_file=None):
  """Visualize binary data."""
  float_data = fn_xbin2float(xbin)
  if output_file is None:  # in-memory plot
    buf = io.BytesIO()
    data_utils.plot_samples(float_data, buf, im_size=4.1, im_fmt='png')
    buf.seek(0)
    image = np.asarray(PIL.Image.open(buf))[None, ...]
    return image
  else:
    with open(output_file, 'wb') as f:
      im_fmt = 'png' if output_file.endswith('.png') else 'pdf'
      data_utils.plot_samples(float_data, f, im_size=4.1, im_fmt=im_fmt)


def fn_plot_data(x_data, config, model, writer):
  x_data = np.reshape(x_data, [-1, config.discrete_dim])
  writer.write_images(0, {'data': model.plot(x_data)})


def model_plot(step, config, writer, model, x0):
  logging.info('num_samples: %d', x0.shape[0])
  writer.write_images(step, {'samples': model.plot(x0)})
  model.plot(x0, os.path.join(config.fig_folder, str(step) + '.png'))
  model.plot(x0, os.path.join(config.fig_folder, str(step) + '.pdf'))


def fn_eval(step, state, rng, sample_fn, config, writer, model):
  step_rng_keys = utils.shard_prng_key(rng)
  x0 = sample_fn(state, step_rng_keys)
  x0 = utils.all_gather(x0)
  if jax.process_index() == 0:
    x0 = jax.device_get(jax_utils.unreplicate(x0))
    x0 = np.reshape(x0, (-1, config.discrete_dim))
    model_plot(step, config, writer, model, x0)


def fn_eval_with_mmd(step, state, rng, eval_mmd, config, writer, model):
  mmd, x0 = eval_mmd(state, rng)
  if jax.process_index() == 0:
    writer.write_scalars(step, {'mmd': mmd})
    model_plot(step, config, writer, model, x0)
  return mmd
