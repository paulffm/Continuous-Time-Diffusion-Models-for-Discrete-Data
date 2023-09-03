"""TF data loader for synthetic data."""

import os
from absl import logging
import jax
import numpy as np
import tensorflow as tf
from sddm.common import utils


def get_dataloader(config):
  """Get synthetic data loader."""
  data_file = os.path.join(config.data_folder, 'data.npy')
  with open(data_file, 'rb') as f:
    data = np.load(f)
    logging.info('data shape: %s', str(data.shape))
  num_shards = jax.process_count()
  shard_id = jax.process_index()
  dataset = tf.data.Dataset.from_tensor_slices(data)
  dataset = dataset.shard(num_shards=num_shards, index=shard_id)
  dataset = dataset.repeat().shuffle(buffer_size=100000,
                                     seed=shard_id)
  dataset = dataset.map(lambda x: tf.cast(x, tf.int32),
                        num_parallel_calls=tf.data.AUTOTUNE)
  proc_batch_size = utils.get_per_process_batch_size(config.batch_size)
  dataset = dataset.batch(proc_batch_size // jax.local_device_count(),
                          drop_remainder=True)
  dataset = dataset.batch(jax.local_device_count(), drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
