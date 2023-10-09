"""MNIST dataloader."""
import joblib
import numpy as np
from urllib.request import urlretrieve
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import lib.datasets.datasets_utils as datasets_utils
import os

def get_dataloader(config, phase):
    """Get cifar10 data loader."""
    is_training = False

    if phase == "train":
        is_training = True

    dataset = tfds.load(
        "mnist", split=phase, shuffle_files=is_training, data_dir="lib/datasets"
    )
    num_shards = jax.process_count()  # cpu 1
    shard_id = jax.process_index()  # cpu 0
    dataset = dataset.shard(num_shards=num_shards, index=shard_id)

    # repeats dataset
    if is_training:
        dataset = dataset.repeat().shuffle(buffer_size=50000)

    def preprocess(x):
        """Preprocess img."""
        img = tf.cast(x["image"], tf.float32)
        img = tf.image.resize(img, [32, 32])
        aug = None
        if config.data_aug:
            if config.rand_flip:
                augment_img = tf.image.flip_left_right(img)
                aug = tf.random.uniform(shape=[]) > 0.5
                img = tf.where(aug, augment_img, img)
            if config.rot90:
                u = tf.random.uniform(shape=[])
                k = tf.cast(tf.floor(4.0 * u), tf.int32)
                img = tf.image.rot90(img, k=k)
                aug = aug | (k > 0)
        if aug is None:
            aug = tf.convert_to_tensor(False, dtype=tf.bool)
        out = tf.cast(img, tf.int32)
        return out

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # proc_batch_size = datasets_utils.get_per_process_batch_size(config.batch_size)

    # dataset = dataset.batch(proc_batch_size // jax.local_device_count(), drop_remainder=is_training)
    # dataset = dataset.batch(jax.local_device_count(), drop_remainder=is_training)
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# test discrete:
# first_batch = next(iter(train_ds))
# unique_values = np.unique(first_batch)
# print(unique_values)

# "../datasets/"
def load_mnist_binarized(config, root):
    datapath = os.path.join(root, "bin-mnist")
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    dataset = os.path.join(datapath, "mnist.pkl.gz")

    if not os.path.isfile(dataset):
        datafiles = {
            "train": "http://www.cs.toronto.edu/~larocheh/public/"
            "datasets/binarized_mnist/binarized_mnist_train.amat",
            "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
            "binarized_mnist/binarized_mnist_valid.amat",
            "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
            "binarized_mnist/binarized_mnist_test.amat",
        }
        datasplits = {}
        for split in datafiles.keys():
            print("Downloading %s data..." % (split))
            datasplits[split] = np.loadtxt(urlretrieve(datafiles[split])[0])

        joblib.dump(
            [datasplits["train"], datasplits["valid"], datasplits["test"]],
            open(dataset, "wb"),
        )

    dataset, _, _ = joblib.load(open(dataset, "rb"))

    num_shards = jax.process_count()  # cpu 1
    shard_id = jax.process_index()  # cpu 0
    dataset = dataset.shard(num_shards=num_shards, index=shard_id)
    
    is_training = True
    # repeats dataset
    if is_training:
        dataset = dataset.repeat().shuffle(buffer_size=50000)

    def preprocess(x):
        """Preprocess img."""
        img = tf.cast(x["image"], tf.float32)
        img = tf.image.resize(img, [32, 32])
        aug = None
        if config.data_aug:
            if config.rand_flip:
                augment_img = tf.image.flip_left_right(img)
                aug = tf.random.uniform(shape=[]) > 0.5
                img = tf.where(aug, augment_img, img)
            if config.rot90:
                u = tf.random.uniform(shape=[])
                k = tf.cast(tf.floor(4.0 * u), tf.int32)
                img = tf.image.rot90(img, k=k)
                aug = aug | (k > 0)
        if aug is None:
            aug = tf.convert_to_tensor(False, dtype=tf.bool)
        out = tf.cast(img, tf.int32)
        return out

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # proc_batch_size = datasets_utils.get_per_process_batch_size(config.batch_size)

    # dataset = dataset.batch(proc_batch_size // jax.local_device_count(), drop_remainder=is_training)
    # dataset = dataset.batch(jax.local_device_count(), drop_remainder=is_training)
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset