"""Dump synthetic data into numpy array."""
import ml_collections
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
import tqdm

"""Synthetic data util."""
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
from sympy.combinatorics.graycode import GrayCode
from ml_collections import config_dict
import dataset_utils

def get_config():
    config = config_dict.ConfigDict()
    config.concat_dim = 32
    config.vocab_size = 2
    config.data = data = ml_collections.ConfigDict()
    data.binmode = "gray"
    data.type = "2spirals"
    data.int_scale = -1.0
    data.plot_size = -1.0
    return config

_CONFIG = config_flags.DEFINE_config_file("data_config", lock_config=False)
flags.DEFINE_integer("num_samples", 1000000, "num samples to be generated")
flags.DEFINE_integer("batch_size", 200, "batch size for datagen")
flags.DEFINE_string(
    "data_root",
    "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/TAUnSDDM/lib/datasets/synthetic",
    "root folder of data",
)

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if not os.path.exists(FLAGS.data_root):
        os.makedirs(FLAGS.data_root)
    data_config = get_config()
    db, bm, inv_bm = dataset_utils.setup_data(data_config)

    with open(os.path.join(FLAGS.data_root, "config.yaml"), "w") as f:
        f.write(data_config.to_yaml())
    data_list = []
    for _ in tqdm.tqdm(range(FLAGS.num_samples // FLAGS.batch_size)):
        data = dataset_utils.float2bin(
            db.gen_batch(FLAGS.batch_size),
            bm,
            data_config.concat_dim,
            data_config.data.int_scale,
        )
        data_list.append(data.astype(bool))
    data = np.concatenate(data_list, axis=0) * 1
    print(data.shape[0], "samples generated", data.shape, data[0, :])
    save_path = os.path.join(FLAGS.data_root, f"data_{data_config.data.type}.npy")
    with open(save_path, "wb") as f:
        np.save(f, data)

    with open(os.path.join(FLAGS.data_root, "samples.pdf"), "wb") as f:
        float_data = dataset_utils.bin2float(
            data[:10].astype(np.int32),
            inv_bm,
            data_config.concat_dim,
            data_config.data.int_scale,
        )
        dataset_utils.plot_samples(float_data, f, im_size=4.1, im_fmt="pdf")


if __name__ == "__main__":
    app.run(main)
