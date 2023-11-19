"""Dump synthetic data into numpy array."""
import ml_collections
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
import tqdm
import torch
from torch.utils.data import Dataset
"""Synthetic data util."""
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
from sympy.combinatorics.graycode import GrayCode
from ml_collections import config_dict
from . import dataset_utils

# from config_hollow_synthetic import get_config

##### Synthetic DataSets
def inf_train_gen(data, rng=None, batch_size=200):
    """Sample batch of synthetic data."""
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(
            n_samples=batch_size, factor=0.5, noise=0.08
        )[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for _ in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes * num_per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        # print("x", x, x.shape)
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        raise NotImplementedError


class OnlineToyDataset(object):
    """Wrapper of inf_datagen."""

    def __init__(self, data_name):
        self.dim = 2
        self.data_name = data_name
        self.rng = np.random.RandomState()

        rng = np.random.RandomState(1)
        samples = inf_train_gen(self.data_name, rng, 5000)
        self.f_scale = np.max(np.abs(samples)) + 1
        self.int_scale = 2**15 / (self.f_scale + 1)
        print("f_scale,", self.f_scale, "int_scale,", self.int_scale)

    def gen_batch(self, batch_size):
        return inf_train_gen(self.data_name, self.rng, batch_size)

    def data_gen(self, batch_size):
        while True:
            yield self.gen_batch(batch_size)


def plot_samples(samples, out_name, im_size=0, axis=False, im_fmt=None):
    """Plot samples."""
    plt.scatter(samples[:, 0], samples[:, 1], marker=".")
    plt.axis("equal")
    if im_size > 0:
        plt.xlim(-im_size, im_size)
        plt.ylim(-im_size, im_size)
    if not axis:
        plt.axis("off")
    if isinstance(out_name, str):
        im_fmt = None
    plt.savefig(out_name, bbox_inches="tight", format=im_fmt)
    plt.show()
    plt.close()


def compress(x, discrete_dim):
    bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
    bx = "0" + bx if x >= 0 else "1" + bx
    return bx


def recover(bx):
    x = int(bx[1:], 2)
    return x if bx[0] == "0" else -x


def float2bin(samples, bm, discrete_dim, int_scale):
    bin_list = []
    for i in range(samples.shape[0]):
        x, y = samples[i] * int_scale
        bx, by = compress(x, discrete_dim), compress(y, discrete_dim)
        bx, by = bm[bx], bm[by]
        bin_list.append(np.array(list(bx + by), dtype=int))
    return np.array(bin_list)


def bin2float(samples, inv_bm, discrete_dim, int_scale):
    """Convert binary to float numpy."""
    floats = []
    for i in range(samples.shape[0]):
        s = ""
        for j in range(samples.shape[1]):
            s += str(samples[i, j])
        x, y = s[: discrete_dim // 2], s[discrete_dim // 2 :]
        x, y = inv_bm[x], inv_bm[y]
        x, y = recover(x), recover(y)
        x /= int_scale
        y /= int_scale
        floats.append((x, y))
    return np.array(floats)


def get_binmap(discrete_dim, binmode):
    """Get binary mapping."""
    b = discrete_dim // 2 - 1
    all_bins = []
    for i in range(1 << b):
        bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
        all_bins.append("0" + bx)
        all_bins.append("1" + bx)
    vals = all_bins[:]
    if binmode == "gray":
        print("remapping binary repr with gray code")
        a = GrayCode(b)
        vals = []
        for x in a.generate_gray():
            vals.append("0" + x)
            vals.append("1" + x)
    else:
        assert binmode == "normal"
    bm = {}
    inv_bm = {}
    for i, key in enumerate(all_bins):
        bm[key] = vals[i]
        inv_bm[vals[i]] = key
    return bm, inv_bm


def setup_data(args):
    bm, inv_bm = get_binmap(args.concat_dim, args.data.binmode)
    db = OnlineToyDataset(args.data.type)
    args.data.int_scale = float(db.int_scale)
    args.data.plot_size = float(db.f_scale)
    return db, bm, inv_bm

@dataset_utils.register_dataset
class SyntheticData(Dataset):
    def __init__(self, cfg, device, root):
        with open(root, "rb") as f:
            data = np.load(f)

        self.data = torch.from_numpy(data)

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data_synth = self.data[index]

        return data_synth


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
flags.DEFINE_integer("num_samples", 10000000, "num samples to be generated")
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
    db, bm, inv_bm = setup_data(data_config)

    with open(os.path.join(FLAGS.data_root, "config.yaml"), "w") as f:
        f.write(data_config.to_yaml())
    data_list = []
    for _ in tqdm.tqdm(range(FLAGS.num_samples // FLAGS.batch_size)):
        data = float2bin(
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
        float_data = bin2float(
            data[:10000].astype(np.int32),
            inv_bm,
            data_config.concat_dim,
            data_config.data.int_scale,
        )
        plot_samples(float_data, f, im_size=4.1, im_fmt="pdf")


if __name__ == "__main__":
    app.run(main)
