import os
import tqdm
import math
import joblib
import numpy as np
from urllib.request import urlretrieve
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sei.selene_utils import MemmapGenome
from utils.dna import GenomicSignalFeatures
import pandas as pd


def load_mnist_binarized(root):
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

    x_train, x_valid, x_test = joblib.load(open(dataset, "rb"))
    return x_train, x_valid, x_test


class BinMNIST(Dataset):
    """Binary MNIST dataset"""

    def __init__(self, data, device="cpu", transform=None):
        h, w, c = 28, 28, 1
        self.device = device
        self.data = torch.tensor(data, dtype=torch.float).view(-1, c, h, w)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample.to(self.device)


def get_binmnist_datasets(root, device="cpu"):
    x_train, x_valid, x_test = load_mnist_binarized(root)
    x_train = np.append(x_train, x_valid, axis=0)
    return (
        BinMNIST(x_train, device=device),
        BinMNIST(x_valid, device=device),
        BinMNIST(x_test, device=device),
    )


class TSSDatasetS(Dataset):
    def __init__(
        self, config, seqlength=1024, split="train", n_tsses=100000, rand_offset=0
    ):
        self.shuffle = False

        self.genome = MemmapGenome(
            input_path=config.ref_file,
            memmapfile=config.ref_file_mmap,
            blacklist_regions="hg38",
        )
        self.tfeature = GenomicSignalFeatures(
            config.fantom_files,
            ["cage_plus", "cage_minus"],
            (2000,),
            config.fantom_blacklist_files,
        )

        self.tsses = pd.read_table(config.tsses_file, sep="\t")
        self.tsses = self.tsses.iloc[:n_tsses, :]

        self.chr_lens = self.genome.get_chr_lens()
        self.split = split
        if split == "train":
            self.tsses = self.tsses.iloc[
                ~np.isin(self.tsses["chr"].values, ["chr8", "chr9", "chr10"])
            ]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses["chr"].values, ["chr10"])]
        elif split == "test":
            self.tsses = self.tsses.iloc[
                np.isin(self.tsses["chr"].values, ["chr8", "chr9"])
            ]
        else:
            raise ValueError
        self.rand_offset = rand_offset
        self.seqlength = seqlength

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm, pos, strand = (
            self.tsses["chr"].values[tssi],
            self.tsses["TSS"].values[tssi],
            self.tsses["strand"].values[tssi],
        )
        offset = 1 if strand == "-" else 0

        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)
        seq = self.genome.get_encoding_from_coords(
            chrm,
            pos - int(self.seqlength / 2) + offset,
            pos + int(self.seqlength / 2) + offset,
            strand,
        )

        signal = self.tfeature.get_feature_data(
            chrm,
            pos - int(self.seqlength / 2) + offset,
            pos + int(self.seqlength / 2) + offset,
        )
        if strand == "-":
            signal = signal[::-1, ::-1]
        return np.concatenate([seq, signal.T], axis=-1).astype(np.float32)

    def reset(self):
        np.random.seed(0)
