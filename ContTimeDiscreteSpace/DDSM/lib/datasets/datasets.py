import os
import tqdm
import math
import joblib
import numpy as np
from urllib.request import urlretrieve
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lib.sei.selene_utils import MemmapGenome
from lib.utils.dna import GenomicSignalFeatures
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.datasets
import torchvision.transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os

# torch.Size([64, 1, 28, 28])
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
            input_path=config.data.ref_file,
            memmapfile=config.data.ref_file_mmap,
            blacklist_regions="hg38",
        )
        self.tfeature = GenomicSignalFeatures(
            config.data.fantom_files,
            ["cage_plus", "cage_minus"],
            (2000,),
            config.fantom_blacklist_files,
        )

        self.tsses = pd.read_table(config.data.tsses_file, sep="\t")
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

def prepare_dna_valid_dataset(config, sei, sei_features):
    valid_set = TSSDatasetS(config, split='valid', n_tsses=40000, rand_offset=0)
    valid_data_loader = DataLoader(valid_set, batch_size=config.data.batch_size, shuffle=False, num_workers=0)
    valid_datasets = []

    for x in valid_data_loader:
        valid_datasets.append(x)

    validseqs = []
    for seq in valid_datasets:
        validseqs.append(seq[:, :, :4])
    validseqs = np.concatenate(validseqs, axis=0)

    with torch.no_grad():
        validseqs_pred = np.zeros((2915, 21907))
        for i in range(int(validseqs.shape[0] / 128)):
            validseq = validseqs[i * 128:(i + 1) * 128]
            validseqs_pred[i * 128:(i + 1) * 128] = sei(
                torch.cat([torch.ones((validseq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(validseq).transpose(1, 2),
                           torch.ones((validseq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
        validseq = validseqs[-128:]
        validseqs_pred[-128:] = sei(
            torch.cat([torch.ones((validseq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(validseq).transpose(1, 2),
                       torch.ones((validseq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
    validseqs_predh3k4me3 = validseqs_pred[:, sei_features[1].str.strip().values == 'H3K4me3'].mean(axis=1)

    print("Validation dataset prepared")
    return valid_datasets, validseqs_predh3k4me3

def denormalize_image(image):
    return image * 255


def create_train_discrete_mnist_dataloader(
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    use_augmentation: bool = False,
) -> DataLoader:
    """
    preprocess=transforms.Compose([transforms.Resize((image_size,image_size)),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]
    
    """
    base_transforms = [transforms.Resize((image_size, image_size))]

    # Add augmentations if needed
    if use_augmentation:
        base_transforms.append(transforms.RandomRotation((-10, 10)))

    base_transforms.append(transforms.ToTensor())
    base_transforms.append(denormalize_image)
    base_transforms = transforms.Compose(
        base_transforms
    )  # Add random rotation of 10 degrees

    train_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/",
        train=True,
        download=True,
        transform=base_transforms,
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def create_discrete_mnist_dataloader(
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    valid_split: float = 0.1,  # fraction of training data used for validation
    use_augmentation: bool = False
):
    # Define base transformations
    base_transforms = [transforms.Resize((image_size, image_size))]
    test_transforms = [transforms.Resize((image_size, image_size))]

    # Add augmentations if needed
    if use_augmentation:
        base_transforms.append(transforms.RandomRotation((-10, 10)))

    base_transforms.append(transforms.ToTensor())
    base_transforms.append(denormalize_image) 
    preprocess = transforms.Compose(base_transforms) # Add random rotation of 10 degrees


    # Load the training dataset
    train_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/",
        train=True,
        download=True,
        transform=preprocess
    )

    # Split the training dataset into training and validation subsets
    num_train = len(train_dataset)
    num_valid = int(valid_split * num_train)
    num_train = num_train - num_valid
    train_subset, valid_subset = random_split(train_dataset, [num_train, num_valid])

    # Load the test dataset
    test_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/",
        train=False,
        download=True,
        transform=preprocess
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
