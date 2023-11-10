import torch
from torch.utils.data import Dataset
from . import dataset_utils
import numpy as np
import torchvision.datasets
import torchvision.transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
import joblib
from urllib.request import urlretrieve
import random
from lib.datasets.maze import maze_gen


@dataset_utils.register_dataset
class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, cfg, device):
        super().__init__(
            root=cfg.data.root, train=cfg.data.train, download=cfg.data.download
        )

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1, 3)
        self.data = self.data.transpose(2, 3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device).view(-1, 3, 32, 32)

        self.random_flips = cfg.data.random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "CIFAR10", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "CIFAR10", "processed")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img


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


@dataset_utils.register_dataset
class DiscreteMNIST(torchvision.datasets.MNIST):
    def __init__(self, cfg, device, root=None):
        super().__init__(root=root, train=cfg.data.train, download=cfg.data.download)
        print("self.data", type(self.data), self.data.shape)
        # self.data = torch.from_numpy(self.data) # (N, H, W, C)
        self.data = self.data.to(device).view(-1, 1, 32, 32)
        # self.data = self.data.transpose(1,3)
        # self.data = self.data.transpose(2,3)

        # self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        # self.data = self.data.to(device).view(-1, 1, 32, 32)

        self.random_flips = cfg.data.use_augm
        if self.random_flips:
            self.flip = torchvision.transforms.RandomRotation((-10, 10))

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img


@dataset_utils.register_dataset
class LakhPianoroll(Dataset):
    def __init__(self, cfg, device):
        S = cfg.data.S
        L = cfg.data.shape[0]
        np_data = np.load(cfg.data.path)  # (N, L) in range [0, S)

        self.data = torch.from_numpy(np_data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


def denormalize_image(image):
    return image * 255


# data
def create_train_discrete_mnist_dataloader(
    root: str,
    image_size: int,
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
    # change path here

    train_dataset = MNIST(
        root=root,
        train=True,
        download=True,
        transform=base_transforms,
    )
    """
    if use_subset:
        subset_size = 5000
        indices = torch.randperm(len(train_dataset))[:subset_size]  # Choose a random subset of specified size
        train_dataset = Subset(train_dataset, indices)
    """

    return train_dataset


def create_discrete_mnist_dataloader(
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    valid_split: float = 0.1,  # fraction of training data used for validation
    use_augmentation: bool = False,
):
    # Define base transformations
    base_transforms = [transforms.Resize((image_size, image_size))]

    # Add augmentations if needed
    if use_augmentation:
        base_transforms.append(transforms.RandomRotation((-10, 10)))

    base_transforms.append(transforms.ToTensor())
    base_transforms.append(denormalize_image)
    base_transforms = transforms.Compose(
        base_transforms
    )  # Add random rotation of 10 degrees

    preprocess = transforms.Compose(base_transforms)

    # Load the training dataset
    train_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/",
        train=True,
        download=True,
        transform=preprocess,
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
        transform=preprocess,
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


def create_train_discrete_cifar10_dataloader(
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

    train_dataset = CIFAR10(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/lib/datasets/CIFAR-10",
        train=True,
        download=True,
        transform=base_transforms,
    )
    """
    if use_subset:
        subset_size = 5000
        indices = torch.randperm(len(train_dataset))[:subset_size]  # Choose a random subset of specified size
        train_dataset = Subset(train_dataset, indices)
    """

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


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
        self.data = torch.tensor(data, device=device, dtype=torch.float).view(
            -1, c, h, w
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, idx


def get_binmnist_datasets(root, device="cpu"):
    x_train, x_valid, x_test = load_mnist_binarized(root)
    x_train = np.append(x_train, x_valid, axis=0)
    return (
        BinMNIST(x_train, device=device),
        BinMNIST(x_valid, device=device),
        BinMNIST(x_test, device=device),
    )


def denormalize_image(image):
    return image * 255


@dataset_utils.register_dataset
class Maze3SComplete(Dataset):
    def __init__(self, cfg, device, _):
        # Wandelt das TensorFlow Dataset in Listen von Bildern und Labels um
        self.device = device
        self.data = maze_gen(
            limit=cfg.data.limit,
            crop=cfg.data.crop_wall,
            dim_x=7,
            dim_y=7,
            pixelSizeOfTile=1,
            weightHigh=97,
            weightLow=97,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataset_utils.register_dataset
class Maze3S(Dataset):
    def __init__(self, cfg, device, _):
        # Wandelt das TensorFlow Dataset in Listen von Bildern und Labels um
        self.cfg = cfg
        self.device = device
        print(device)

    def __len__(self):
        return int(self.cfg.data.batch_size)

    def __getitem__(self, idx):
        self.maze = maze_gen(
            limit=self.cfg.data.limit,
            device=self.device,
            crop=self.cfg.data.crop_wall,
            dim_x=7,
            dim_y=7,
            pixelSizeOfTile=1,
            weightHigh=97,
            weightLow=97,
        )
        return self.maze[0]  # .to(self.device)


#############################################
############ SUDOKU DATASET #################
#############################################
def define_relative_encoding():
    colind = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )

    rowind = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]
    )

    blockind = np.array(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
        ]
    )

    colenc = np.zeros((81, 9))
    rowenc = np.zeros((81, 9))
    blockenc = np.zeros((81, 9))
    colenc[np.arange(81), colind.flatten()] = 1
    rowenc[np.arange(81), rowind.flatten()] = 1
    blockenc[np.arange(81), blockind.flatten()] = 1
    allenc = np.concatenate([colenc, rowenc, blockenc], axis=1)
    return torch.FloatTensor(allenc[:, None, :] == allenc[None, :, :])


def construct_puzzle_solution():
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    while True:
        try:
            puzzle = [[0] * 9 for i in range(9)]  # start with blank puzzle
            rows = [set(range(1, 10)) for i in range(9)]  # set of available
            columns = [set(range(1, 10)) for i in range(9)]  # numbers for each
            squares = [set(range(1, 10)) for i in range(9)]  # row, column and square
            for i in range(9):
                for j in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = (
                        rows[i]
                        .intersection(columns[j])
                        .intersection(squares[(i // 3) * 3 + j // 3])
                    )
                    choice = random.choice(list(choices))

                    puzzle[i][j] = choice

                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i // 3) * 3 + j // 3].discard(choice)

            # success! every cell is filled.
            return puzzle

        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass


def gen_sudoku(num):
    """
    Generates `num` games of Sudoku.
    """
    solutions = np.zeros((num, 9, 9), np.int32)
    for i in range(num):
        solutions[i] = construct_puzzle_solution()

    return solutions


@dataset_utils.register_dataset
class SudokuDataset(Dataset):
    def __init__(self, cfg, device, root=None):
        self.batch_size = cfg.data.batch_size

    def __len__(self):
        return int(self.batch_size * 1000)

    def __getitem__(self, idx):
        sudoku = gen_sudoku(1)
        dataset = np.eye(9)[sudoku.reshape(sudoku.shape[0], -1) - 1]
        return dataset


def sudoku_acc(sample, return_array=False):
    sample = sample.detach().cpu().numpy()
    correct = 0
    total = sample.shape[0]
    ans = sample.argmax(-1) + 1
    numbers_1_N = np.arange(1, 9 + 1)
    corrects = []
    for board in ans:
        if np.all(np.sort(board, axis=1) == numbers_1_N) and np.all(
            np.sort(board.T, axis=1) == numbers_1_N
        ):
            # Check blocks

            blocks = board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
            if np.all(np.sort(board.T, axis=1) == numbers_1_N):
                correct += 1
                corrects.append(True)
            else:
                corrects.append(False)
        else:
            corrects.append(False)

    if return_array:
        return corrects
    else:
        print("correct {} %".format(100 * correct / total))
