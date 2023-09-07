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


@dataset_utils.register_dataset
class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, train=cfg.data.train,
            download=cfg.data.download)

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device).view(-1, 3, 32, 32)

        self.random_flips = cfg.data.random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()


    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'processed')

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
class DiscreteMNIST(torchvision.datasets.MNIST):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, train=cfg.data.train,
            download=cfg.data.download)

        self.data = torch.from_numpy(self.data) # (N, H, W, C)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device).view(-1, 1, 32, 32)

        self.random_flips = cfg.data.random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomRotation((-10, 10))


    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

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
        np_data = np.load(cfg.data.path) # (N, L) in range [0, S)

        self.data = torch.from_numpy(np_data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]
    

 
def denormalize_image(image):
    return image * 255   

# data 
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
    base_transforms = transforms.Compose(base_transforms) # Add random rotation of 10 degrees


    train_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/",
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



def create_discrete_mnist_dataloader(
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    valid_split: float = 0.1,  # fraction of training data used for validation
    use_augmentation: bool = False
):
    # Define base transformations
    base_transforms = [transforms.Resize((image_size, image_size))]

    # Add augmentations if needed
    if use_augmentation:
        base_transforms.append(transforms.RandomRotation((-10, 10)))

    base_transforms.append(transforms.ToTensor())
    base_transforms.append(denormalize_image) 
    base_transforms = transforms.Compose(base_transforms) # Add random rotation of 10 degrees

    preprocess = transforms.Compose(base_transforms)

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
    base_transforms = transforms.Compose(base_transforms) # Add random rotation of 10 degrees


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