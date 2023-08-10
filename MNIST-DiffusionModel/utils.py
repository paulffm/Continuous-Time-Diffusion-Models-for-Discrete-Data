import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from einops import rearrange, reduce
from math import pi, sqrt, log
from math import pi, sqrt, log

BITS = 8


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_end: float = 0.02):
    # beta_end = 0.005
    # beta_start = 0.0001
    # beta_end = 0.02
    beta_start = 1e-4

    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int):
    beta_start = 0.001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# helper functions
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def exists(x) -> bool:
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: torch.Tensor) -> torch.Tensor:
    return (t + 1) * 0.5


def l2norm(t):
    return F.normalize(t, dim=-1)

# data 
def create_train_mnist_dataloaders(
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
    base_transforms = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    if use_augmentation:
        base_transforms.insert(0, transforms.RandomRotation((-10, 10)))  

    train_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/MNIST-DiffusionModel",
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

def create_full_mnist_dataloaders(
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    valid_split: float = 0.1,  # fraction of training data used for validation
    use_augmentation: bool = False
):
    # Define base transformations
    base_transforms = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    
    # Add augmentations if needed
    if use_augmentation:
        base_transforms.insert(0, transforms.RandomRotation((-10, 10)))  # Add random rotation of 10 degrees

    preprocess = transforms.Compose(base_transforms)

    # Load the training dataset
    train_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/MNIST-DiffusionModel",
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
        root="/Users/paulheller/PythonRepositories/Master-Thesis/MNIST-DiffusionModel",
        train=False,
        download=True,
        transform=preprocess
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

# for discrete bit diffusion model
def decimal_to_bits(x: torch.Tensor, bits=BITS) -> torch.Tensor:
    """
    expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1
    """

    device = x.device

    # multiplies values between 0 and 1 to between 0 and 255 as integers
    x = (x * 255).int().clamp(0, 255)

    mask = 2 ** torch.arange(
        bits - 1, -1, -1, device=device
    )  # tensor([128,  64,  32,  16,   8,   4,   2,   1])
    mask = rearrange(mask, "d -> d 1 1")  # shape 8, 1, 1
    x = rearrange(
        x, "b c h w -> b c 1 h w"
    ).long()  # shape (B, C, 1, H ,W) long() from me

    bits = ((x & mask) != 0).float()  # binary form of x
    bits = rearrange(bits, "b c d h w -> b (c d) h w")  # shape (B, C, H, W)
    bits = bits * 2 - 1  # scaling from zero and ones to -1 and 1
    return bits


def bits_to_decimal(x: torch.Tensor, bits: int = BITS) -> torch.Tensor:
    """expects bits from -1 to 1, outputs image tensor from 0 to 1"""
    device = x.device

    x = (x > 0).int()  # converts values that are larger than 0 to 1 and otherwise to 0
    mask = 2 ** torch.arange(
        bits - 1, -1, -1, device=device, dtype=torch.int32
    )  # tensor([128,  64,  32,  16,   8,   4,   2,   1],

    mask = rearrange(mask, "d -> d 1 1")  #  torch.Size([8, 1, 1])
    # normalization of 8
    x = rearrange(x, "b (c d) h w -> b c d h w", d=bits)
    #  multipliziert die Eingabetensoren Bit für Bit mit ihren entsprechenden Maskenwerten und
    # summiert dann über die resultierenden Produkte, um die Dezimalzahl für jedes Pixel zu berechnen.
    dec = reduce(x * mask, "b c d h w -> b c h w", "sum")
    # normalization to decimals between 0 and 1
    return (dec / 255).clamp(0.0, 1.0)


# for learned covariance model


def log(t, eps=1e-15):
    return torch.log(t.clamp(min=eps))


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * (x**3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres: float = 0.999):
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1.0 - cdf_min)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -thres,
        log_cdf_plus,
        torch.where(x > thres, log_one_minus_cdf_min, log(cdf_delta)),
    )

    return log_probs
