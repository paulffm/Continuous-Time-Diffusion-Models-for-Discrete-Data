import math
import torch
from torch.special import expm1
from einops import rearrange, reduce
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# model
BITS = 8
# convert to bit representations and back


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


def bits_to_decimal(x: torch.Tensor, bits: int = BITS):
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


# bit diffusion class


def log(t: torch.Tensor, eps=1e-20):
    """
    Calculates log: if number in t is smaller than eps, it will return log(eps) for stability reasons
    Args:
        t (_type_): _description_
        eps (_type_, optional): _description_. Defaults to 1e-20.

    Returns:
        _type_: _description_
    """
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x: torch.Tensor, t: torch.Tensor):
    """
    Adds dimension to t such that x and t has the same dimension

    Args:
        x (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def beta_linear_log_snr(t: torch.Tensor):
    """
    expm1 = exp(x) - 1 due to numerical instability

    Args:
        t (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    return -torch.log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t: torch.Tensor, s: float = 0.008):
    return -log(
        (torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5
    )  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


def create_mnist_dataloaders(
    batch_size, image_size=32, num_workers=4, use_subset: bool = False
):
    """
    preprocess=transforms.Compose([transforms.Resize((image_size,image_size)),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]
    
    """
    preprocess = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    train_dataset = MNIST(
        root="/Users/paulheller/PythonRepositories/Master-Thesis/MNIST-DiffusionModel",
        train=True,
        download=True,
        transform=preprocess,
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
