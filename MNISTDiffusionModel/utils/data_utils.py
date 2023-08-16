import yaml
import torch
from einops import rearrange, reduce
import matplotlib.pyplot as plt
BITS = 8

def save_config_to_yaml(filename: str = "configs/config.yml", **kwargs) -> None:
    """
    Speichert ein gegebenes Konfigurations-Daten-Dictionary in eine YAML-Datei.
    Jedes Schlüsselwort-Argument stellt einen Abschnitt in der YAML-Datei dar.
    """

    # Ein leeres Dictionary, in das die Konfigurationsdaten eingegeben werden.
    config = {}

    # Iteration über jedes Schlüsselwort-Argument
    for key, value in kwargs.items():
        # Ein neuer Abschnitt in der Konfiguration für jedes Schlüsselwort-Argument.
        config[key] = value

    # Speichern des Konfigurations-Dictionarys in eine YAML-Datei.
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def load_config_from_yaml(filename: str) -> dict:
    """
    Lädt eine gegebene Konfigurationsdatei und gibt das daraus resultierende Dictionary zurück.
    """
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return config

def plot_figure(samples, n_samples: int):
    # Helper function for plotting and saving samples

    fig = plt.figure(figsize=(16, 16))  
    # int_s2root = int(np.sqrt(n_samples))
    for i in range(n_samples):
        plt.subplot(5, 4, 1 + i)
        plt.axis("off")
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap="gray")

    return fig 

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