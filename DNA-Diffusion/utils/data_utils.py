import numpy as np
from torch.utils.data import Dataset
import torch
import yaml
from einops import rearrange, reduce
import matplotlib.pyplot as plt
from torchvision import transforms 
# raw data encoding:
    # onehot => implemented
    # binary 
    # k mer => implemented

# use pretrained bert

"""
X_train = np.array([one_hot_encode(x, NUCLEOTIDES, 200) for x in tqdm_notebook(raw_dataset['raw_sequence']) if 'N' not in x])
X_train = np.array([x.T.tolist() for x in X_train])
X_train[X_train == 0] = -1
"""
"""
from the following link
https://github.com/jerryji1993/DNABERT/issues/11

import torch
from transformers import BertModel, BertConfig, DNATokenizer

dir_to_pretrained_model = "xxx/xxx"

config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
tokenizer = DNATokenizer.from_pretrained('dna6')
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)

sequence = "AATCTA ATCTAG TCTAGC CTAGCA"
model_input = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=512)["input_ids"]
model_input = torch.tensor(model_input, dtype=torch.long)
model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one

output = model(model_input)"""

def one_hot_encode(seq, nucleotides: list[str], max_seq_len):
    """
    One-hot encode a sequence of nucleotides.
    """
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(nucleotides)))
    for i in range(seq_len):
        seq_array[i, nucleotides.index(seq[i])] = 1
    return seq_array

def encode(seq, nucleotides: list[str]):
    """Encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros(len(nucleotides))
    for i in range(seq_len):
        seq_array[nucleotides.index(seq[i])] = 1

    return seq_array

def convert_to_seq(x, nucleotides: list[str]):
    """
    x = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]*100) => 800 einträge

    x.reshape(4, 200)
    4 x 200 nun und in jeder Spalte steht, das was am wahrscheinlichsten ist:
    [[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, ...],
    [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, ...],
    [0.2, 0.3, 0.4, 0.1, 0.2, 0.3, ...],
    [0.8, 0.7, 0.6, 0.9, 0.8, 0.7, ...]]
    
    Wenn jetzt alphabet = ['A', 'C', 'T' 'G]
    dann ist output: CCCGCC

    Args:
        x (_type_): _description_
        alphabet (_type_): _description_

    Returns:
        _type_: _description_
    """
    return "".join([nucleotides[i] for i in np.argmax(x.reshape(4, 200), axis=0)])

class SequenceDataset(Dataset):
    def __init__(
        self,
        sequence: np.ndarray,
        labels: torch.Tensor,
        transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
    ):
        """
        Initialization
        so würde ich verwenden:
            
        tf = T.Compose([T.ToTensor()])
        seq_dataset = SequenceDataset(seqs=X_train, c=x_train_cell_type, transform=tf)
        train_dl = DataLoader(seq_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        
        """
        self.sequence = sequence
        self.c = labels
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.sequence)

    def __getitem__(self, index):
        "Generates one sample of data: returns data and label"
        # Select sample
        image = self.sequence[index]

        if self.transform:
            x = self.transform(image)
        else:
            x = image

        y = self.c[index]

        return x, y

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

    fig = plt.figure(figsize=(16, 16))  # Erstelle einen Figure-Objekt
    # int_s2root = int(np.sqrt(n_samples))
    for i in range(n_samples):
        plt.subplot(5, 4, 1 + i)
        plt.axis("off")
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap="gray")
    
    # Wenn Sie das Bild sofort in der Funktion anzeigen möchten:

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


"""
import numpy as np
import torch
from data import SequenceDataset
from torchvision import transforms 
from torch.utils.data import DataLoader
x = np.random.randn(200, 4)
y = np.arange(0, 200).reshape(-1, 1)
print(y.shape)
from torchvision.transforms import Lambda

# Definieren einer benutzerdefinierten Transformationsfunktion
tf = transforms.Compose([Lambda(lambda x: torch.tensor(x).float())])
#tf = transforms.Compose([transforms.ToTensor()])
seq_dataset = SequenceDataset(sequence=x, labels=y, transform=tf)
train_dl = DataLoader(seq_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

for step, batch in enumerate(train_dl):
    x_seq, y = batch
    print("x_seq", x_seq.shape) # x_seq torch.Size([16, 4]), las 8,4 
    print("y", y.shape) # y torch.Size([16, 1]), last 8, 1
"""
def kmer2seq(kmers):
    """
    Convert kmers to original sequence
    
    Arguments:
    kmers -- str, kmers separated by space.
    
    Returns:
    seq -- str, original sequence.

    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq

def seq2kmer(seq, k, stride=1):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    if stride > 1:
        seq = seq[:-stride]
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)[::stride]]
    kmers = " ".join(kmer)
    return kmers