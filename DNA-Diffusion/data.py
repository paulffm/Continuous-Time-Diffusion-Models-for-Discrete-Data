import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms 

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
