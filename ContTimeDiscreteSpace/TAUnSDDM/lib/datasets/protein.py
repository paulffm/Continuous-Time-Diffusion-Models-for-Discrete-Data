import torch
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
from . import dataset_utils

CHAR2IDX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    '-': 20  
}

IDX2CHAR = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
    10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V',
    20: '-'  
}

def pad_sequence(seq, max_length=48, pad_char='-'):
    return seq.ljust(max_length, pad_char)

def sequence_to_numbers(seq):
    return [CHAR2IDX[char] for char in seq]

def numbers_to_sequence(numbers):
    return [IDX2CHAR[numb] for numb in numbers]

@dataset_utils.register_dataset
class ProteinDataset(Dataset):
    def __init__(self, cfg, device, root):
        seq = np.load('lib/datasets/Protein_sequences/grampa_numarr.npy')
        self.seq = torch.from_numpy(seq).to(device)

    def __len__(self):
        return len(self.seq.shape[0])

    def __getitem__(self, idx):
        return self.seq[idx]