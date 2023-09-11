import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 1. Daten mit TensorFlow holen
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../Maze/rectangular_mazes",
    labels="inferred",
    batch_size=64,
    image_size=(28, 28),
    shuffle=True,
    seed=82,
)

class MazeDataset(Dataset):
    def __init__(self, tf_dataset):
        # Wandelt das TensorFlow Dataset in Listen von Bildern und Labels um
        self.images = []
        self.labels = []
        
        for batch in tf_dataset:
            images, labels = batch
            self.images.extend(images.numpy())
            self.labels.extend(labels.numpy())
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).permute(2, 0, 1).float()  # TF: HWC, PT: CHW
        label = torch.tensor(self.labels[idx])
        return image


def get_maze_data(config):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "rectangular_mazes/",
        labels="inferred",
        batch_size=config.data.batch_size,
        image_size=(config.data.image_size, config.data.image_size),
        shuffle=True,
        seed=82,
    )
    torch_ds = MazeDataset(train_ds)
    torch_dataloader = DataLoader(torch_ds, batch_size=config.data.batch_size, shuffle=True)

    return torch_dataloader


