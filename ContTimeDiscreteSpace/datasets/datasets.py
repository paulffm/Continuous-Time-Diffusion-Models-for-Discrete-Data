import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms

def denormalize_image(image):
    return image * 255   

class MazeDataset(Dataset):
    def __init__(self, tf_dataset, image_size):
        # Wandelt das TensorFlow Dataset in Listen von Bildern und Labels um
        self.images = []
        self.labels = []
        self.image_size = image_size

        
        for batch in tf_dataset:
            images, labels = batch
            self.images.extend(images.numpy())
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #image = tf.image.resize(self.images[idx], [self.image_size, self.image_size])
        # Konvertieren Sie das TensorFlow Tensor zu einem Numpy-Array
        #image = image.numpy()
        # Konvertieren Sie das Numpy-Array zu einem PyTorch Tensor und ändern Sie die Dimensionen: TF: HWC, PT: CHW
        image = torch.tensor(self.images[idx]).permute(2, 0, 1).float()
        return image

class BinMazeDataset(Dataset):
    def __init__(self, tf_dataset, image_size):
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

        image[image >= 127.5] = 255
        image[image < 127.5] = 0
        image = image.byte()
        return image
    
#ToDo: load dataset: load dataset with pytorch

# size is important changes pixels
def get_maze_data(config):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "datasets/rectangular_mazes/",
        labels="inferred",
        batch_size=config.data.batch_size,
        image_size=(224,224),
        shuffle=True,
        color_mode='grayscale',
        seed=82,
    )
    torch_ds = MazeDataset(train_ds, config.data.image_size)
    torch_dataloader = DataLoader(torch_ds, batch_size=config.data.batch_size, shuffle=True)

    return torch_dataloader


