import torch
import torchvision.transforms as T
import torchvision.datasets
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
TRAINSUBSET = 0



def get_train_data(conf):
    if conf.dataset.name == 'mnist':
        transform = T.Compose(
            [transforms.Resize((conf.dataset.resolution, conf.dataset.resolution)),
                T.ToTensor(),
                lambda x: x * 255
            ]
        )
        transform_test = T.Compose(
            [transforms.Resize((conf.dataset.resolution, conf.dataset.resolution)),
                T.ToTensor(),
                lambda x: x * 255
            ]
        )

        train_set = torchvision.datasets.MNIST(conf.dataset.path,
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
        eval_set = torchvision.datasets.MNIST(conf.dataset.path,
                                                  train=False,
                                                  transform=transform_test,
                                                  download=True)

        if TRAINSUBSET:
            # limit_size = list(range(min(len(train_set), conf.training.dataloader.batch_size*10+1000)))
            limit_size = list(range(128))
            train_set = Subset(train_set, limit_size)
            eval_set = Subset(eval_set, limit_size)

    return train_set, eval_set