import torch
import torch.nn as nn
import ml_collections
import yaml
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import torch.utils.tensorboard as tensorboard
from tqdm import tqdm
import sys
import signal
import argparse
from config.config_train_sample import get_config
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.datasets as datasets
import lib.datasets.dataset_utils as dataset_utils
import lib.losses.losses as losses
import lib.losses.losses_utils as losses_utils
import lib.training.training as training
import lib.training.training_utils as training_utils
import lib.optimizers.optimizers as optimizers
import lib.optimizers.optimizers_utils as optimizers_utils
import lib.loggers.loggers as loggers
import lib.loggers.logger_utils as logger_utils
from lib.datasets.datasets import (
    create_train_discrete_mnist_dataloader,
    create_train_discrete_cifar10_dataloader,
)

import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils
import numpy as np


def main():
    cfg = get_config()
    custom_name = None

    device = torch.device(cfg.device)
    save_dir, checkpoint_dir, config_dir = bookkeeping.create_experiment_folder(
        cfg.save_location,
        cfg.experiment_name if custom_name is None else custom_name,
        custom_name is None,
    )
    bookkeeping.save_config_as_yaml(cfg, config_dir)

    model = model_utils.create_model(cfg, device)
    print("number of parameters: ", sum([p.numel() for p in model.parameters()]))

    # dataset = dataset_utils.get_dataset(cfg, device)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle)
    dataloader = create_train_discrete_mnist_dataloader(batch_size=32)
    # dataloader = create_train_discrete_cifar10_dataloader(batch_size=32)
    loss = losses_utils.get_loss(cfg)

    training_step = training_utils.get_train_step(cfg)

    optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)

    state = {"model": model, "optimizer": optimizer, "n_iter": 0}

    n_samples = 9

    sampler = sampling_utils.get_sampler(cfg)

    training_loss = []
    exit_flag = False
    while True:
        # for minibatch, _ in tqdm(dataloader):
        for minibatch, _ in tqdm(dataloader):
            # print("minibatch", minibatch, minibatch.shape)
            l = training_step.step(state, minibatch, loss)
            print("Loss:", l.item())
            training_loss.append(l.item())

            # just to save model
            if (
                state["n_iter"] + 1 % cfg.saving.checkpoint_freq == 0
                or state["n_iter"] == cfg.training.n_iters - 1
            ):
                bookkeeping.save_checkpoint(
                    checkpoint_dir, state, cfg.saving.num_checkpoints_to_keep
                )

            if (
                state["n_iter"] + 1 % cfg.sampler.sample_freq == 0
                or state["n_iter"] == cfg.training.n_iters - 1
            ):
                state["model"].eval()
                samples, x_hist, x0_hist = sampler.sample(state["model"], n_samples, 10)
                samples = samples.reshape(n_samples, 1, 32, 32)
                #x_hist = x_hist.reshape(10, n_samples, 1, 32, 32)
                #x0_hist = x0_hist.reshape(10, n_samples, 1, 32, 32)
                state["model"].train()

                fig = plt.figure(figsize=(9, 9))  
                for i in range(n_samples):
                    plt.subplot(3, 3, 1 + i)
                    plt.axis("off")
                    plt.imshow(np.transpose(samples[i, ...], (1,2,0)), cmap="gray")
                

                plt.savefig(f"/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST/samples_epoch_.png")
                #plt.show()
                plt.close()

            state["n_iter"] += 1
            if state["n_iter"] > cfg.training.n_iters - 1:
                exit_flag = True
                break

        if exit_flag:
            break

    plt.plot(training_loss)
    plt.title("Training loss")
    plt.savefig(
        "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST/training_loss.png"
    )
    plt.close()


if __name__ == "__main__":
    main()
