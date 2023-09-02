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
from config.train.cifar10 import get_config
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
from lib.datasets.datasets import create_train_discrete_mnist_dataloader, create_train_discrete_cifar10_dataloader


def main():

    cfg = get_config()
    custom_name = None

    preempted_path = Path("null")
    if cfg.saving.enable_preemption_recovery:

        preempted_path = bookkeeping.check_for_preempted_run(cfg.save_location,
            cfg.saving.preemption_start_day_YYYYhyphenMMhyphenDD,
            cfg,
            cfg.saving.prepare_to_resume_after_timeout
        )

    if preempted_path.as_posix() == "null":
        save_dir, checkpoint_dir, config_dir = \
            bookkeeping.create_experiment_folder(
                cfg.save_location,
                cfg.experiment_name if custom_name is None else custom_name,
                custom_name is None
        )
        bookkeeping.save_config_as_yaml(cfg, config_dir)

        # bookkeeping.save_git_hash(save_dir)

    else:
        print("Resuming from preempted run: ", preempted_path)
        save_dir = preempted_path
        checkpoint_dir, config_dir = bookkeeping.create_inner_experiment_folders(save_dir)

    writer = bookkeeping.setup_tensorboard(save_dir, 0)

    device = torch.device(cfg.device)

    model = model_utils.create_model(cfg, device)
    print("number of parameters: ", sum([p.numel() for p in model.parameters()]))

    #dataset = dataset_utils.get_dataset(cfg, device)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle)
    dataloader = create_train_discrete_mnist_dataloader(batch_size=32)
    #dataloader = create_train_discrete_cifar10_dataloader(batch_size=32)
    loss = losses_utils.get_loss(cfg)

    training_step = training_utils.get_train_step(cfg)

    optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)

    state = {
        'model': model,
        'optimizer': optimizer,
        'n_iter': 0
    }


    training_loss = []
    while True:
        #for minibatch, _ in tqdm(dataloader):
        for minibatch in tqdm(dataloader):
            #print("minibatch", minibatch, minibatch.shape)
            l = training_step.step(state, minibatch, loss, writer)
            print('l', l, l.shape)
            training_loss.append(l)


            # just to save model
            if state['n_iter'] % cfg.saving.checkpoint_freq == 0 or state['n_iter'] == cfg.training.n_iters-1:
                bookkeeping.save_checkpoint(checkpoint_dir, state,
                    cfg.saving.num_checkpoints_to_keep)
                

            state['n_iter'] += 1
            if state['n_iter'] > cfg.training.n_iters - 1:
                exit_flag = True
                break

        if exit_flag:
            break

    plt.plot(training_loss)
    plt.title("Training loss")
    plt.savefig("/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/SavedModels/MNIST/PNGs/training_loss.png")
    plt.close()




if __name__ == "__main__":
    main()
