import torch
import ml_collections
import yaml
import lib.utils.bookkeeping as bookkeeping
from tqdm import tqdm
#from config.mnist_config.config_bert_mnist import get_config
from config.mnist_config.config_hollow_mnist import get_config
import matplotlib.pyplot as plt
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.mnist as mnist
import lib.datasets.dataset_utils as dataset_utils
import lib.losses.losses as losses
import lib.losses.losses_utils as losses_utils
import lib.training.training as training
import lib.training.training_utils as training_utils
import lib.optimizers.optimizers as optimizers
import lib.optimizers.optimizers_utils as optimizers_utils
import lib.loggers.loggers as loggers
import lib.loggers.logger_utils as logger_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils
import time
from torch.utils.data import DataLoader
import lib.sampling.sampling_utils as sampling_utils
import numpy as np


def main():
    train_resume = True
    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_location = os.path.join(script_dir, 'SavedModels/MNIST/')
    save_location_png = os.path.join(save_location, 'PNGs/')
    dataset_location = os.path.join(script_dir, 'lib/datasets')

    if not train_resume:
        cfg = get_config()
        bookkeeping.save_config(cfg, save_location)

    else:
        date = "2023-12-10"
        config_name = "config_001_hollowMLE.yaml"
        config_path = os.path.join(save_location, date, config_name)
        cfg = bookkeeping.load_config(config_path)

    device = torch.device(cfg.device)

    model = model_utils.create_model(cfg, device)

    optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)

    state = {"model": model, "optimizer": optimizer, "n_iter": 0}

    if train_resume:
        model_name = "model_154999_hollowMLE.pt"
        checkpoint_path = os.path.join(save_location, date, model_name)
        state = bookkeeping.load_state(state, checkpoint_path)
        cfg.training.n_iters = 500000 
        cfg.sampler.sample_freq = 100000000
        cfg.saving.checkpoint_freq = 5000
        cfg.sampler.num_steps = 1000
        bookkeeping.save_config(cfg, save_location)
    
    loss = losses_utils.get_loss(cfg)

    training_step = training_utils.get_train_step(cfg)

    sampler = sampling_utils.get_sampler(cfg)

    dataset = dataset_utils.get_dataset(cfg, device, dataset_location)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle)
    
    print("Info:")
    print("--------------------------------")
    print("State Iter:", state["n_iter"])
    print("--------------------------------")
    print("Name Dataset:", cfg.data.name)
    print("Loss Name:", cfg.loss.name)
    #print("Loss Type: None" if cfg.loss.name == "GenericAux" else f"Loss Type: {cfg.loss.loss_type}")
    #print("Logit Type:", cfg.loss.logit_type)
    #print("Ce_coeff: None" if cfg.loss.name == "GenericAux" else f"Ce_Coeff: {cfg.loss.ce_coeff}")
    print("--------------------------------")
    print("Model Name:", cfg.model.name)
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))
    #print("Net Arch:", cfg.model.net_arch)
    #print("Bidir Readout:None" if cfg.loss.name == "GenericAux" else f"Loss Type: {cfg.model.bidir_readout}")
    print("Sampler:", cfg.sampler.name)

    n_samples = 16

    print("cfg.saving.checkpoint_freq", cfg.saving.checkpoint_freq)
    training_loss = []
    exit_flag = False
    while True:
        for minibatch in tqdm(dataloader):
            minibatch = minibatch.to(device)
            l = training_step.step(state, minibatch, loss)

            training_loss.append(l.item())

            if (state["n_iter"] + 1) % cfg.saving.checkpoint_freq == 0 or state[
                "n_iter"
            ] == cfg.training.n_iters - 1:
                bookkeeping.save_state(state, save_location)
                saving_train_path = os.path.join(
                    save_location_png, f"loss_{cfg.loss.name}{state['n_iter']}.png"
                )
                plt.plot(training_loss)
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.title("Training loss")
                plt.savefig(saving_train_path)
                plt.close()
                print("Model saved in Iteration:", state["n_iter"] + 1)

            if (state["n_iter"] + 1) % cfg.sampler.sample_freq == 0 or state[
                "n_iter"
            ] == cfg.training.n_iters - 1:
                state["model"].eval()
                samples, _ = sampler.sample(state["model"], n_samples)
                samples = samples.reshape(
                    n_samples, 1, cfg.data.image_size, cfg.data.image_size
                )

                state["model"].train()

                fig = plt.figure(figsize=(9, 9))
                for i in range(n_samples):
                    plt.subplot(4, 4, 1 + i)
                    plt.axis("off")
                    plt.imshow(np.transpose(samples[i, ...], (1, 2, 0)), cmap="gray")

                saving_plot_path = os.path.join(
                    save_location_png, f"{cfg.loss.name}{state['n_iter']}_{cfg.sampler.name}{cfg.sampler.num_steps}.png"
                )
                print(saving_plot_path)
                plt.savefig(saving_plot_path)
                plt.close()

            state["n_iter"] += 1
            if state["n_iter"] > cfg.training.n_iters - 1:
                exit_flag = True
                break

        if exit_flag:
            break

    saving_train_path = os.path.join(
        save_location_png, f"loss_{cfg.loss.name}{state['n_iter']}.png"
    )
    plt.plot(training_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Training loss")
    plt.savefig(saving_train_path)
    plt.close()


if __name__ == "__main__":
    main()
