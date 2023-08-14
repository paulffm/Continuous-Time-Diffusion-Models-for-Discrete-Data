from models.diffusion_model import DiffusionModel
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataloader import (
    create_train_mnist_dataloaders,
    create_full_mnist_dataloaders,
)
from utils.data_utils import plot_figure
from models.ema import EMA
from torch.utils.data import DataLoader
import copy
import json


class Trainer:
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        optimizer: torch.optim,
        use_cfg: bool = False,
        use_ema: bool = False,
        cond_weight: float = 2.0,
        nb_epochs: int = 1,
        image_size: int = 32,
        batch_size: int = 64,
        loss_show_epoch: int = 1,
        sample_epoch: int = 2,
        save_epoch: int = 1,
        early_stopping: bool = False,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        device: str = "cpu",
        model_name: str = "checkpoint",
    ):
        self.config = {
            "use_cfg": use_cfg,
            "use_ema": use_ema,
            "cond_weight": cond_weight,
            "nb_epochs": nb_epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "loss_show_epoch": loss_show_epoch,
            "sample_epoch": sample_epoch,
            "save_epoch": save_epoch,
            "early_stopping": early_stopping,
            "device": device,
            "model_name": model_name,
        }

        self.diffusion_model = diffusion_model
        self.optimizer = optimizer
        self.use_cfg = use_cfg
        self.use_ema = use_ema
        self.cond_weight = cond_weight

        self.nb_epochs = nb_epochs

        self.loss_show_epoch = loss_show_epoch
        self.sample_epoch = sample_epoch
        self.save_epoch = save_epoch

        self.early_stopping = early_stopping
        self.device = device

        # no parameters in constructor
        self.start_epoch = 0

        # early stopping regarded:
        self.patience = 10
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # not really necessary for now
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.model_name = model_name

        if not early_stopping:
            self.train_dataloader = create_train_mnist_dataloaders(
                batch_size=batch_size,
                image_size=image_size,
                num_workers=4,
                use_augmentation=True,
            )
        else:
            (
                self.train_dataloader,
                self.val_dataloader,
                _,
            ) = create_full_mnist_dataloaders(
                batch_size=batch_size,
                image_size=image_size,
                num_workers=4,
                valid_split=0.1,
                use_augmentation=True,
            )

        if self.use_ema:
            self.ema = EMA(beta=0.995)
            self.ema_model = (
                copy.deepcopy(self.diffusion_model.model).eval().requires_grad_(False)
            )

    def train_loop(self) -> None:
        training_loss = []

        for epoch in range(self.start_epoch, self.nb_epochs):
            epoch_plus1 = epoch + 1
            self.diffusion_model.model.train()
            print(f"Epoch: {epoch_plus1}")
            pbar = tqdm(self.train_dataloader, desc="Training Loop")

            for step, batch in enumerate(pbar):
                x, y = batch
                # x = x.type(torch.float32).to(self.device)
                # y = y.type(torch.long).to(self.device)  # contains 0

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                # sampling a t to generate t and t+1
                if self.use_cfg:
                    loss = self.diffusion_model(x=x, classes=y, p_uncond=0.1)
                else:
                    loss = self.diffusion_model(x=x, classes=None, p_uncond=0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.use_ema:
                    self.ema.step_ema(self.ema_model, self.diffusion_model.model)

                training_loss.append(loss.item())
 
                # early stopping:
                #if step == 700:
                #    break

            # not useful for now
            """
            if self.early_stopping:
                val_loss = self.compute_validation_loss()

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # If patience is exceeded, stop the training
                if self.patience_counter >= self.patience:
                    self.save_model(epoch_plus1)
                    print("Early stopping due to no improvement.")

                    break
            """

            # Saving model
            if (epoch_plus1) % self.save_epoch == 0 or (epoch_plus1) == self.nb_epochs:
                # epoch + 1
                self.save_model(epoch_plus1)

            # Plot and Save loss figure
            if (epoch_plus1 % self.loss_show_epoch == 0) or (
                epoch_plus1 == self.nb_epochs
            ):
                print(f"Epoch {epoch_plus1} Loss:", training_loss[-1])
                plt.plot(training_loss)
                plt.title("Training loss")
                plt.savefig(f"PNGs/training_loss{epoch_plus1}.png")
                plt.close()

            # save generated images
            if (epoch_plus1 % self.sample_epoch == 0) or (
                epoch_plus1 == self.nb_epochs
            ):
                self.sampling(
                    epoch_plus1, cond_weight=self.cond_weight, sample_random=False
                )

    def compute_validation_loss(self) -> float:
        total_loss = 0
        self.diffusion_model.model.eval()  # for smth like dropout and batch norm => deactivate it
        with torch.no_grad():  # no unecessary gradient calculation
            for batch in self.val_dataloader:
                x, y = batch

                if self.use_cfg:
                    loss = self.diffusion_model(x=x, classes=y, p_uncond=0.1)
                else:
                    loss = self.diffusion_model(x=x, classes=None, p_uncond=0)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_dataloader)
        self.diffusion_model.model.train()
        return avg_loss

    def sampling(
        self,
        epoch: int,
        n_samples: int = 10,
        cond_weight: float = 2.0,
        sample_random: bool = False,
    ) -> None:
        """

        Args:
            n_samples (int): must have a int square root
            epoch (int): just for naming the plots
        """
        self.diffusion_model.model.eval()  # also in diffusion_model.sample()

        if self.use_cfg:
            if sample_random:
                classes_list = np.arange(0, 10)
                classes = torch.from_numpy(np.random.choice(classes_list, n_samples))
            # classes = sampled.cuda()
            else:
                classes = torch.arange(0, 10).to(
                    "cpu"
                )  # context for us just cycles throught the mnist labels
                classes = classes.repeat(int(n_samples / classes.shape[0]))
            print("We sample the following classes: ")
            print(classes)

            samples = self.diffusion_model.sample(
                n_samples=n_samples, classes=classes, cond_weight=cond_weight
            )
            if self.use_ema:
                samples_ema = self.diffusion_model.sample(
                    n_samples=n_samples,
                    ema_model=self.ema_model,
                    classes=classes,
                    cond_weight=0,
                )

        else:
            samples = self.diffusion_model.sample(
                n_samples=n_samples, classes=None, cond_weight=0
            )

            if self.use_ema:
                samples_ema = self.diffusion_model.sample(
                    n_samples=n_samples,
                    ema_model=self.ema_model,
                    classes=None,
                    cond_weight=0,
                )

        # Plot
        fig = plot_figure(samples=samples, n_samples=n_samples)
        fig.savefig(f"PNGS/samples_epoch_{epoch}.png")

        if self.use_ema:
            fig_ema = plot_figure(samples=samples_ema, n_samples=n_samples)
            fig_ema.savefig(f"PNGS/samples_ema_epoch_{epoch}.png")

    def save_model(self, epoch) -> None:
        checkpoint_dict = {
            "diffusion_model_state": self.diffusion_model.state_dict(),
            "unet_model_state": self.diffusion_model.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model_state": self.ema_model.state_dict() if self.use_ema else None,
        }

        torch.save(checkpoint_dict, f"checkpoints/{self.model_name}_{epoch}.pth.tar")

    def load(self, path) -> None:
        checkpoint_dict = torch.load(path)
        self.diffusion_model.load_state_dict(checkpoint_dict["diffusion_model_state"])
        self.diffusion_model.model.load_state_dict(checkpoint_dict["unet_model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
        self.start_epoch = checkpoint_dict["epoch"]

        if self.use_ema and checkpoint_dict["ema_model"]:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])

        ### Tutorial: How to sample without trainer class:
        # to load and train or sample from it:
        # trainer.load(path)
        # trainer.sampling(epoch_name)
        # trainer.train_loop()

        ### load outside of existing trainer class:
