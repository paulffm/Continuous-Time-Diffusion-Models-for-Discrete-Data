from bit_diffusion import BitDiffusion
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import create_mnist_dataloaders
from ema import EMA
from torch.utils.data import DataLoader
import copy


class TrainerBit:
    def __init__(
        self,
        diffusion_model: BitDiffusion,
        optimizer: torch.optim,
        device: str = "cpu",
        use_ema: bool = False,
        nb_epochs: int = 1,
        loss_show_epoch: int = 10,
        sample_epoch: int = 10,
        save_epoch: int = 1,
        image_size: int = 32,
        batch_size: int = 64,
        dataloader: DataLoader = None,
    ):
        self.diffusion_model = diffusion_model
        self.optimizer = optimizer
        self.nb_epochs = nb_epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.loss_show_epoch = loss_show_epoch
        self.sample_epoch = sample_epoch
        self.save_epoch = save_epoch
        self.device = device
        self.use_ema = use_ema

        # no parameters in constructor
        self.start_epoch = 0
        self.dataloader = create_mnist_dataloaders(
            batch_size=self.batch_size, image_size=self.image_size, num_workers=4
        )
        if self.use_ema:
            self.ema = EMA(beta=0.995)
            self.ema_model = (
                copy.deepcopy(self.diffusion_model.model).eval().requires_grad_(False)
            )

    def train_loop(self):
        training_loss = []

        for epoch in range(self.start_epoch, self.nb_epochs):
            self.diffusion_model.model.train()
            print(f"Epoch: {epoch+1}")
            pbar = tqdm(self.dataloader, desc="Training Loop")

            for step, batch in enumerate(pbar):
                loss = self._train_step(batch)
                training_loss.append(loss.item())
                break

            # Saving model
            if (epoch + 1) % self.save_epoch == 0 or (epoch + 1) == self.nb_epochs:
                self.save_model(epoch+1)

            # Plot and Save loss figure
            if (epoch + 1 % self.loss_show_epoch == 0) or (epoch + 1 == self.nb_epochs):
                print(f"Epoch {epoch+1} Loss:", training_loss[-1])
                plt.plot(training_loss)
                plt.title("Training loss")
                plt.savefig(f"PNGs/training_loss{epoch+1}.png")
                plt.close()

            # save generated images
            if (epoch + 1 % self.sample_epoch == 0) or (epoch + 1 == self.nb_epochs):
                self.sampling(epoch+1)

    def _train_step(self, batch: torch.Tensor):
        x, y = batch
        x = x.type(torch.float32).to(self.device)
        y = y.type(torch.long).to(self.device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        # sampling a t to generate t and t+1

        loss = self.diffusion_model(img=x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_ema:
            self.ema.step_ema(self.ema_model, self.diffusion_model.model)

        return loss

    def sampling(self, epoch):
        self.diffusion_model.model.eval()
        n_samples = 1

        samples = self.diffusion_model.sample(batch_size=n_samples)

        plt.figure(figsize=(17, 17))
        for i in range(n_samples):
            plt.subplot(4, 4, 1 + i)
            plt.axis("off")
            plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap="gray")
        plt.savefig(f"PNGS/samples_epoch_{epoch}.png")
        plt.close()
        # hier noch metric

    def save_model(self, epoch):
        checkpoint_dict = {
            "diffusion_model_state": self.diffusion_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.ema_model.state_dict() if self.use_ema else None,
        }
        torch.save(checkpoint_dict, f"checkpoints/epoch_{epoch}_model.pt")

    def load(self, path):
        checkpoint_dict = torch.load(path)
        self.diffusion_model.load_state_dict(checkpoint_dict["diffusion_model_state"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
        self.start_epoch = checkpoint_dict["epoch"]
        if self.use_ema and checkpoint_dict["ema_model"]:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])
