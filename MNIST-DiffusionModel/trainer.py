from diffusion_model import DiffusionModel
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import create_mnist_dataloaders
from ema import EMA
from torch.utils.data import DataLoader
import copy


class Trainer:
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        optimizer: torch.optim,
        use_guided_diff: bool = False,
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
    ):
        self.diffusion_model = diffusion_model
        self.optimizer = optimizer
        self.use_guided_diff = use_guided_diff
        self.use_ema = use_ema
        self.cond_weight = cond_weight
        self.nb_epochs = nb_epochs
        self.image_size = image_size
        self.batch_size = batch_size
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
        self.val_dataloader = val_dataloader

        if not early_stopping:
            self.train_dataloader = create_mnist_dataloaders(
                batch_size=self.batch_size, image_size=self.image_size, num_workers=4
            )
        else:
            self.train_dataloader = train_dataloader

        if self.use_ema:
            self.ema = EMA(beta=0.995)
            self.ema_model = (
                copy.deepcopy(self.diffusion_model.model).eval().requires_grad_(False)
            )

    def train_loop(self):
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
                if self.use_guided_diff:
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
                if step == 700:
                    break

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
                        print("Early stopping due to no improvement.")
                        break

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

    def compute_validation_loss(self):
        # Compute validation loss using your validation data loader
        # Ensure you switch the model to eval mode and use torch.no_grad()
        # You will compute loss similar to your training loop, but without any backward pass or optimizer step
        total_loss = 0
        self.diffusion_model.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                x, y = batch
                
                if self.use_guided_diff:
                    loss = self.diffusion_model(x=x, classes=y, p_uncond=0.1)
                else:
                    loss = self.diffusion_model(x=x, classes=None, p_uncond=0)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_dataloader)
        return avg_loss

    def sampling(
        self,
        epoch: int,
        n_samples: int = 20,
        cond_weight: float = 2.0,
        sample_random: bool = False,
    ) -> None:
        """

        Args:
            n_samples (int): must have a int square root
            epoch (int): just for naming the plots
        """
        self.diffusion_model.model.eval()

        if self.use_guided_diff:
            if sample_random:
                classes_list = np.arange(0, 10)
                classes = torch.from_numpy(np.random.choice(classes_list, n_samples))
            # classes = sampled.cuda()
            else:
                classes = torch.arange(0, 10).to(
                    "cpu"
                )  # context for us just cycles throught the mnist labels
                classes = classes.repeat(int(20 / classes.shape[0]))
            print("We sample the following classes: ")
            print(classes)

            samples = self.diffusion_model.sample(
                n_samples=n_samples, classes=classes, cond_weight=cond_weight
            )
        else:
            samples = self.diffusion_model.sample(
                n_samples=n_samples, classes=None, cond_weight=0
            )

        plt.figure(figsize=(16, 16))
        # int_s2root = int(np.sqrt(n_samples))
        for i in range(n_samples):
            plt.subplot(5, 4, 1 + i)
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

        # to load and train or sample from it:
        # trainer.load(path)
        # trainer.sampling(epoch_name)
        # trainer.train_loop()

        ### load außerhalb einer trainer klasse:
        # Lade das Checkpoint-Dictionary
        # checkpoint_dict = torch.load("path/to/checkpoint.pt")

        # Erstelle das Modell (wobei 'DiffusionModel' die Klasse deines Modells ist)
        # diffusion_model = DiffusionModel()

        # Lade den Zustand des Modells
        # diffusion_model.load_state_dict(checkpoint_dict["diffusion_model_state"])

        # Setze das Modell in den Evaluierungsmodus
        # diffusion_model.eval()

        # Führe das Sampeln durch
        # samples = diffusion_model.sample(n_samples=n_samples, classes=classes, cond_weight=cond_weight)
