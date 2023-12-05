from sklearn.datasets import make_swiss_roll
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from initialdiffusionmodel import InitialDiffusionModel
import torch.optim
from typing import Tuple
#
def sample_batch(batch_size: int, device='cpu') -> torch.Tensor:
    data, _ = make_swiss_roll(batch_size)
    # Only 2D
    data = data[:, [2, 0]] / 10
    # Flip
    data = data * np.array([1, -1])

    return torch.from_numpy(data).to(device)

def create_3_subplots(x0: torch.Tensor, xT2: torch.Tensor, xT: torch.Tensor, use_forw: bool=True) -> None:
    fontsize = 14

    fig = plt.figure(figsize=(10, 3))
    data = [x0, xT2, xT]
    for i in range(3):
        
        plt.subplot(1, 3, 1+i)
        plt.scatter(data[i][:, 0].data.numpy(), data[i][:, 1].data.numpy(), alpha=0.1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        
        if use_forw:
            if i == 0: plt.ylabel(r'$q(\mathbf{x}^{(0..T)})$', fontsize=fontsize)
        else:
            if i == 0: plt.ylabel(r'$p(\mathbf{x}^{(0..T)})$', fontsize=fontsize)

        if i == 0: plt.title(r'$t=0$', fontsize=fontsize)
        if i == 1: plt.title(r'$t=\frac{T}{2}$', fontsize=fontsize)
        if i == 2: plt.title(r'$t=T$', fontsize=fontsize)
    #plt.savefig('forward_process.png', bbox_inches='tight')
    



def train_loop(diffusion_model: InitialDiffusionModel, optimizer: torch.optim, batch_size: int, nb_epochs: int, device: str='cpu') -> Tuple[InitialDiffusionModel, list]:
    
    training_loss = []
    # tdqm: progress bar
    for epoch in tqdm(range(nb_epochs)):
        x0 = sample_batch(batch_size).to(device)

        loss = diffusion_model.get_loss(x0)

        # set gradient to 0 because pytorch accumulates the gradient
        optimizer.zero_grad()

        # calculates the gradient of the loss function and propgates it backwards through the model
        loss.backward()

        # aktualisiert parameters
        optimizer.step()
        
        training_loss.append(loss.item())
        
        if epoch % 5000 == 0:
            plt.plot(training_loss)
            #plt.savefig(f'figs/training_loss_epoch_{epoch}.png')
            plt.close()
            
            #plot(diffusion_model, f'figs/training_epoch_{epoch}.png', device)
        
    return diffusion_model, training_loss



"""
def forward_process_old(data: np.ndarray, T: int, betas: np.ndarray):

  
    for t in range(T):
        beta_t = betas[t]
        mu = data * torch.sqrt(1- beta_t)
        std = torch.sqrt(beta_t)

        # resampling data: repermization trick: data ~ N(mu, std)
        # sample from q(x_t | x_t-1)
        data = mu + torch.randn_like(data) * std

    # N(0, sigma_1^2) + N(0, sigma_2^2) = N(0, sigma_1^2 + sigma_2^2) 

    return data
"""