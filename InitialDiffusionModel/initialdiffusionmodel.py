import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


# Implementation of the initial paper for Diffusion Models called:
# Deep Unsupervised Learning using Nonequilibrium Thermodynamics


class InitialDiffusionModel():

    def __init__(self, T: int, betas: torch.Tensor, model: nn.Module, dim: int=2):
        """
        Init

        Args:
            T (int): _description_
            betas (torch.Tensor): _description_
            model: neural network that will predict mu and cov for reverse process
            dim: dim of data
        """

        self.T = T
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.model = model
        self.dim = dim

    def forward_process(self, x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward Process 

        Args:
            data (torch.Tensor): _description_
            t (int): _description_
        """

        # Start indexing at 0
        t = t - 1

        mu = torch.sqrt(self.alphas_bar[t]) * x0
        sigma = torch.sqrt(1 - self.alphas_bar[t])
        epsilon = torch.randn_like(x0)
        xt = mu + epsilon * sigma # data ~ N(mu, std)

        # directly return mu_q, sigma_ q
        sigma_q = torch.sqrt((1 - self.alphas_bar[t-1])/ (1 - self.alphas_bar[t]) * self.betas[t])
        m1 = torch.sqrt(self.alphas_bar[t-1]) * self.betas[t] / (1 - self.alphas_bar[t])
        m2 = torch.sqrt(self.alphas[t]) * (1 - self.alphas_bar[t-1]) / (1 - self.alphas_bar[t])
        mu_q = m1 * x0 + m2 * xt

        return mu_q, sigma_q, xt

    def reverse_process(self, xt: torch.Tensor, t:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reverse Process

        Args:
            xt (torch.Tensor): _description_
            t (int): 0 < t <= self.T
        """

        # Start indexing at 0
        t = t - 1
        mu, sigma = self.model(xt, t) 
        epsilon = torch.randn_like(xt)
        return mu, sigma, mu + epsilon * sigma
    
    def sample(self, batch_size: int) -> list:
        """
        Generate samples

        Args:
            batch_size (torch.Tensor): _description_

        Returns:
            list: _description_
        """
        noise = torch.randn((batch_size, self.dim))
        x = noise
        # t can be 0 and t can be t=T
        samples = [x]
        for t in range(self.T, 0, -1):
            # only t > 1: Edge effect
            if not (t == 1):
                _, _, x = self.reverse_process(x, t)
            else:
                x = x
            samples.append(x)
        # index 0 should be the T-th sample 
        return samples[::-1]
    

    def get_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss

        Args:
            x0 (torch.Tensor): _description_
        """
        # x0 [batch_size, self.dim]

        # take random timestep
        t = torch.randint(2, self.T + 1, (1, ))

        # 
        mu_q, sigma_q, xt = self.forward_process(x0, t)
        mu_p, sigma_p, xt_minus1 = self.reverse_process(xt.float(), t)

        # Calculate KL Divergence between two Gaussian
        kl_div = torch.log(sigma_p) - torch.log(sigma_q) + (
            sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
        k = - kl_div.mean() # Should be maximized
        loss = - k # Should be minimized

        # we fix beta_t thus the approximate posterior q has no learnable parameters
        #print(mu_q.requires_grad)
        #print(sigma_q.requires_grad)

        #print(mu_p.requires_grad)
        #print(sigma_p.requires_grad)

        return loss

        

        