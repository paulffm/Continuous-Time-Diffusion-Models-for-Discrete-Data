import torch.nn as nn
import torch


class MLP(nn.Module):
    # Noise Shape = data shape
    def __init__(self, data_dim: int=2, hidden_dim: int=64, N: int=40) -> None:
        """
        2-Layer MLP

        Args:
            data_dim (int, optional): _description_. Defaults to 2.
            hidden_dim (int, optional): _description_. Defaults to 64.
            N (int, optional): _description_. Defaults to 40.
        """
        super(MLP, self).__init__()
        self.network_head = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        # data_dim *2 => predict mean and cov => cov 2x2?
        # ModuleList: Pytorch will know that there are trainable params
        self.network_tail = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, data_dim * 2), nn.ReLU()) for t in range(N)])
        
    def forward(self, x: torch.Tensor, t: int):
        """_summary_

        Args:
            x (torch.Tensor): _description_
            t (int): _description_

        Returns:
            _type_: _description_
        """
        h = self.network_head(x) # [batch_size, hidden_dim]
        tmp = self.network_tail[t](h) # [batch_size, data_dim * 2]
        # transform one side mean and other side cov
        mu, h = torch.chunk(tmp, 2, dim=1)
        
        # std has to be greater than 0
        var = torch.exp(h)
        sigma = torch.sqrt(var)

        return mu, sigma