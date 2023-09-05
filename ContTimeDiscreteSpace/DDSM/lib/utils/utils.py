import torch

def binary_to_onehot(x):
    xonehot = []
    xonehot.append((x == 1)[..., None])
    xonehot.append((x == 0)[..., None])
    return torch.cat(xonehot, -1)



