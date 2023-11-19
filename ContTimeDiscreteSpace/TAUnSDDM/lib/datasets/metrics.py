import torch
import functools


def binary_hamming_sim(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = torch.sum(torch.abs(x - y), axis=-1)
    return x.shape[-1] - d


def binary_exp_hamming_sim(x, y, bd):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = torch.sum(torch.abs(x - y), axis=-1)
    return torch.exp(-bd * d)


def binary_mmd(x, y, sim_fn):
    """MMD for binary data."""
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    kxx = sim_fn(x, x)
    kxx = kxx * (1 - torch.eye(x.shape[0]))
    kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = sim_fn(y, y)
    kyy = kyy * (1 - torch.eye(y.shape[0]))
    kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = torch.sum(sim_fn(x, y))
    kxy = kxy / x.shape[0] / y.shape[0]
    mmd = kxx + kyy - 2 * kxy
    return mmd


def binary_exp_hamming_mmd(x, y, bandwidth=0.1):
    sim_fn = functools.partial(binary_exp_hamming_sim, bd=bandwidth)
    return binary_mmd(x, y, sim_fn)


def binary_hamming_mmd(x, y):
    return binary_mmd(x, y, binary_hamming_sim)


def eval_mmd(config, sampler, dataloader, n_rounds: int=10, num_samples: int=1024):
    """Eval mmd."""
    avg_mmd = 0.0
    num_samples // config.data.batch_size
    for i in range(n_rounds):
        gt_data = []
        for batch in dataloader:
            gt_data.append(batch)
        gt_data = torch.stack(gt_data, axis=0)
        gt_data = gt_data.view(-1, config.model.concat_dim)
        x0 = sampler()
        x0 = x0.view(gt_data.shape)
        mmd = binary_exp_hamming_mmd(x0, gt_data)
        avg_mmd += mmd
    mmd = avg_mmd / n_rounds
    return mmd