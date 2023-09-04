import functools
from typing import Any
import torch
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from absl import logging


class TrainState:
    def __init__(self, step, params, opt_state, ema_params):
        self.step = step
        self.params = params
        self.opt_state = opt_state
        self.ema_params = ema_params


def apply_ema(decay, avg, new):
    def ema(a, b):
        return decay * a + (1.0 - decay) * b

    return ema(avg, new)


def copy_pytree(pytree):
    return torch.tensor(pytree)


def build_lr_schedule(config):
    """Build lr schedule."""
    if config.lr_schedule == "constant":
        lr_schedule = lambda step: step * 0 + config.learning_rate
    elif config.lr_schedule == "updown":
        warmup_steps = int(config.warmup_frac * config.total_train_steps)

        def lr_schedule(step):
            return torch.where(
                step < warmup_steps,
                step * config.learning_rate / warmup_steps,
                config.learning_rate
                * (
                    1.0
                    - (step - warmup_steps) / (config.total_train_steps - warmup_steps)
                ),
            )

    elif config.lr_schedule == "up_exp_down":
        warmup_steps = int(config.warmup_frac * config.total_train_steps)

        def lr_schedule(step):
            return torch.where(
                step < warmup_steps,
                0.0,
                config.learning_rate * (0.9 ** ((step - warmup_steps) / 20000)),
            )

    else:
        raise ValueError("Unknown lr schedule %s" % config.lr_schedule)
    return lr_schedule


def build_optimizer(config, parameters):
    """Build optimizer."""
    lr_schedule = build_lr_schedule(config)
    optimizer_name = config.get("optimizer", "adamw")
    optimizer = None
    grad_norm = config.get("grad_norm", 0.0)

    if grad_norm > 0.0:
        parameters = list(parameters)
        optimizer = optim.AdamW(parameters, lr=0.0)  # Placeholder, wird später ersetzt
        optimizer = torch.nn.utils.clip_grad_norm_(parameters, grad_norm, optimizer)

    opt_args = {}
    if optimizer_name in ["adamw", "lamb"]:
        opt_args["weight_decay"] = config.get("weight_decay", 0.0)

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(parameters, lr_schedule(0), **opt_args)
    elif optimizer_name == "lamb":
        optimizer = optim.Lamb(parameters, lr_schedule(0), **opt_args)
    else:
        raise ValueError("Unknown optimizer %s" % optimizer_name)

    return optimizer


def init_host_state(params, optimizer):
    state = TrainState(
        step=0,
        params=params,
        opt_state=optimizer.state_dict(),
        ema_params=copy_pytree(params),
    )
    return state


def torch_to_numpy(torch_batch):
    """PyTorch to NumPy."""

    def convert_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    return convert_tensor(torch_batch)


def numpy_iter(torch_dataset):
    return map(torch_to_numpy, iter(torch_dataset))


def shard_prng_key(prng_key):
    # PRNG keys can be used at train time to drive stochastic modules
    # e.g. DropOut. We would like a different PRNG key for each local
    # device so that we end up with different random numbers on each one,
    # hence we split our PRNG key and put the resulting keys into the batch
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    prng_keys = torch.chunk(prng_key, num_devices)
    return prng_keys


@functools.partial(torch.distributed.barrier)
def all_gather(x):
    # In PyTorch, torch.distributed.barrier synchronizes all processes before the gather
    gathered_data = [
        torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered_data, x)
    return gathered_data


def get_per_process_batch_size(batch_size):
    num_processes = torch.distributed.get_world_size()
    assert (
        batch_size % num_processes == 0
    ), f"Batch size {batch_size} must be divisible by num_processes {num_processes}"
    batch_size_per_process = batch_size // num_processes
    logging.info("Batch size per process: %d", batch_size_per_process)
    return batch_size_per_process


def categorical_kl_logits(logits1, logits2, eps=1e-6):
    p1 = F.softmax(logits1 + eps, dim=-1)
    kl = (
        p1
        * (F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1))
    ).sum(dim=-1)
    return kl


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def categorical_log_likelihood(x, logits):
    log_probs = F.log_softmax(logits, dim=-1)
    x_onehot = F.one_hot(x, logits.shape[-1]).float()
    return (log_probs * x_onehot).sum(dim=-1)


def log1mexp(x):
    x = -torch.abs(x)
    return torch.where(
        x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x))
    )


def binary_hamming_sim(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = (x - y).abs().sum(dim=-1)
    return x.shape[-1] - d


def binary_exp_hamming_sim(x, y, bd):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = (x - y).abs().sum(dim=-1)
    return torch.exp(-bd * d)


def binary_mmd(x, y, sim_fn):
    """MMD for binary data."""
    x = x.astype(torch.float32)
    y = y.astype(torch.float32)
    kxx = sim_fn(x, x)
    kxx *= 1 - torch.eye(x.shape[0], device=x.device)
    kxx = torch.sum(kxx) / (x.shape[0] * (x.shape[0] - 1))

    kyy = sim_fn(y, y)
    kyy *= 1 - torch.eye(y.shape[0], device=y.device)
    kyy = torch.sum(kyy) / (y.shape[0] * (y.shape[0] - 1))

    kxy = torch.sum(sim_fn(x, y))
    kxy /= x.shape[0] * y.shape[0]

    mmd = kxx + kyy - 2 * kxy
    return mmd


def binary_exp_hamming_mmd(x, y, bandwidth=0.1):
    sim_fn = functools.partial(binary_exp_hamming_sim, bd=bandwidth)
    return binary_mmd(x, y, sim_fn)


def binary_hamming_mmd(x, y):
    return binary_mmd(x, y, binary_hamming_sim)


def np_tile_imgs(imgs, pad_pixels=1, pad_val=255, num_col=0):
    if pad_pixels < 0:
        raise ValueError("Expected pad_pixels >= 0")
    if not 0 <= pad_val <= 255:
        raise ValueError("Expected pad_val in [0, 255]")

    imgs = np.asarray(imgs)
    if imgs.dtype != np.uint8:
        raise ValueError("Expected uint8 input")
    n, h, w, c = imgs.shape
    if c not in [1, 3]:
        raise ValueError("Expected 1 or 3 channels")

    if num_col <= 0:
        ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
        num_row = ceil_sqrt_n
        num_col = ceil_sqrt_n
    else:
        assert n % num_col == 0
        num_row = int(np.ceil(n / num_col))

    pad_width = (
        (0, num_row * num_col - n),
        (0, 0),
        (pad_pixels, pad_pixels),
        (pad_pixels, pad_pixels),
    )
    imgs = np.pad(imgs, pad_width=pad_width, mode="constant", constant_values=pad_val)
    h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
    imgs = imgs.reshape(num_row, num_col, h, w, c)
    imgs = imgs.transpose(0, 2, 1, 3, 4)
    imgs = imgs.reshape(num_row * h, num_col * w, c)

    if pad_pixels > 0:
        imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
    if c == 1:
        imgs = imgs[..., 0]
    return imgs
