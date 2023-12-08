import numpy as np
import numpy.linalg as linalg
import torch
import functools
import torch.nn.functional as F

def flatten_dict(dd, separator ='*', prefix =''): 
    """
    https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/
    """
    return { str(prefix) + separator + str(k) if prefix != '' else str(k) : v 
                for kk, vv in dd.items() 
                for k, v in flatten_dict(vv, separator, kk).items() 
                } if isinstance(dd, dict) else { prefix : dd } 

def set_in_nested_dict(nested_dict, keys, new_val):
    """
        Sets a value in a nested dictionary (or ml_collections config)
        e.g.
        nested_dict = \
        {
            'outer1': {
                'inner1': 4,
                'inner2': 5
            },
            'outer2': {
                'inner3': 314,
                'inner4': 654
            }
        } 
        keys = ['outer2', 'inner3']
        new_val = 315
    """
    if len(keys) == 1:
        nested_dict[keys[-1]] = new_val
        return
    return set_in_nested_dict(nested_dict[keys[0]], keys[1:], new_val)

def is_model_state_DDP(dict):
    for key in dict.keys():
        if '.module.' in key:
            return True
    return False

def remove_module_from_keys(dict):
    # dict has keys of the form a.b.module.c.d
    # changes to a.b.c.d
    new_dict = {}
    for key in dict.keys():
        if '.module.' in key:
            new_key = key.replace('.module.', '.')
            new_dict[new_key] = dict[key]
        else:
            new_dict[key] = dict[key]

    return new_dict


def expand_dims(x, axis):
    for i in axis:
        x = x.unsqueeze(i)
    return x



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

# expm1(x) = exp(x) - 1
# log1p(x) = log(1+x)
def log1mexp(x):
    # log(1 - exp(x))
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

def expand_dims(x, axis):
    #if axis == 0:
     #   x =x.unsqueeze(0)

    for i in axis:
        x = x.unsqueeze(i)
    return x