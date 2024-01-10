import torch
import functools
from sklearn import metrics
import numpy as np
from lib.datasets import synthetic
def binary_hamming_sim(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = torch.sum(torch.abs(x - y), axis=-1)
    return x.shape[-1] - d


def binary_exp_hamming_sim(x, y, bd):
    x = x.unsqueeze(1) # B, D, 1
    y = y.unsqueeze(0)

    d = torch.sum(torch.abs(x - y), axis=-1)
    #d = torch.sum(x != y, dim=-1)

    #print(d.shape)
    return torch.exp(-bd * d)


def binary_mmd(x, y, cfg, sim_fn):
    """MMD for binary data."""
    device = x.device

    #bm, inv_bm = synthetic.get_binmap(cfg.model.concat_dim, cfg.data.binmode)
    #x = synthetic.bin2float(x.detach().cpu().numpy().astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)
    #y = synthetic.bin2float(y.detach().cpu().numpy().astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)
    #x = torch.tensor(x, device=device)
    #y = torch.tensor(y, device=device)
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    kxx = sim_fn(x, x)

    kxx = kxx * (1 - torch.eye(x.shape[0], device=device))
    kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = sim_fn(y, y)
    kyy = kyy * (1 - torch.eye(y.shape[0], device=device))
    kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = torch.sum(sim_fn(x, y))
    kxy = kxy / x.shape[0] / y.shape[0]
    #print(kxx, kxy, kyy)
    mmd = kxx + kyy - 2 * kxy
    return mmd


def binary_exp_hamming_mmd(x, y, cfg, bandwidth=0.1):
    sim_fn = functools.partial(binary_exp_hamming_sim, bd=bandwidth)
    return binary_mmd(x, y, cfg, sim_fn)


def binary_hamming_mmd(x, y):
    return binary_mmd(x, y, binary_hamming_sim)

def MMD(x, y, kernel, cfg):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    bm, inv_bm = synthetic.get_binmap(cfg.model.concat_dim, cfg.data.binmode)
    x = synthetic.bin2float(x.detach().cpu().numpy().astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)
    y = synthetic.bin2float(y.detach().cpu().numpy().astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)
    x = torch.tensor(x, device=device)
    y = torch.tensor(y, device=device)
    x = x.float()
    y = y.float()
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t()) # B, B 
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [5] #, 10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    
    idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
    XX[idx, idx] = 0.0
    idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
    YY[idx, idx] = 0.0
 
      

    return torch.mean(XX + YY - 2. * XY)

def mmd_rbf(X, Y, cfg, gamma=0.2):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    bm, inv_bm = synthetic.get_binmap(cfg.model.concat_dim, cfg.data.binmode)
    X = synthetic.bin2float(X.astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)
    Y = synthetic.bin2float(Y.astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)

    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def exp_hamming_sim(x, y, bd):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = torch.sum(torch.abs(x - y), dim=-1)
    return torch.exp(-bd * d)


def exp_hamming_mmd(x, y, cfg, bandwidth=0.1):
    device = x.device
    bm, inv_bm = synthetic.get_binmap(cfg.model.concat_dim, cfg.data.binmode)
    x = synthetic.bin2float(x.detach().cpu().numpy().astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)
    y = synthetic.bin2float(y.detach().cpu().numpy().astype(np.int32), inv_bm, cfg.model.concat_dim, cfg.data.int_scale)
    x = torch.tensor(x, device=device)
    y = torch.tensor(y, device=device)
    x = x.float()
    y = y.float()

    with torch.no_grad():
        kxx = exp_hamming_sim(x, x, bd=bandwidth)
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx[idx, idx] = 0.0
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        kyy = exp_hamming_sim(y, y, bd=bandwidth)
        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)

        kxy = torch.sum(exp_hamming_sim(x, y, bd=bandwidth)) / x.shape[0] / y.shape[0]

        mmd = kxx + kyy - 2 * kxy
    return mmd

def eval_mmd(config, model, sampler, dataloader, n_rounds: int=10, n_samples: int=1024):
    """Eval mmd."""
    avg_mmd = 0.0
    n_data = n_samples // config.data.batch_size
    neg_mmd = 0
    neg_rounds = 0
    pos_mmd = 0
    pos_rounds = 0
    exit_flag = False
    print("eval")
    with torch.no_grad():
        for i in range(n_rounds):
            n = 1
            gt_data = []
            while True:
                for batch in dataloader:
                    gt_data.append(batch)
                    if (n) == n_data:
                        exit_flag = True
                        break
                    n += 1
                if exit_flag:
                    break
            gt_data = torch.stack(gt_data, axis=0)
            gt_data = gt_data.view(-1, config.model.concat_dim)
            x0, _ = sampler.sample(model, n_samples)
            x0 = torch.from_numpy(x0).to(device=config.device)
            #mmd =  exp_hamming_mmd(x0, gt_data, config, bandwidth=0.1)
            mmd = binary_exp_hamming_mmd(gt_data, x0, config)
            #mmd = mmd_rbf(x0.detach().cpu().numpy(), gt_data.detach().cpu().numpy(), config)
            #mmd = MMD(gt_data, x0,'rbf', config)
            #mmd = binary_hamming_mmd(gt_data, x0)
            
            if mmd < 0:
                neg_mmd += mmd
                neg_rounds += 1
            else:
                pos_mmd += mmd
                pos_rounds += 1

            avg_mmd += mmd

        if neg_rounds == 0: #or pos_rounds == 0:
            neg_rounds = 1
        if pos_rounds == 0: #or pos_rounds == 0:
            pos_rounds = 1

    print("neg MMD:", (neg_mmd / neg_rounds))#.item())
    print("pos MMD:", (pos_mmd / pos_rounds).item())
    print("Pos Rounds", pos_rounds)
    #print("minus", (neg_mmd / neg_rounds)+ (pos_mmd / pos_rounds))# .item())
    mmd = avg_mmd / n_rounds
    #print("MMD:", mmd.item())
    return mmd