import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.integrate
import math
from tqdm import tqdm

#import lib.sampling.sampling_utils as sampling_utils

def get_initial_samples(N, D, device, S, initial_dist, initial_dist_std=None):
    if initial_dist == 'uniform':
        x = torch.randint(low=0, high=S, size=(N, D), device=device)
    elif initial_dist == 'gaussian':
        target = np.exp(
            - ((np.arange(1, S+1) - S//2)**2) / (2 * initial_dist_std**2)
        )
        target = target / np.sum(target)

        cat = torch.distributions.categorical.Categorical(
            torch.from_numpy(target)
        )
        x = cat.sample((N*D,)).view(N,D)
        x = x.to(device)
    else:
        raise NotImplementedError('Unrecognized initial dist ' + initial_dist)
    return x


#@sampling_utils.register_sampler
class TauLeaping():
    def __init__(self, cfg):
        self.cfg =cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0
        C,H,W = self.cfg.data.shape
        D = C*H*W
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)


            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                qt0 = model.transition(t * torch.ones((N,), device=device)) # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                p0t = F.softmax(model(x, t * torch.ones((N,), device=device)), dim=2) # (N, D, S)


                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())



                qt0_denom = qt0[
                    torch.arange(N, device=device).repeat_interleave(D*S),
                    torch.arange(S, device=device).repeat(N*D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N,D,S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0 # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(D*S),
                    torch.arange(S, device=device).repeat(N*D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N, D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(D),
                    torch.arange(D, device=device).repeat(N),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1,1,S) - x.view(N,D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()
                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist
