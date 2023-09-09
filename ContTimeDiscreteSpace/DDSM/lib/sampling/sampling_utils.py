import torch
import numpy as np
from lib.models.ddsm import *
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os


def importance_sampling(config, data_loader,  diffuser_func, sb, s):
    time_dependent_cums = torch.zeros(config.n_time_steps).to(config.device)
    time_dependent_counts = torch.zeros(config.n_time_steps).to(config.device)

    for i, x in enumerate(data_loader):
        # x = binary_to_onehot(x.squeeze())
        #x = x[..., :4]
        x = F.one_hot(x.long(), num_classes=config.data.num_cat)
        random_t = torch.randint(0, config.n_time_steps, (x.shape[0],))

        if config.random_order:
            order = np.random.permutation(np.arange(config.data.num_cat))
            # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
            perturbed_x, perturbed_x_grad = diffuser_func(x=x[..., order], time_ind=random_t)
            perturbed_x = perturbed_x[..., np.argsort(order)]
            perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
        else:
            # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)
            perturbed_x, perturbed_x_grad = diffuser_func(x=x, time_ind=random_t)
        perturbed_x = perturbed_x.to(config.device)
        perturbed_x_grad = perturbed_x_grad.to(config.device)
        random_t = random_t.to(config.device)
        perturbed_v = sb._inverse(perturbed_x)


        if config.random_order:
            order = np.random.permutation(np.arange(config.data.num_cat))
            perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
        else:
            perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()

        time_dependent_counts[random_t] += 1

        if config.random_order:
            time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
                gx_to_gv(perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2).view(x.shape[0], -1).mean(
                dim=1).detach()
        else:
            time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
                gx_to_gv(perturbed_x_grad, perturbed_x)) ** 2).view(x.shape[0], -1).mean(dim=1).detach()

    time_dependent_weights = time_dependent_cums / time_dependent_counts
    time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()
    plt.plot(np.arange(1, config.n_time_steps + 1), time_dependent_weights.cpu())

    print("Importance Sampling done")
    return time_dependent_weights

"""
if not os.path.exists(config.saving.time_dep_weights_path):
    os.makedirs(config.saving.time_dep_weights_path)
str_speed = ".speed_balance" if config.speed_balanced  else ""
str_random_order = ".random_order" if config.random_order else ""
filename = (f"time_depend_weights_steps{config.n_time_steps}.cat{config.data.num_cat}{str_speed}{str_random_order}")
filepath = os.path.join(config.saving.time_dep_weights_path, filename + ".pth")
torch.save(time_dependent_weights, filepath)
"""


def dna_sampler(config, sampler, score_model, sei, seifeatures, valid_datasets):
    torch.set_default_dtype(torch.float32)
    allsamples = []
    for t in valid_datasets:
        allsamples.append(sampler(score_model,
                                config.data.shape,
                                batch_size=t.shape[0],
                                max_time=4,
                                min_time=4 / 400,
                                time_dilation=1,
                                num_steps=100,
                                eps=1e-5,
                                speed_balanced=config.speed_balanced,
                                device=config.device,
                                concat_input=t[:, :, 4:5].cuda()
                                ).detach().cpu().numpy()
                        )

    allsamples = np.concatenate(allsamples, axis=0)
    allsamples_pred = np.zeros((2915, 21907))
    for i in range(int(allsamples.shape[0] / 128)):
        seq = 1.0 * (allsamples[i * 128:(i + 1) * 128] > 0.5)
        allsamples_pred[i * 128:(i + 1) * 128] = sei(
            torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                    torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
    seq = allsamples[-128:]
    allsamples_pred[-128:] = sei(
        torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()

    allsamples_predh3k4me3 = allsamples_pred[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=-1)
    return allsamples_predh3k4me3