import torch
import numpy as np
from lib.models.ddsm import *
import time
from functools import partial
from tqdm import tqdm
import lib.utils.bookkeeping as bookkeeping
from lib.sampling.sampling_utils import dna_sampler, importance_sampling
from lib.datasets.datasets import prepare_dna_valid_dataset
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

class Trainer:
    def __init__(self, config):
        self.config = self.config
        
        self.sb = UnitStickBreakingTransform()
        v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(config.diffusion_weights_file)
        v_one = v_one.cpu()
        v_zero = v_zero.cpu()
        v_one_loggrad = v_one_loggrad.cpu()
        v_zero_loggrad = v_zero_loggrad.cpu()
        self.timepoints = timepoints.cpu()

        alpha = torch.ones(config.data.num_cat - 1).float()
        beta =  torch.arange(config.data.num_cat - 1, 0, -1).float()

        
        if config.use_fast_diff:
            self.diffuser_func = partial(diffusion_factory(v_one=v_one, noise_factory_zero=v_zero, noise_factory_one_loggrad=v_one_loggrad, noise_factory_zero_loggrad=v_zero_loggrad, alpha=alpha, beta=beta, device=self.config.device))
        else: 
            self.diffuser_func = partial(diffusion_fast_flatdirichlet(v_one=v_one, v_one_loggrad=v_one_loggrad))


    def dna_train(self, state: dict, sampler, sei, sei_features, data_loader, valid_datasets, valid_seqs):
        #time_dependent_weights = self.importance_sampling()
        time_dependent_weights = importance_sampling(self.config, data_loader,  self.diffuser_func, self.sb)
        if self.config.speed_balanced:
            s = 2 / (torch.ones(self.config.data.num_cat - 1, device=self.config.device) + torch.arange(self.config.data.num_cat - 1, 0, -1, device=self.config.device).float())
        else:
            s = torch.ones(self.config.data.num_cat - 1, device=self.config.device)

        #tqdm_epoch = tqdm.trange(self.config.num_epochs)

        while True:
        # epochen umrechnen in n_iter => epoch = 20 and dataset 10000 => n_iter = 20 * 10000
            avg_loss = 0.
            num_items = 0
            exit_flag = False
            bestsei_validloss = float('Inf')

            stime = time.time()

            for xS in tqdm(data_loader):
                x = xS[:, :, :4]
                xs = xS[:, :, 4:5]

                # Optional : there are several options for importance sampling here. it needs to match the loss function
                random_t = torch.LongTensor(np.random.choice(np.arange(self.config.n_time_steps), size=x.shape[0],
                                                            p=(torch.sqrt(time_dependent_weights) / torch.sqrt(
                                                                time_dependent_weights).sum()).cpu().detach().numpy()))

                if self.config.random_order:
                    order = np.random.permutation(np.arange(self.data.num_cat))
                    # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
                    #perturbed_x, perturbed_x_grad = diffusion_factory(x[..., order], random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta) used diffusion facoty
                    perturbed_x, perturbed_x_grad = self.diffuser_func(x=x[..., order], time_ind=random_t)

                    perturbed_x = perturbed_x[..., np.argsort(order)]
                    perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
                else:
                    perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x=x.cpu(), time_ind=random_t) # used this: Flat dirichlet
                    # perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta)

                perturbed_x = perturbed_x.to(self.config.device)
                perturbed_x_grad = perturbed_x_grad.to(self.config.device)
                random_timepoints = self.timepoints[random_t].to(self.config.device)

                # random_t = random_t.to(config.device)

                xs = xs.to(self.config.device)

                score = state['model'](torch.cat([perturbed_x, xs], -1), random_timepoints)

                # the loss weighting function may change, there are a few options that we will experiment on
                if self.config.random_order:
                    order = np.random.permutation(np.arange(self.config.data.num_cat))
                    perturbed_v = self.sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
                    loss = torch.mean(torch.mean(
                        1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                            (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                                    gx_to_gv(score[..., order], perturbed_x[..., order], create_graph=True) - gx_to_gv(
                                perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2, dim=(1)))
                else:
                    perturbed_v = self.sb._inverse(perturbed_x, prevent_nan=True).detach()
                    loss = torch.mean(torch.mean(
                        1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                            (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                                    gx_to_gv(score, perturbed_x, create_graph=True) - gx_to_gv(perturbed_x_grad,
                                                                                            perturbed_x)) ** 2, dim=(1)))

                state['optimizer'].zero_grad()
                loss.backward()
                state['optimizer'].step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                print("Average Loss:", avg_loss / num_items)


                if (self.config.training.n_iter + 1) % self.config.sampler.sampler_freq == 0 or self.config.training.n_iter == state['n_iter'] - 1 : # 5 => 5 * n_iter
                    state['model'].eval()
                    # with torch no grad?
                    # generate sequence samples
                    allsamples_predh3k4me3 = dna_sampler(self.config, sampler, state['model'], sei, sei_features, valid_datasets)
                    valid_loss = ((valid_seqs - allsamples_predh3k4me3) ** 2).mean()
                    print("Validation Loss:", valid_loss)
                    
                    # save best model
                    if valid_loss < bestsei_validloss:
                        print('Best valid SEI loss!')
                        bestsei_validloss = valid_loss
                        torch.save(state['model'].state_dict(), 'sdedna_promoter_revision.sei.bestvalid.pth')

                    state['model'].train()

                if (state['n_iter'] + 1) % self.config.saving.checkpoint_freq == 0 or state['n_iter']== self.config.training.n_iters - 1:
                    bookkeeping.save_state(state['model'], state['optimizer'], state['n_iter'], self.config.save_location)

                if self.config.training.n_iter == state['n_iter'] - 1:
                    exit_flag = True
                    break

                state['n_iter'] += 1

            print("Average Loss:", avg_loss / num_items)
            if exit_flag:
                break

    def bin_mnist_train(self, state: dict, sampler, sei, sei_features, data_loader, valid_datasets, valid_seqs):
        #time_dependent_weights = self.importance_sampling()
        time_dependent_weights = importance_sampling(self.config, data_loader,  self.diffuser_func, self.sb)
        if self.config.speed_balanced:
            s = 2 / (torch.ones(self.config.data.num_cat - 1, device=self.config.device) + torch.arange(self.config.data.num_cat - 1, 0, -1,
                                                                                            device=self.config.device).float())
        else:
            s = torch.ones(self.config.data.num_cat - 1, device=self.config.device)

        #tqdm_epoch = tqdm.trange(self.config.num_epochs)

        while True:
        # epochen umrechnen in n_iter => epoch = 20 and dataset 10000 => n_iter = 20 * 10000
            avg_loss = 0.
            num_items = 0
            exit_flag = False

            stime = time.time()

            for x in tqdm(data_loader):
                # änderung hier
                x = binary_to_onehot(x.squeeze())

                # Optional : there are several options for importance sampling here. it needs to match the loss function
                random_t = torch.LongTensor(np.random.choice(np.arange(self.config.n_time_steps), size=x.shape[0],
                                                            p=(torch.sqrt(time_dependent_weights) / torch.sqrt(
                                                                time_dependent_weights).sum()).cpu().detach().numpy()))
                # noise data
                if self.config.random_order:
                    order = np.random.permutation(np.arange(self.config.data.num_cat))
                    # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
                    #perturbed_x, perturbed_x_grad = diffusion_factory(x[..., order], random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta) used diffusion facoty
                    perturbed_x, perturbed_x_grad = self.diffuser_func(x=x[..., order], time_ind=random_t)

                    perturbed_x = perturbed_x[..., np.argsort(order)]
                    perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
                else:
                    perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x=x.cpu(), time_ind=random_t) # used this: Flat dirichlet
                    # perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta)

                perturbed_x = perturbed_x.to(self.config.device)
                perturbed_x_grad = perturbed_x_grad.to(self.config.device)
                random_timepoints = self.timepoints[random_t].to(self.config.device)

                random_t = random_t.to(self.config.device)

                # änderung hier kein cat x, s
                # predict noise?
                score = state['model'](perturbed_x, random_timepoints)

                # the loss weighting function may change, there are a few options that we will experiment o
                if self.config.random_order:
                    order = np.random.permutation(np.arange(self.config.data.num_cat))
                    perturbed_v = self.sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
                    loss = torch.mean(torch.mean(
                        1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                            (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                                    gx_to_gv(score[..., order], perturbed_x[..., order], create_graph=True) - gx_to_gv(
                                perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2, dim=(1)))
                else:
                    perturbed_v = self.sb._inverse(perturbed_x, prevent_nan=True).detach()
                    loss = torch.mean(torch.mean(
                        1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                            (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                                    gx_to_gv(score, perturbed_x, create_graph=True) - gx_to_gv(perturbed_x_grad,
                                                                                            perturbed_x)) ** 2, dim=(1)))

                state['optimizer'].zero_grad()
                loss.backward()
                state['optimizer'].step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]


                if (self.config.training.n_iter + 1) % self.config.training.validation_freq == 0: # 5 => 5 * n_iter
                    state['model'].eval()


                    state['model'].train()
                if (self.config.training.n_iter + 1) % self.config.sampler.sampler_freq == 0 or self.config.training.n_iter == state['n_iter'] - 1: 
                    state['model'].eval()
                    samples =sampler(state['model'], self.config.data.shape, batch_size=self.config.sampler.num_samples, max_time=4, min_time=0.01, num_steps=100, eps=1e-5, device=self.config.device)
                    # (28, 28, 2)
                    ## Sample visualization.
                    samples = samples.clamp(0.0, 1.0)
                    sample_grid = make_grid(samples[:,None, :,:,0].detach().cpu(), nrow=int(np.sqrt(self.config.sampler.num_samples)))

                    plt.figure(figsize=(6,6))
                    plt.axis('off')
                    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
                    plt.show()
                    state['model'].train()

                if (state['n_iter'] + 1) % self.config.saving.checkpoint_freq == 0 or state['n_iter']== self.config.training.n_iters - 1:
                    bookkeeping.save_state(state['model'], state['optimizer'], state['n_iter'], self.config.save_location)

                if self.config.training.n_iter == state['n_iter'] - 1:
                    exit_flag = True
                    break

                state['n_iter'] += 1
                            # Print the averaged training loss so far.
            #print(avg_loss / num_items)
            #tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            print("Average Loss:", avg_loss / num_items)
            if exit_flag:
                break
                

