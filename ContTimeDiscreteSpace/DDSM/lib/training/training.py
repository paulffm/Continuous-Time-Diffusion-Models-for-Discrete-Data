import torch
import numpy as np
from lib.models.ddsm import *

class Trainer:
    def __init__(self, config, data_loader):
        self.config = self.config
        self.sb = UnitStickBreakingTransform()
        
        if config.use_fast_diff:
            self.diffuser_func = partial(diffusion_factory)
        else: 
            self.diffuser_func = partial(diffusion_fast_flatdirichlet)

    # mabye außerhalb der class: 2 Klassen für dna und mnist
    def importance_sampling(self):
        time_dependent_cums = torch.zeros(self.config.n_time_steps).to(self.config.device)
        time_dependent_counts = torch.zeros(self.config.n_time_steps).to(self.config.device)


        for i, x in enumerate(self.data_loader):
            x = x[..., :4]
            random_t = torch.randint(0, self.config.n_time_steps, (x.shape[0],))

            order = np.random.permutation(np.arange(self.config.ncat))
            if self.config.random_order:
                # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
                perturbed_x, perturbed_x_grad = self.diffuser_func(x[..., order], random_t, v_one, v_zero, v_one_loggrad,       # used diffusion_factory
                                                                v_zero_loggrad, alpha, beta)
                perturbed_x = perturbed_x[..., np.argsort(order)]
                perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
            else:
                # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)
                perturbed_x, perturbed_x_grad = self.diffuser_func(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, # used diffusion_factory
                                                                alpha, beta)
            perturbed_x = perturbed_x.to(self.config.device)
            perturbed_x_grad = perturbed_x_grad.to(self.config.device)
            random_t = random_t.to(self.config.device)
            perturbed_v = self.sb._inverse(perturbed_x)

            order = np.random.permutation(np.arange(self.config.ncat))

            if self.config.random_order:
                perturbed_v = self.sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
            else:
                perturbed_v = self.sb._inverse(perturbed_x, prevent_nan=True).detach()

            time_dependent_counts[random_t] += 1
            if self.config.speed_balanced:
                s = 2 / (torch.ones(self.config.ncat - 1, device=self.config.device) + torch.arange(self.config.ncat - 1, 0, -1,
                                                                                        device=self.config.device).float())
            else:
                s = torch.ones(self.config.ncat - 1, device=self.config.device)

            if self.config.random_order:
                time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
                    gx_to_gv(perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2).view(x.shape[0], -1).mean(
                    dim=1).detach()
            else:
                time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
                    gx_to_gv(perturbed_x_grad, perturbed_x)) ** 2).view(x.shape[0], -1).mean(dim=1).detach()

        time_dependent_weights = time_dependent_cums / time_dependent_counts
        time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()

        return time_dependent_weights

    def train(self, score_model, optimizer, sei, n_iter):
        time_dependent_weights = self.importance_sampling()
        tqdm_epoch = tqdm.trange(self.config.num_epochs)
        while True:
            avg_loss = 0.
            num_items = 0
            stime = time.time()

            for xS in self.data_loader:
                x = xS[:, :, :4]
                s = xS[:, :, 4:5]

                # Optional : there are several options for importance sampling here. it needs to match the loss function
                random_t = torch.LongTensor(np.random.choice(np.arange(self.config.n_time_steps), size=x.shape[0],
                                                            p=(torch.sqrt(time_dependent_weights) / torch.sqrt(
                                                                time_dependent_weights).sum()).cpu().detach().numpy()))

                if self.config.random_order:
                    order = np.random.permutation(np.arange(C))
                    # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
                    #perturbed_x, perturbed_x_grad = diffusion_factory(x[..., order], random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta) used diffusion facoty
                    perturbed_x, perturbed_x_grad = self.diffuser_func(x[..., order], random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta)


                    perturbed_x = perturbed_x[..., np.argsort(order)]
                    perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
                else:
                    perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x.cpu(), random_t, v_one, v_one_loggrad) # used this: Flat dirichlet
                    # perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta)

                perturbed_x = perturbed_x.to(self.config.device)
                perturbed_x_grad = perturbed_x_grad.to(self.config.device)
                random_timepoints = timepoints[random_t].to(self.config.device)

                # random_t = random_t.to(config.device)

                s = s.to(self.config.device)

                score = score_model(torch.cat([perturbed_x, s], -1), random_timepoints)

                # the loss weighting function may change, there are a few options that we will experiment on
                if self.config.speed_balanced:
                    s = 2 / (torch.ones(self.config.ncat - 1, device=self.config.device) + torch.arange(self.config.ncat - 1, 0, -1,
                                                                                            device=self.config.device).float())
                else:
                    s = torch.ones(self.config.ncat - 1, device=self.config.device)

                if self.config.random_order:
                    order = np.random.permutation(np.arange(self.config.ncat))
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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            # Print the averaged training loss so far.
            print(avg_loss / num_items)
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

            if self.config.training.n_iter % self.config.sampler.sampler_freq == 0 or self.config.training.n_iter == n_iter - 1 : # 5
                score_model.eval()

                # generate sequence samples
                torch.set_default_dtype(torch.float32)
                allsamples = []
                for t in valid_datasets:
                    allsamples.append(sampler(score_model,
                                            (1024, 4),
                                            batch_size=t.shape[0],
                                            max_time=4,
                                            min_time=4 / 400,
                                            time_dilation=1,
                                            num_steps=100,
                                            eps=1e-5,
                                            speed_balanced=self.config.speed_balanced,
                                            device=self.config.device,
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
                valid_loss = ((validseqs_predh3k4me3 - allsamples_predh3k4me3) ** 2).mean()
                print(f"{epoch} valid sei loss {valid_loss} {time.time() - stime}", flush=True)

                if valid_loss < bestsei_validloss:
                    print('Best valid SEI loss!')
                    bestsei_validloss = valid_loss
                    torch.save(score_model.state_dict(), 'sdedna_promoter_revision.sei.bestvalid.pth')

                score_model.train()

            if self.config.training.n_iter == n_iter - 1:

