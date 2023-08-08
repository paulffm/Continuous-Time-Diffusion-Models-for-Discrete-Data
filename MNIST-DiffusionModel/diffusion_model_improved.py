import utils
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
from typing import Tuple
from einops import rearrange, reduce
from torch.cuda.amp import autocast
from math import log
from random import random


# parametrize cov in the log domain: exp(v log(beta) + (1-v) log(post_variance)
# L_hybrid = L_simple + lambda * L_vlb
# many different possibilites to parametrize posterior_mean:
# could predict:
# - posterior mean directly
# - could predict x_0 and use the equation above to calculate posterior_mean
# - could predict noise and use:
# equation 9
#   x_t = sqrt(alpha_bar) x_0 + sqrt(a - alpha_bar) eps  => and
# equation 11 => exactly the model_mean in my model
#   post_mean = 1 / sqrt(alpha) * (x_t - beta_t / (sqrt(1 - alpha_bar) * pred_noise)


# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/learned_gaussian_diffusion.py
# Improved


class TargetDiffusion(nn.Module):
    """
    Diffusion Model where you can choose which target you want to predict:
    You can choose between x0, Noise and v
    It is also Classifier free guidance, self conditioning and weighing of the loss function included.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        in_channels: int,
        timesteps: int,
        beta_schedule: str = "linear",
        objective: str = "pred_noise",
        min_snr_loss_weight: bool = False,  # https://arxiv.org/abs/2303.09556 # if False => Loss weigth is simply 1 if we predict noise
        min_snr_gamma: int = 5,  # default in the paper
        offset_noise_strength: float = 0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.image_size = image_size
        # self.in_channels = in_channels
        self.in_channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.objective = objective
        # claimed to bet that 0.1 is ideal
        self.offset_noise_strength = offset_noise_strength

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            betas = utils.linear_beta_schedule(timesteps, beta_end=0.02)
        elif beta_schedule == "cosine":
            # cosine better: Improved Denoising Diffusion Probabilistic Models https://arxiv.org/abs/2102.09672
            betas = utils.cosine_beta_schedule(timesteps, s=0.008)
        elif beta_schedule == "sigmoid":
            betas = utils.sigmoid_beta_schedule(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # sigma of q
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(self.posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod),
        )

        # https://arxiv.org/abs/2303.09556
        snr = alphas_cumprod / (1 - alphas_cumprod)
        # predicting noise ε is mathematically equivalent to predicting x0 by intrinsically involving
        # Signal-to-Noise Ratio as a weight factor, thus we divide the SNR term in practice

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            self.register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            self.register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            self.register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

    # Helper Functions:
    # Logic is as follows:
    # - Predict with NN noise or x_0
    # => in func: _model_predictions
    # - with noise/x_0 calculate x_0/noise
    # => in func: _predict_start_from_noise,  ...
    # - with noise and x_0 calculate posterior mean and variance
    # => in p_mean_variance with the help of q_posterior
    # - sample from Gaussian with posterior mean and variance to get x_{t-1}
    # Formel finden

    @torch.no_grad()
    def sample(
        self, n_samples: int, classes: torch.Tensor = None, cond_weight: float = 0
    ):
        """
        Generates samples denoised (images)

        Args:
            classes (_type_): _description_
            shape (_type_): _description_
            cond_weight (_type_): _description_

        Returns:
            _type_: _description_
        """
        shape = (n_samples, self.in_channels, self.image_size, self.image_size)
        device = next(self.model.parameters()).device

        # start from pure noise (for each example in the batch)
        # img = x_t
        img = torch.randn(shape, device=device)

        sampling_fn = partial(self.p_sample, classes=classes, cond_weight=cond_weight)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling Time Step:"):
            img = sampling_fn(
                x=img,
                t=torch.full((n_samples,), i, device=device, dtype=torch.long),
                t_index=i,
            )
        img.clamp_(-1.0, 1.0)
        img = utils.unnormalize_to_zero_to_one(img)
        return img


    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        classes: torch.Tensor = None,
        cond_weight: float = 0,
        x_self_cond=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, classes=classes, cond_weight=cond_weight, t=t, clip_denoised=True
        )
        if t_index == 0:
            # return model_mean, x_start
            # return model_mean, x_start
            return model_mean
        else:
            noise = torch.randn_like(x)
            pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
            # return pred_img, x_start
            # return pred_img, x_start
            return pred_img

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor = None,
        cond_weight: float = 0,
        x_self_cond=None,
        clip_denoised=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # x_self_cond = None
        pred_noise, pred_x0 = self._model_predictions(
            x, t, classes=classes, cond_weight=cond_weight, x_self_cond=x_self_cond
        )

        if clip_denoised:
            pred_x0.clamp_(-1.0, 1.0)

        # Compute the mean and variance of the diffusion posterior:
        # q(x_{t-1} | x_t, x_0) = p_theta(x_{t-1} | x_t, x_0) => reverse_process
        posterior_mean = (
            utils.extract(self.posterior_mean_coef1, t, x.shape) * pred_x0
            + utils.extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = utils.extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped = utils.extract(
            self.posterior_log_variance_clipped, t, x.shape
        )
        # model_mean, posterior_variance, posterior_log_variance = self._q_posterior(x_start=pred_x0, x_t=x, t=t)

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
            pred_x0,
        )

    def _model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor = None,
        cond_weight: float = 0,
        x_self_cond=None,
        clip_x_start=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if classes is not None:
            device = x.device
            n_samples = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_samples:] = 0.0

            batch_size = x.shape[0]
            # double to do guidance with
            t_double = t.repeat(2)
            x_double = x.repeat(2, 1, 1, 1)

            classes_masked = classes * context_mask
            classes_masked = classes_masked.type(torch.long)
            # first half is gui, second
            preds = self.model(
                x=x_double,
                time=t_double,
                classes=classes_masked,
                x_self_cond=x_self_cond,
            )
            eps1 = (1 + cond_weight) * preds[:batch_size]
            eps2 = cond_weight * preds[batch_size:]
            model_output = eps1 - eps2

        else:
            model_output = self.model(
                x=x, time=t, classes=None, x_self_cond=x_self_cond
            )

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else utils.identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            pred_x_start = self._predict_start_from_noise(x_t=x, t=t, noise=pred_noise)
            pred_x_start = self._predict_start_from_noise(x_t=x, t=t, noise=pred_noise)
            pred_x_start = maybe_clip(pred_x_start)

            # if clip_x_start and rederive_pred_noise:
            #    pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            pred_x_start = model_output
            pred_x_start = maybe_clip(pred_x_start)
            pred_noise = self._predict_noise_from_start(x_t=x, t=t, x0=pred_x_start)
            pred_noise = self._predict_noise_from_start(x_t=x, t=t, x0=pred_x_start)

        elif self.objective == "pred_v":
            v = model_output
            pred_x_start = self._predict_start_from_v(x_t=x, t=t, v=v)
            pred_x_start = self._predict_start_from_v(x_t=x, t=t, v=v)
            pred_x_start = maybe_clip(pred_x_start)
            pred_noise = self._predict_noise_from_start(x_t=x, t=t, x0=pred_x_start)
            pred_noise = self._predict_noise_from_start(x_t=x, t=t, x0=pred_x_start)

        return pred_noise, pred_x_start

    def _predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        # If model_output is noise
        return (
            utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        # if model output is x_0
        return (
            utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_v(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * x_start
        )

    def _predict_start_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v
    ) -> torch.Tensor:
        return (
            utils.extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def forward(
        self,
        x: torch.Tensor,
        classes: torch.Tensor,
        p_uncond: float = 0.1,
    ) -> float:
        # loss type not included
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        x = utils.normalize_to_neg_one_to_one(x)

        noise = torch.randn_like(x)

        if self.offset_noise_strength > 0.0:
            offset_noise = torch.randn(x.shape[:2], device=self.device)
            noise += self.offset_noise_strength * rearrange(
                offset_noise, "b c -> b c 1 1"
            )

        # q_sample: noise the input image/data
        # with autocast(enabled=False):
        x_noisy = (
            utils.extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

        # setting some class labels with probability of p_uncond to 0
        if classes is not None:
            context_mask = torch.bernoulli(
                torch.zeros(classes.shape[0]) + (1 - p_uncond)
            ).to(device)

            # mask for unconditinal guidance
            classes = classes * context_mask
            classes = classes.type(torch.long)  # multiplication changes type

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                _, x_self_cond = self._model_predictions(
                    x=x_noisy, t=t, classes=classes, cond_weight=1
                )  # predict x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(
            x=x_noisy, time=t, classes=classes, x_self_cond=x_self_cond
        )

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            v = self._predict_v(x, t, noise)
            target = v
        else:
            raise ValueError(f"Unknown objective {self.objective}")

        # no reduction => loss same shape as model_out
        loss = F.mse_loss(model_out, target, reduction="none")
        # mean over all dimension but the batch_size
        loss = reduce(loss, "b ... -> b (...)", "mean")
        # weighing every entry of loss with loss_weight => before will be extendend to the size of
        loss = loss * utils.extract(self.loss_weight, t, loss.shape)
        return loss.mean()


class LearnedVarDiffusion(TargetDiffusion):
    def __init__(
        self,
        model: nn.Module,
        image_size: int = 32,
        in_channels: int = 1,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        objective: str = "pred_noise",
        min_snr_loss_weight: bool = False,  # not included => as in the paper
        min_snr_gamma: int = 5,
        offset_noise_strength: float = 0.0,
        vb_loss_weight: float = 0.001,
    ):
        super().__init__(
            model=model,
            image_size=image_size,
            in_channels=in_channels,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            objective=objective,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            offset_noise_strength=offset_noise_strength,
        )

        assert model.out_dim == (
            model.channels * 2
        ), "dimension out of unet must be twice the number of channels for learned variance - you can also set the `learned_variance` keyword argument on the Unet to be `True`"
        assert not model.self_condition, "not supported yet"

        self.vb_loss_weight = vb_loss_weight

    def _model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor = None,
        cond_weight: float = 0,
        x_self_cond=None,
        clip_x_start=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if classes is not None:
            device = x.device
            n_samples = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_samples:] = 0.0

            batch_size = x.shape[0]
            # double to do guidance with
            t_double = t.repeat(2)
            x_double = x.repeat(2, 1, 1, 1)

            classes_masked = classes * context_mask
            classes_masked = classes_masked.type(torch.long)
            # first half is gui, second
            preds = self.model(
                x=x_double,
                time=t_double,
                classes=classes_masked,
                x_self_cond=x_self_cond,
            )
            eps1 = (1 + cond_weight) * preds[:batch_size]
            eps2 = cond_weight * preds[batch_size:]
            model_output = eps1 - eps2

        else:
            model_output = self.model(
                x=x, time=t, classes=None, x_self_cond=x_self_cond
            )
        model_output, pred_variance = model_output.chunk(2, dim=1)

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else utils.identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self._predict_start_from_noise(x_t=x, t=t, noise=model_output)

        elif self.objective == "pred_x0":
            pred_noise = self._predict_noise_from_start(x_t=x, t=t, x0=model_output)
            x_start = model_output

        x_start = maybe_clip(x_start)

        return pred_noise, x_start, pred_variance

    def p_mean_variance(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor = None,
        cond_weight: float = 0,
        x_self_cond=None,
        clip_denoised: bool = True,
        model_output=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if model_output is None:
            # is None in Sample process
            pred_noise, _, var_interp_frac_unnormalized = self._model_predictions(
                x=x,
                t=t,
                classes=classes,
                cond_weight=cond_weight,
                x_self_cond=x_self_cond,
            )

        else:
            # used in learning process
            pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim=1)

        min_log = utils.extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = utils.extract(torch.log(self.betas), t, x.shape)
        var_interp_frac = utils.unnormalize_to_zero_to_one(var_interp_frac_unnormalized)

        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        model_variance = model_log_variance.exp()

        pred_x0 = self._predict_start_from_noise(x, t, pred_noise)

        if clip_denoised:
            pred_x0.clamp_(-1.0, 1.0)

        # Compute the mean of the diffusion posterior:
        # q(x_{t-1} | x_t, x_0) = p_theta(x_{t-1} | x_t, x_0) => reverse_process
        model_mean = (
            utils.extract(self.posterior_mean_coef1, t, x.shape) * pred_x0
            + utils.extract(self.posterior_mean_coef2, t, x.shape) * x
        )

        return model_mean, model_variance, model_log_variance, pred_x0

    def forward(
        self,
        x: torch.Tensor,
        classes: torch.Tensor = None,
        clip_denoised=False,
        p_uncond: float = 0.1,
    ) -> float:
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        x = utils.normalize_to_neg_one_to_one(x)

        noise = torch.randn_like(x)

        if self.offset_noise_strength > 0.0:
            offset_noise = torch.randn(x.shape[:2], device=self.device)
            noise += self.offset_noise_strength * rearrange(
                offset_noise, "b c -> b c 1 1"
            )

        # q_sample: noise the input image/data
        x_noisy = (
            utils.extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

        # setting some class labels with probability of p_uncond to 0
        if classes is not None:
            context_mask = torch.bernoulli(
                torch.zeros(classes.shape[0]) + (1 - p_uncond)
            ).to(device)

            # mask for unconditinal guidance
            classes = classes * context_mask
            classes = classes.type(torch.long)  # multiplication changes type

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                # or classes None, bzw: model_output = self.model(x=x_noisy, time=t, classes=classes, x_self_cond=x_self_cond)
                _, x_self_cond, _ = self._model_predictions(
                    x=x_noisy, t=t, classes=classes, cond_weight=1
                )  # predict x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_output = self.model(
            x=x_noisy, time=t, classes=classes, x_self_cond=x_self_cond
        )

        # calculating kl loss for learned variance (interpolation)
        true_mean, _, true_log_variance_clipped = self._q_posterior(
            x_start=x, x_t=x_noisy, t=t
        )
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            x=x_noisy,
            t=t,
            classes=classes,
            clip_denoised=clip_denoised,
            model_output=model_output,
        )

        # kl loss with detached model predicted mean, for stability reasons as in paper
        detached_model_mean = model_mean.detach()

        kl = utils.normal_kl(
            true_mean,
            true_log_variance_clipped,
            detached_model_mean,
            model_log_variance,
        )
        kl = utils.meanflat(kl) * 1.0 / log(2)

        decoder_nll = -utils.discretized_gaussian_log_likelihood(
            x, means=detached_model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = utils.meanflat(decoder_nll) * 1.0 / log(2)

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb_losses = torch.where(t == 0, decoder_nll, kl)

        # simple loss - predicting noise, x0, or x_prev
        pred_noise, _ = model_output.chunk(2, dim=1)

        # no weighing of loss for now: Just Simple loss L_{Simple}
        simple_losses = F.mse_loss(pred_noise, noise)

        return simple_losses + vb_losses.mean() * self.vb_loss_weight
        return simple_losses + vb_losses.mean() * self.vb_loss_weight
