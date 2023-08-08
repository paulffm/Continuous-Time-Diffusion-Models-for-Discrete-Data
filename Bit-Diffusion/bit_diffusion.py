from functools import partial
import torch
from torch import nn
from torch.special import expm1
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
from utils import *
from helpers import *


class BitDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        use_ddim=False,
        noise_schedule="cosine",
        time_difference=0.0,
        bit_scale=1.0,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

        self.bit_scale = bit_scale

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        """
        Generates continous timesteps between 1 and 0

        Args:
            batch (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
            Shape: (2, Batch_size, Timesteps=N)
            wieso 2?
        """
        times = torch.linspace(1.0, 0.0, self.timesteps + 1, device=device)
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference=None):
        # shape = (batch_size, channels, image_size, image_size)
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", total=self.timesteps
        ):
            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min=0.0)

            noise_cond = self.log_snr(time)

            # get predicted x0

            x_start = self.model(img, noise_cond, x_start)

            # clip x0
            # alle werte kleiner als -1 auf -1 und alle werte größer als 1 auf 1 setzen
            # rest bleibt unverändert => begrenzt quasi auf werte -1 bis +1
            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get log(snr)

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next)
            )

            # get alpha sigma of time and next time

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next**2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, "b -> b 1 1 1"),
                torch.randn_like(img),
                torch.zeros_like(img),
            )

            img = mean + (0.5 * log_variance).exp() * noise
        #print("Sample img.shape", img.shape)
        return bits_to_decimal(img)

    @torch.no_grad()
    def ddim_sample(self, shape, time_difference=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        img = torch.randn(shape, device=device)

        x_start = None

        for times, times_next in tqdm(time_pairs, desc="sampling loop time step"):
            # get times and noise levels

            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next)
            )

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min=0.0)

            # predict x0

            x_start = self.model(img, log_snr, x_start)

            # clip x0

            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get predicted noise

            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)

            # calculate x next

            img = x_start * alpha_next + pred_noise * sigma_next

        return bits_to_decimal(img)

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    def forward(self, img, *args, **kwargs):
        (
            batch,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"

        # sample random times
        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.0)

        # convert image to bit representation
        img = decimal_to_bits(img) * self.bit_scale

        # noise sample
        noise = torch.randn_like(img)
        """     
                return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise
        """
        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

        noised_img = alpha * img + sigma * noise

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        self_cond = None
        if torch.rand((1)) < 0.5:
            with torch.no_grad():
                self_cond = self.model(noised_img, noise_level).detach_()

        # predict and take gradient step

        # anstatt t wird in unet noise level gegeben
        # noise level = log_snr(times) => noise schedule aber in log 
        pred = self.model(noised_img, noise_level, self_cond)

        #print("pred noise", pred.shape)
        return F.mse_loss(pred, img)
