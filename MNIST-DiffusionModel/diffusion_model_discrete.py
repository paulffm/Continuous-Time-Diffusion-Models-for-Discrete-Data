import utils
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm


class BitDiffusionModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        timesteps: int,
        loss_type: str = "cpu",
        bit_scale: float = 1.0,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.image_size = image_size

        # channels = channels * number_of_bits=8
        self.in_channels = self.model.channels

        # bit scale
        self.bit_scale = bit_scale

        if loss_type == "l1":
            self.loss_func = F.l1_loss
        elif loss_type == "l2":
            self.loss_func = F.mse_loss
        elif loss_type == "huber":
            self.loss_func = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        if beta_schedule == "linear":
            betas = utils.linear_beta_schedule(timesteps, beta_end=0.02)
        elif beta_schedule == "cosine":
            # cosine better: Improved Denoising Diffusion Probabilistic Models https://arxiv.org/abs/2102.09672
            betas = utils.cosine_beta_schedule(timesteps, s=0.008)
        elif beta_schedule == "sigmoid":
            betas = utils.sigmoid_beta_schedule(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

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
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @torch.no_grad()
    def sample(
        self, n_samples: int, classes: torch.Tensor = None, cond_weight: float = 1
    ) -> torch.Tensor:
        """
        Generates samples denoised (images)

        Args:
            classes (_type_): _description_
            shape (_type_): _description_
            cond_weight (_type_): _description_

        Returns:
            _type_: _description_
        """
        device = next(self.model.parameters()).device

        # number of samples to generate

        # start from pure noise (for each example in the batch)
        # img = x_t
        shape = (n_samples, self.in_channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=device)

        if classes is not None:
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 0.0  # makes second half of batch context free
            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                context_mask=context_mask,
            )
        else:
            sampling_fn = partial(self.p_sample)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling Time Step:"):
            img = sampling_fn(
                x=img,
                t=torch.full((n_samples,), i, device=device, dtype=torch.long),
                t_index=i,
            )
            # imgs.append(img.cpu().numpy())
        # I only need last img
        return utils.bits_to_decimal(img)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """
        Generates samples after DDPM Paper

        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_

        Returns:
            _type_: _description_
        """
        # self.model.eval()

        betas_t = utils.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = utils.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        print("p_sample", x.shape)
        pred_noise = self.model(x, time=t)
        pred_noise.clamp_(-self.bit_scale, self.bit_scale)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )
        # self.model.train()
        if t_index == 0:
            return model_mean

        else:
            posterior_variance_t = utils.extract(self.posterior_variance, t, x.shape)
            # posterior_variance_t = betas_t
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:

            # bits to decimal
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_ddim_sample(
        self, x: torch.Tensor, t: torch.Tensor, t_index: int, eta=0, temp=1.0
    ) -> torch.Tensor:
        """
        Generates samples after DDIM Paper


        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            eta (int, optional): _description_. Defaults to 0.
            temp (float, optional): _description_. Defaults to 1.0.

        Returns:
            torch.Tensor: _description_
        """
        alpha_t = utils.extract(self.alphas_cumprod, t, x.shape)
        alpha_prev_t = utils.extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * ((1 - alpha_prev_t) / (1 - alpha_t) * (1 - alpha_t / alpha_prev_t)) ** 0.5
        )
        sqrt_one_minus_alphas_cumprod = utils.extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        pred_noise = self.model(x, time=t)
        pred_noise.clamp_(-self.bit_scale, self.bit_scale)

        pred_x0 = (x - sqrt_one_minus_alphas_cumprod * pred_noise) / (alpha_t**0.5)

        dir_xt = (1.0 - alpha_prev_t - sigma**2).sqrt() * pred_noise
        if sigma == 0.0:
            noise = 0.0
        else:
            noise = torch.randn((1, x.shape[1:]))
        noise *= temp

        x_prev = (alpha_prev_t**0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev

    @torch.no_grad()
    def p_sample_guided(
        self,
        x: torch.Tensor,
        classes: int,
        t: torch.Tensor,
        t_index: int,
        context_mask,
        cond_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Generates guided samples adapted from: https://openreview.net/pdf?id=qw8AKxfYbI

        Args:
            x (torch.Tensor): _description_
            classes (int): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            context_mask (_type_): _description_
            cond_weight (float, optional): _description_. Defaults to 0.0.

        Returns:
            torch.Tensor: _description_
        """

        batch_size = x.shape[0]
        # double to do guidance with
        t_double = t.repeat(2)
        x_double = x.repeat(2, 1, 1, 1)
        betas_t = utils.extract(self.betas, t_double, x_double.shape)
        sqrt_one_minus_alphas_cumprod_t = utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape
        )
        sqrt_recip_alphas_t = utils.extract(
            self.sqrt_recip_alphas, t_double, x_double.shape
        )

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)

        # first half is guided, second non guided
        pred_noise = self.model(x_double, time=t_double, classes=classes_masked)
        # scale to -1 to 1
        pred_noise.clamp_(-self.bit_scale, self.bit_scale)

        eps1 = (1 + cond_weight) * pred_noise[:batch_size]
        eps2 = cond_weight * pred_noise[batch_size:]
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x
            - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = utils.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_guided2(
        self,
        x: torch.Tensor,
        classes: int,
        t: torch.Tensor,
        t_index: int,
        cond_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        More intuitive implementation

        Args:
            x (torch.Tensor): _description_
            classes (int): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            cond_weight (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """

        betas_t = utils.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = utils.extract(self.sqrt_recip_alphas, t, x.shape)

        cond_pred_noise = self.model(x, t, classes)

        if cond_weight > 0:
            uncond_pred_noise = self.model(x, t, None)
            cond_pred_noise = torch.lerp(
                uncond_pred_noise, cond_pred_noise, cond_weight
            )

        cond_pred_noise.clamp_(-self.bit_scale, self.bit_scale)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * cond_pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = utils.extract(self.posterior_variance, t, x.shape)
            noise = torch.zeros_like(x)
            # output x_{t-1}
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def forward(
        self,
        x: torch.Tensor,
        classes: torch.Tensor = None,
        p_uncond: float = 0.1,
    ):
        # self.model.train()
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()

        # shape: (Batch_size, channels * number of bits, image_size, image_size)
        # (B, C * BITS, W, H)
        x = utils.decimal_to_bits(x) * self.bit_scale

        noise = torch.randn_like(x)

        # q_sample: noise images
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

        pred_noise = self.model(x=x_noisy, time=t, classes=classes)

        loss = self.loss_func(noise, pred_noise)
        return loss
