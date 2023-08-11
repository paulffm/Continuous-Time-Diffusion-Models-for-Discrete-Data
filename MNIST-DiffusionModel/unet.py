from functools import partial
from typing import Optional
from layers import *
from utils import *


class Unet(nn.Module):
    """
    Refer to the main paper for the architecture details https://arxiv.org/pdf/2208.04202.pdf
    take in a batch of noisy images and their respective noise levels, and output the noise added to the input
    """

    def __init__(
        self,
        dim,  # ch = 128
        init_dim: int = None,  # int=32,
        out_dim: int = None,
        dim_mults=(1, 2, 4),
        channels: int = 1,
        resnet_block_groups=8,
        learned_sinusoidal_dim=18,
        num_classes: int = 10,
        class_embed_dim: int = 3,
        # added
        use_bits: bool = False,
        # added
        use_learned_var: bool = False,
        use_sinposemb: bool = False,
        self_condition: bool = False,
    ):
        super().__init__()
        # for saving the model config
        self.config = {
            'dim': dim,
            'init_dim': init_dim,
            'out_dim': out_dim,
            'dim_mults': dim_mults,
            'channels': channels,
            'resnet_block_groups': resnet_block_groups,
            'learned_sinusoidal_dim': learned_sinusoidal_dim,
            'num_classes': num_classes,
            'class_embed_dim': class_embed_dim,
            'use_bits': use_bits,
            'use_learned_var': use_learned_var,
            'use_sinposemb': use_sinposemb,
            'self_condition': self_condition
        }

        if use_bits:
            bits = 8
            channels *= bits
            print("using bits")

        self.channels = channels
        # added
        self.self_condition = self_condition

        # if you want to do self conditioning uncomment this
        # input_channels = channels * 2
        input_channels = channels * (2 if self_condition else 1)

        # if learning variance => output dim = 2 * input_dim=channels
        out_channels = channels * (2 if use_learned_var else 1)
        self.out_dim = default(out_dim, out_channels)
        print("self.out_dim", self.out_dim)

        # if init_dim = None init_dim = dim
        init_dim = default(init_dim, dim)

        # in udemy: init_dim = dim = 128
        # could change from 7,3 to 1 and 0
        # self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print("dims", dims)
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        if use_sinposemb:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        else:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)

        # self.final_conv = nn.Conv2d(dim, 1, 1) # channels = 1
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)  #
        print("final", dim, channels, self.final_conv)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        classes: torch.Tensor = None,
        x_self_cond: torch.Tensor = None,
    ):
        # print("unet forward x", x.shape)
        # self conditioning
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        # print("x after init_conv", x.shape)
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()

        # add class labels as additional input and embedd them
        if classes is not None:
            t_start += self.label_emb(classes)
            t_mid += self.label_emb(classes)
            t_end += self.label_emb(classes)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)
            h.append(x)

            x = block2(x, t_start)
            x = attn(x)
            h.append(x)

            x = downsample(x)
        # print("x after down", x.shape)
        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)
        # print("x after mid", x.shape)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)

            x = upsample(x)
        # print("x after up", x.shape)

        # print("r", r.shape)
        # residual connection
        x = torch.cat((x, r), dim=1)
        # print("x after concat r", x.shape)
        x = self.final_res_block(x, t_end)
        # print("x after final res block", x.shape)
        x = self.final_conv(x)
        # print("x final", x.shape)
        return x
