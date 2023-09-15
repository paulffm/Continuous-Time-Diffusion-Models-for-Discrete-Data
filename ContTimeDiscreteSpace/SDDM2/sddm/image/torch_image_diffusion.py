import os
import torch
import numpy as np
from PIL import Image
from sddm.common import torch_utils
from ml_collections import config_dict
from sddm.image import mm_vq


class ImageDiffusion(object):
    """Image model."""

    def __init__(self, config, image_size):
        self.config = config
        self.image_size = image_size

    def log_image(self, images, writer, fname):
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        writer.write_images(0, {"data": images_tensor.unsqueeze(0)})
        if torch.distributed.get_rank() == 0:
            img = Image.fromarray(images)
            img.save(os.path.join(self.config.fig_folder, f"{fname}.png"))

    def plot_data(self, batch_images, writer):
        images = np.reshape(batch_images, [-1, 3, self.image_size, self.image_size])
        images = images[:, :, :, :].transpose(0, 2, 3, 1)
        images = images.astype(np.uint8)
        if self.image_size <= 32:
            num_plot = 256
        else:
            num_plot = 64
        images = images[:num_plot]
        images = torch_utils.np_tile_imgs(images)
        self.log_image(images, writer, "data")

    def plot(self, step, state, rng, sample_fn, writer):
        step_rng_keys = torch_utils.shard_prng_key(rng)
        x0 = sample_fn(state, step_rng_keys)
        x0 = torch_utils.all_gather(x0)
        if torch.distributed.get_rank() == 0:
            x0 = x0.cpu().numpy()
            x0 = x0.transpose(0, 2, 3, 1)
            images = torch_utils.np_tile_imgs(x0.astype(np.uint8))
            self.log_image(images, writer, f"samples_{step}")

    def encode_batch(self, batch):
        return batch


class VqImageDiffusion(ImageDiffusion):
    """VQ image model."""

    def __init__(self, config, image_size):
        super(VqImageDiffusion, self).__init__(config, image_size)
        model_info = config.vq_model_info
        model = config.vq_model
        vq_config = config_dict.ConfigDict()
        vq_config.eval_from = config_dict.ConfigDict()
        vq_config.eval_from.xm = model_info[model]["xm"]
        vq_config.eval_from.checkpoint_path = model_info[model]["checkpoint_path"]
        vq_config.eval_from.step = -1
        self.tokenizer_dict = mm_vq.load_mm_vq_model(vq_config)
        self.std_rgb = 1.0
        self.mean_rgb = 0.0

    def encode_single_batch(self, batch):
        batch = batch / 255.0
        batch = (batch - self.mean_rgb) / self.std_rgb
        inputs_with_t = torch.unsqueeze(torch.tensor(batch, dtype=torch.float32), 1)
        tokens_with_t = self.tokenizer_dict["tokenizer"](inputs_with_t)
        tokens = torch.squeeze(tokens_with_t, 1)
        return tokens

    def decode_tokens(self, tokens):
        tokens_with_t = torch.unsqueeze(tokens, 1)
        x0 = self.tokenizer_dict["detokenizer"](tokens_with_t)
        x0 = torch.squeeze(x0, 1)
        x0 = x0 * self.std_rgb + self.mean_rgb
        x0 = torch.clamp(x0, min=0.0, max=1.0) * 255
        return x0

    def plot_data(self, batch_images, writer):
        x0 = self.pmap_decode(batch_images)
        x0 = torch_utils.all_gather(x0)
        if torch.distributed.get_rank() == 0:
            x0 = x0.cpu().numpy()
            x0 = x0.transpose(0, 2, 3, 1)
            images = torch_utils.np_tile_imgs(x0.astype(np.uint8))
            self.log_image(images, writer, "data")
