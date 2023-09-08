import argparse
import os.path
import torch
from lib.models.ddsm import noise_factory
from lib.config.config_bin_mnist import get_config



cfg = get_config()

if not os.path.exists(cfg.noise_sample.out_path):
    os.makedirs(cfg.noise_sample.out_path)
elif not os.path.isdir(cfg.noise_sample.out_path):
    print(f"{cfg.out_path} is already exists and it is not a directory")
    exit(1)

str_speed = ".speed_balance" if cfg.noise_sample.speed_balance  else ""
filename = (
    f"steps{cfg.noise_sample.num_time_steps}.cat{cfg.noise_sample.num_cat}{str_speed}.time{cfg.noise_sample.max_time}."
    f"samples{cfg.noise_sample.num_samples}"
)
filepath = os.path.join(cfg.noise_sample.out_path, filename + ".pth")

if os.path.exists(filepath):
    print("File is already exists.")
    exit(1)

torch.set_default_dtype(torch.float64)

alpha = torch.ones(cfg.data.num_cat - 1)
beta = torch.arange(cfg.data.num_cat - 1, 0, -1)

v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = noise_factory(
    cfg.noise_sample.num_samples,
    cfg.noise_sample.num_time_steps,
    alpha,
    beta,
    total_time=cfg.noise_sample.max_time,
    order=cfg.noise_sample.order,
    time_steps=cfg.noise_sample.steps_per_tick,
    logspace=cfg.noise_sample.logspace,
    speed_balanced=cfg.noise_sample.speed_balance,
    mode=cfg.noise_sample.mode,
)

v_one = v_one.cpu()
v_zero = v_zero.cpu()
v_one_loggrad = v_one_loggrad.cpu()
v_zero_loggrad = v_zero_loggrad.cpu()
timepoints = torch.FloatTensor(timepoints)

torch.save((v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints), filepath)