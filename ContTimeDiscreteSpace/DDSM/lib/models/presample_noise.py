import argparse
import os.path
import torch
from ddsm import noise_factory
from config.config_dna import get_config


def main():
    cfg = get_config()

    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)
    elif not os.path.isdir(cfg.out_path):
        print(f"{cfg.out_path} is already exists and it is not a directory")
        exit(1)

    str_speed = ".speed_balance" if cfg.speed_balance else ""
    filename = (
        f"steps{cfg.num_time_steps}.cat{cfg.num_cat}{str_speed}.time{cfg.max_time}."
        f"samples{cfg.num_samples}"
    )
    filepath = os.path.join(cfg.out_path, filename + ".pth")

    if os.path.exists(filepath):
        print("File is already exists.")
        exit(1)

    torch.set_default_dtype(torch.float64)

    alpha = torch.ones(cfg.num_cat - 1)
    beta = torch.arange(cfg.num_cat - 1, 0, -1)

    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = noise_factory(
        cfg.num_samples,
        cfg.num_time_steps,
        alpha,
        beta,
        total_time=cfg.max_time,
        order=cfg.order,
        time_steps=cfg.steps_per_tick,
        logspace=cfg.logspace,
        speed_balanced=cfg.speed_balance,
        mode=cfg.mode,
    )

    v_one = v_one.cpu()
    v_zero = v_zero.cpu()
    v_one_loggrad = v_one_loggrad.cpu()
    v_zero_loggrad = v_zero_loggrad.cpu()
    timepoints = torch.FloatTensor(timepoints)

    torch.save((v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints), filepath)


if __name__ == "__main__":
    main()
