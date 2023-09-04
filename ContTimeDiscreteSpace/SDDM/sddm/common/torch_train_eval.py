import os
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import distributed as dist
from torchvision import transforms
from functools import partial
from sddm.common import torch_utils
from torch.nn.parallel import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict


def setup_logging(config):
    """Setup logging and writer."""
    if dist.get_rank() == 0:
        logging.info(config)
        logging.info("process count: %d", dist.get_world_size())
        logging.info("device count: %d", torch.cuda.device_count())
        logging.info("device/host: %d", torch.cuda.device_count())

    writer = SummaryWriter(config.save_root) if dist.get_rank() == 0 else None
    if dist.get_rank() == 0:
        fig_folder = os.path.join(config.save_root, "figures")
        os.makedirs(fig_folder, exist_ok=True)
        config.fig_folder = fig_folder
    return writer


def eval_latest_model(folder, writer, global_key, state, fn_eval, prefix="checkpoint_"):
    checkpoint = torch.load(os.path.join(folder, f"{prefix}latest.pth"))
    state.load_state_dict(checkpoint["state_dict"])
    loaded_step = checkpoint["step"]
    logging.info("Restored from %s at step %d", folder, loaded_step)
    process_rng_key = torch.manual_seed(loaded_step + dist.get_rank())
    with writer if writer is not None else torch_utils.nullcontext():
        fn_eval(loaded_step, state, process_rng_key)


def train_loop(
    config,
    writer,
    global_key,
    state,
    train_ds,
    train_step_fn,
    fn_plot_data=None,
    fn_eval=None,
    fn_data_preprocess=None,
):
    """Train loop."""
    if os.path.exists(config.get("model_init_folder", "")):
        checkpoint = torch.load(
            os.path.join(config.model_init_folder, "checkpoint_latest.pth")
        )
        state.load_state_dict(checkpoint["state_dict"])
        logging.info(
            "Restored from %s at step %d", config.model_init_folder, checkpoint["step"]
        )
    ckpt_folder = os.path.join(config.save_root, "ckpts")
    if os.path.exists(ckpt_folder):
        checkpoint = torch.load(os.path.join(ckpt_folder, "checkpoint_latest.pth"))
        state.load_state_dict(checkpoint["state_dict"])
        logging.info("Restored from %s at step %d", ckpt_folder, checkpoint["step"])
    init_step = checkpoint["step"]
    process_rng_key = torch.manual_seed(init_step + dist.get_rank())
    state = DataParallel(state, device_ids=range(torch.cuda.device_count()))

    lr_schedule = torch_utils.build_lr_schedule(config)
    optimizer = torch.optim.AdamW(
        state.parameters(),
        lr=lr_schedule(init_step),
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_metric = None
    if fn_data_preprocess is None:
        fn_data_preprocess = lambda x: x

    def save_model(state, step, prefix="checkpoint_", overwrite=False):
        if dist.get_rank() == 0:
            checkpoint = {"step": step, "state_dict": state.module.state_dict()}
            torch.save(checkpoint, os.path.join(ckpt_folder, f"{prefix}latest.pth"))

    with writer if writer is not None else torch_utils.nullcontext():
        num_params = sum(x.numel() for x in state.parameters())
        writer.add_scalar("num_params", num_params, 0)
        if fn_plot_data is not None:
            x_data = [
                fn_data_preprocess(next(train_ds))
                for _ in range(config.get("plot_num_batches", 10))
            ]
            x_data = torch.cat(x_data, dim=0)
            fn_plot_data(x_data)
        for step in range(init_step + 1, config.total_train_steps + 1):
            batch = fn_data_preprocess(next(train_ds))
            process_rng_key = torch.manual_seed(step + dist.get_rank())
            state.train()
            optimizer.zero_grad()
            outputs = train_step_fn(batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % config.log_every_steps == 0:
                aux = defaultdict(float)
                aux.update(outputs)
                aux["train/lr"] = optimizer.param_groups[0]["lr"]
                for key, value in aux.items():
                    writer.add_scalar(key, value, step)
            if step % config.plot_every_steps == 0 and fn_eval is not None:
                metric = fn_eval(step, state, process_rng_key)
                if metric is not None:
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        save_model(state, step, prefix="bestckpt_", overwrite=True)
            if step % config.save_every_steps == 0:
                save_model(state, step)
