from pathlib import Path
from datetime import datetime
import torch
import os
import ruamel.yaml
from ml_collections.config_dict import config_dict
import torch.nn as nn


def save_state(state: dict, save_dir) -> None:
    # /Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST
    current_date = datetime.now().strftime("%Y-%m-%d")
    model_filename = f"model_{state['n_iter']}.pt"

    model_path = os.path.join(save_dir, current_date)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, model_filename)

    checkpoint_dict = {
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "n_iter": state["n_iter"],
        # "ema_model_state": self.ema_model.state_dict() if self.use_ema else None,
    }
    torch.save(checkpoint_dict, model_path)


def load_state(state: dict, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state["model"].load_state_dict(checkpoint["model"])

    state["optimizer"].load_state_dict(checkpoint["optimizer"])

    state["n_iter"] = checkpoint["n_iter"]

    return state


def save_config(config: dict, config_dir: str) -> None:
    # /Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_name = 'config_001.yaml'
    
    config_dir = os.path.join(config_dir, current_date)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    config_dir = os.path.join(config_dir, file_name)

    yaml = ruamel.yaml.YAML()
    with open(config_dir, "w") as yaml_file:
        yaml.dump(config.to_dict(), yaml_file)


def load_config(config_dir: str):
    yaml = ruamel.yaml.YAML()
    with open(config_dir, "r") as yaml_file:
        config = yaml.load(yaml_file)

    return config_dict.ConfigDict(config)
