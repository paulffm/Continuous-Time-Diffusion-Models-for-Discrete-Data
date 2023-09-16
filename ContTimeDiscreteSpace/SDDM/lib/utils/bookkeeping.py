import jax
from flax.training import checkpoints
from absl import logging
from flax import jax_utils
import os
from datetime import datetime
import ruamel.yaml
from ml_collections.config_dict import config_dict


def save_model(save_dir, state, step, overwrite=True):
    if jax.process_index() == 0:
        # from GPU to CPU
        # state = jax.device_get(jax_utils.unreplicate(state))
        current_date = datetime.now().strftime("%Y-%m-%d")
        model_filename = f"model_{step}.pt"

        model_path = os.path.join(save_dir, current_date)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, model_filename)

        checkpoints.save_checkpoint(
            model_path,
            state,
            step,
            keep=5,
            overwrite=overwrite,
        )
        logging.info(f"Model saved in Iteration {step}")


def load_model(load_dir):
    pass


def save_config(config: dict, config_dir: str) -> None:
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_name = "config_001.yaml"

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
