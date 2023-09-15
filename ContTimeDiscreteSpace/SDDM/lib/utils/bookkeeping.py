import jax
from flax.training import checkpoints
from absl import logging
from flax import jax_utils
import os

def save_model(config, state, step, prefix="checkpoint_", overwrite=True):
    if jax.process_index() == 0:
        # from GPU to CPU
        #state = jax.device_get(jax_utils.unreplicate(state))
        current_date = datetime.now().strftime("%Y-%m-%d")
        model_filename = f"model_{step}.pt"

        model_path = os.path.join(save_dir, current_date)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, model_filename)

        checkpoints.save_checkpoint(
            ckpt_folder,
            state,
            step,
            keep=config.get("ckpt_keep", 1),
            prefix=prefix,
            overwrite=overwrite,
        )
        logging.info(f"Model saved in Iteration {step}")


def load_model(config):
