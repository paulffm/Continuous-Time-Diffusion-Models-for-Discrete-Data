from trainer_lightning import DiffusionModel
from config import get_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

def main():
    config = get_config()
    d3pm = DiffusionModel(config, exp_dir='SavedModels/')

    checkpoint_callback = ModelCheckpoint(dirpath='SavedModels/',
                                              verbose=False,
                                              save_last=True,
                                              save_weights_only=False,
                                              every_n_epochs=20,
                                              save_on_train_epoch_end=True
                                              )

    comet_logger = CometLogger(
            api_key="",
            save_dir="logs/",  # Optional
            project_name="D3PM",  # Optional
            experiment_name="d3pm-gaussian",  # Optional
        )

    trainer = pl.Trainer(
            max_steps=config.train.num_train_steps,
            gradient_clip_val=1.,
            check_val_every_n_epoch=config.train.eval_every_epoch,
            enable_progress_bar=True,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback],
            logger=comet_logger,
            accelerator="cpu",
            strategy="ddp"
        )

    trainer.fit(d3pm)
    


if __name__ == '__main__':
    main()

