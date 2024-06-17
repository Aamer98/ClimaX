# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from climax.wildfireTS.datamodule import FireSpreadDataModule
from climax.wildfireTS.module import FireSpreadModule
from utils import wandb_setup
from pytorch_lightning.cli import LightningCLI
import wandb



def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=FireSpreadModule,
        datamodule_class=FireSpreadDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)
    wandb_setup(cli)

    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()