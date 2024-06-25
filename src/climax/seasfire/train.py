import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import os
from typing import Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only

import wandb
from pytorch_lightning.cli import LightningCLI
from climax.seasfire.module import SeasfireModule
from climax.seasfire.datamodules.seasfire_spatial_datamodule import SeasFireSpatialDataModule
from utils import wandb_setup



class MyLightningCLI(LightningCLI):

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self, cli):
        """
        Save the config used by LightningCLI to disk, then save that file to wandb.
        Using wandb.config adds some strange formating that means we'd have to do some 
        processing to be able to use it again as CLI input.

        Also define min and max metrics in wandb, because otherwise it just reports the 
        last known values, which is not what we want.
        """

        wandb.init(
        # Set the project where this run will be logged
        project=f"ClimaX_seasfire", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # Track hyperparameters and run metadata
        config={
                "learning_rate": self.model.hparams.lr,
                "seed": self.config.seed_everything,
                "batch_size": self.datamodule.hparams.batch_size,
            })
        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")
        # breakpoint()
        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1", summary="max")
        wandb.define_metric("val_f1", summary="max")
        wandb.define_metric("train_precision", summary="max")
        wandb.define_metric("val_precision", summary="max")
        wandb.define_metric("train_recall", summary="max")
        wandb.define_metric("val_recall", summary="max")



def main():
    # cli = MyLightningCLI(BaseModel, FireSpreadDataModule, subclass_mode_model=True, save_config_kwargs={
    #     "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)

    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = MyLightningCLI(
        model_class=SeasfireModule,
        datamodule_class=SeasFireSpatialDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
        # subclass_mode_model=True, 
        save_config_kwargs={"overwrite": True}
    )
    cli.wandb_setup(cli)
    # os.makedirs(cli.trainer.default_root_dir, exist_ok=True)
    # wandb_setup(cli)


    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


    # return optimized metric
    return None


if __name__ == "__main__":
    main()
