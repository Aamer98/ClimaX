import os
from typing import List
import torch
import numpy as np
import wandb


def wandb_setup(cli) -> None:
    """
    Save the config used by LightningCLI to disk, then save that file to wandb.
    Using wandb.config adds some strange formating that means we'd have to do some 
    processing to be able to use it again as CLI input.

    Also define min and max metrics in wandb, because otherwise it just reports the 
    last known values, which is not what we want.
    """
    # Initialize wandb
    os.environ["WANDB_API_KEY"] = "7a9cbed74d12db3de9cef466bb7b7cf08bdf1ea4"

    wandb.init(
        # Set the project where this run will be logged
        # project=f"ClimaX_seasfire_{cli.model.experiment}",
        project=f"ClimaX_seasfire", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # Track hyperparameters and run metadata
        config={
        "learning_rate": cli.model.hparams.lr,
        "seed": cli.config.seed_everything,
        "batch_size": cli.datamodule.hparams.batch_size,
        })
    wandb.watch(cli.model, log="all")

    config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")
    cfg_string = cli.parser.dump(cli.config, skip_none=False)
    with open(config_file_name, "w") as f:
        f.write(cfg_string)

    wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
    wandb.define_metric("train_loss_epoch", summary="min")
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("train_f1_epoch", summary="max")
    wandb.define_metric("val_f1", summary="max")
    wandb.define_metric("test_f1", summary="max")
    wandb.define_metric("train_precision_epoch", summary="max")
    wandb.define_metric("val_precision", summary="max")
    wandb.define_metric("test_precision", summary="max")
    wandb.define_metric("train_avg_precision_epoch", summary="max")
    wandb.define_metric("val_avg_precision", summary="max")
    wandb.define_metric("test_avg_precision", summary="max")
    wandb.define_metric("train_recall_epoch", summary="max")
    wandb.define_metric("val_recall", summary="max")
    wandb.define_metric("test_recall", summary="max")      
    wandb.define_metric("train_iou_epoch", summary="max")
    wandb.define_metric("val_iou", summary="max") 
    wandb.define_metric("test_iou", summary="max") 

