import os
import time
import torch
import argparse
from glob import glob
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def print_with_time(msg):
    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] {msg}")

def config_env():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    torch.set_float32_matmul_precision('medium')


class TrainerLogger:
    def __init__(self, model, config, run_name, project_name, logging_home):
        self.log_folder = logging_home / f"{project_name}_{run_name}"
        if not self.log_folder.exists():
            os.makedirs(self.log_folder, exist_ok=True)
        self.lit_dir = self.log_folder / "lightning_logs"
        wandb_dir = self.log_folder

        self.wandb_log = WandbLogger(
            project=project_name,
            log_model=True,
            save_dir=wandb_dir,
            config=config
        )
        
        self.wandb_log.watch(model, log="all", log_freq=10)

        val_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="validate/loss",
            mode="min",
            dirpath=self.lit_dir,
            filename="{epoch:02d}-{validate/loss:.2f}",
            auto_insert_metric_name=False,
        )

        latest_checkpoint_callback = ModelCheckpoint(dirpath=self.lit_dir, save_last=True)

        self.callbacks = [val_checkpoint_callback, latest_checkpoint_callback]

def get_args():
    parser = argparse.ArgumentParser(description="Train a model to align the FUCCI dataset reference channels with the FUCCI channels",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_dir", required=True, help="path to dataset folder")
    parser.add_argument("-n", "--name_data", required=True, help="dataset version name")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="maximum number of epochs to train for")
    parser.add_argument("-c", "--checkpoint", help="path to checkpoint to load from")
    parser.add_argument("--best", action="store_true", help="load best checkpoint instead of last checkpoint")
    parser.add_argument("-r", "--name_run", default=time.strftime('%Y_%m_%d_%H_%M'), help="Name to help lookup the run's logging directory")
    args = parser.parse_args()
    return args

def find_checkpoint_file(checkpoint, log_dirs_home, best=False):
    chkpt_dir_pattern = f"{log_dirs_home}/*/*/*-{checkpoint}/"
    checkpoint_folder = glob(chkpt_dir_pattern)
    if len(checkpoint_folder) > 1:
        raise ValueError(f"Multiple possible checkpoints found: {checkpoint_folder}")
    if len(checkpoint_folder) == 0:
        raise ValueError(f"No checkpoint found for glob pattern: {chkpt_dir_pattern}")
    models_folder = Path(checkpoint_folder[0]).parent.parent / "lightning_logs"
    if best:
        models_list = list(models_folder.iterdir())
        models_list.sort()
        # the elements should be ###-##.ckpt, 'epoch=###.ckpt', and 'last.ckpt'
        checkpoint_file = models_list[0]
    else:
        checkpoint_file = models_folder / "last.ckpt"
    if not checkpoint_file.exists():
        raise ValueError(f"Checkpoint path {checkpoint_file} does not exist")
    return checkpoint_file