import sys
import os
import time
from pathlib import Path
import argparse
from importlib import import_module

import torch
import lightning.pytorch as pl
from HPA_CC.models.train import config_env, get_args, TrainerLogger, find_checkpoint_file, print_with_time
from HPA_CC.models.train import ClassifierLit
from HPA_CC.models.dino import DINO, DINO_HPA
from HPA_CC.data.dataset import RefCLSDM


log_dirs_home = Path("/data/ishang/pseudotime_pred/")

config_env()
args = get_args()

focal = True
soft = False
HPA = True # use HPA DINO embedding or normal
scope = True
ref_concat = True
if not HPA:
    DINO_INPUT = DINO.CLS_DIM
else:
    DINO_INPUT = DINO_HPA.CLS_DIM if not ref_concat else DINO_HPA.CONCAT_CLS_DIM

if ref_concat and not HPA:
    raise ValueError("Can't use ref concat without HPA")

NUM_CLASSES = 4

project_name = f"{'hpa_' if HPA else ''}dino_{'soft_' if soft else ''}{'focal_' if focal else ''}{'scope_' if scope else ''}classifier"
print_with_time(f"Running under project {project_name}, press enter to continue...")
input()

config = {
    "alpha": None,
    "batch_size": 64,
    "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    "num_workers": 1,
    "split": (0.64, 0.16, 0.2),
    "lr": 1e-4,
    "epochs": args.epochs,
    "soft": soft,
    "focal": focal,
    "scope": scope,
    "n_hidden": 0,
    "d_hidden": DINO.CLS_DIM * 2,
    # "dropout": (0.8, 0.5, 0.2)
    "dropout": False,
    "batchnorm": False,
    "num_classes": NUM_CLASSES
}


##########################################################################################
# Set up data, model, and trainer
##########################################################################################

print_with_time("Setting up data module...")
fucci_path = Path(args.data_dir)
dm = RefCLSDM(fucci_path, args.name_data, config["batch_size"], config["num_workers"], config["split"], HPA, "phase", scope=scope)
if args.checkpoint is not None:
    checkpoint_file = find_checkpoint_file(args.checkpoint, log_dirs_home, args.best)
    print_with_time(f"Loading checkpoint from {checkpoint_file}")
    model = ClassifierLit.load_from_checkpoint(checkpoint_file)
else:
    print("Training from scratch")
    model = ClassifierLit(d_input=DINO_INPUT, d_output=NUM_CLASSES, d_hidden=config["d_hidden"], n_hidden=config["n_hidden"], 
                            dropout=config["dropout"], batchnorm=config["batchnorm"], lr=config["lr"], soft=config["soft"], 
                            focal=config["focal"], alpha=config["alpha"])
model.lr = config["lr"]
model.loss_type = config["loss_type"]

print_with_time("Setting up trainer...")
logger = TrainerLogger(model, config, args.name_run, project_name, log_dirs_home)
trainer = pl.Trainer(
    default_root_dir=logger.lit_dir,
    # accelerator="cpu",
    accelerator="gpu",
    devices=config["devices"],
    # strategy=DDPStrategy(find_unused_parameters=True),
    logger=logger.wandb_log,
    max_epochs=config["epochs"],
    gradient_clip_val=5e5,
    callbacks=logger.callbacks
)

##########################################################################################
# Train and test model
##########################################################################################

print_with_time("Training model...")
trainer.fit(model, dm)

print_with_time("Testing model...")
trainer = pl.Trainer(
    default_root_dir=logger.lit_dir,
    accelerator="gpu",
    devices=config["devices"][0:1],
    logger=logger.wandb_log,
)
trainer.test(model, dm)