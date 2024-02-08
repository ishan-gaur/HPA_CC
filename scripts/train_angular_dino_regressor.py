from glob import glob
from pathlib import Path

import lightning.pytorch as pl
from HPA_CC.models.train import config_env, get_args, TrainerLogger, find_checkpoint_file, print_with_time
# TODO: these don't exist!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from HPA_CC.models import PseudoRegressorLit, DINO
from HPA_CC.data import RefChannelPseudoDM


config_env()
args = get_args()

HPA = True
# dataset = (("fucci_cham", "fucci_tile"), "fucci_over", "fucci_over")
# dataset = (("fucci_cham", "fucci_tile"), "fucci_cham", "fucci_over")
# dataset = (("fucci_cham", "fucci_tile"), "fucci_tile", "fucci_over")
# dataset = ("fucci_cham", ("fucci_tile", "fucci_over"), "fucci_over")
dataset = ["fucci_cham", "fucci_tile"]
# dataset = "fucci_cham"
# dataset = "fucci_tile"
concat_well_stats = True
loss_type = "reg" # "cart" "arc"
if not HPA:
    DINO_INPUT = DINO.CLS_DIM
else:
    DINO_INPUT = 768 if dataset is None else 2 * 768
if concat_well_stats:
    DINO_INPUT += 2 * 64

config = {
    "HPA": HPA,
    "dataset": dataset,
    "concat_well_stats": concat_well_stats, # TODO what is this?
    "loss_type": loss_type, # TODO reg above, what is this?
    "reweight_loss": True, # TODO maybe start without this?
    "bins": 6, # TODO is this for the classification loss? Mb I shouldn't use this
    # "batch_size": 32,
    "batch_size": 64,
    "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    # "devices": [0, 1, 2, 3],
    # "devices": [1, 2, 3, 4],
    # "devices": [4, 5, 6, 7],
    # "devices": [0],
    # "devices": [7],
    # "devices": [0, 1, 2, 3, 4],
    "num_workers": 1,
    # "split": (0.64, 0.16, 0.2),
    "split": (0.8, 0.2, 0.0), # TODO need to make this so that you send the split per subset and print the actual split and log later
    "conv": False, # TODO maybe logging can happen automatically via a class in the trainer? calls to functions here can be logged accordingly
    "lr": 1e-4,
    # "lr": 0,
    "gradient_clip_val": 5e5,
    "epochs": args.epochs,
    "nf": 16,
    "n_hidden": 3,
    # "n_hidden": 1,
    # "d_hidden": DINO_INPUT * 12,
    "d_hidden": DINO_INPUT * 2,
    "dropout": False,
    "batchnorm": True,
    "num_classes": 1,
}

NUM_CHANNELS, NUM_CLASSES = 2, config["num_classes"]

log_dirs_home = Path("/data/ishang/pseudotime_pred/")
project_name = f"pseudo_conv_regressor" if config["conv"] else f"pseudo_dino_regressor"


fucci_path = Path(args.data_dir)

##########################################################################################
# Set up data and model
##########################################################################################

dm = RefChannelPseudoDM(fucci_path, args.name_data, config["batch_size"], config["num_workers"], config["split"], HPA=config["HPA"], 
                        dataset=config["dataset"], concat_well_stats=config["concat_well_stats"])

print_with_time("Setting up model and data module...")

if args.checkpoint is not None:
    checkpoint_file = find_checkpoint_file(args.checkpoint, log_dirs_home, args.best)
    print_with_time(f"Loading checkpoint from {checkpoint_file}")
    model = PseudoRegressorLit.load_from_checkpoint(checkpoint_file)
else:
    print("Training from scratch")
    model = PseudoRegressorLit(d_input=DINO_INPUT, d_output=NUM_CLASSES, d_hidden=config["d_hidden"], n_hidden=config["n_hidden"], 
                        dropout=config["dropout"], batchnorm=["batchnorm"], lr=config["lr"], loss_type=config["loss_type"],
                        reweight_loss=config["reweight_loss"], bins=config["bins"])
    model = PseudoRegressorLit.load_from_checkpoint(args.checkpoint)

model.lr = config["lr"]
model.loss_type = config["loss_type"]
model.reweight_loss = config["reweight_loss"]
model.bins = config["bins"]

##########################################################################################
# Train and test
##########################################################################################

print_with_time("Setting up trainer...")
logger = TrainerLogger(model, config, args.name_run, project_name, log_dirs_home)
trainer = pl.Trainer(
    default_root_dir=TrainerLogger.lit_dir,
    # accelerator="cpu",
    accelerator="gpu",
    devices=config["devices"],
    # strategy=DDPStrategy(find_unused_parameters=True),
    logger=TrainerLogger.wandb_log,
    max_epochs=config["epochs"],
    gradient_clip_val=config["gradient_clip_val"],
    callbacks=TrainerLogger.callbacks,
)

print_with_time("Training model...")
trainer.fit(model, dm)

print_with_time("Testing model...")
trainer = pl.Trainer(
    default_root_dir=TrainerLogger.lit_dir,
    accelerator="gpu",
    devices=config["devices"][0],
    logger=TrainerLogger.wandb_log,
)
trainer.test(model, dm)