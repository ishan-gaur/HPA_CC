from pathlib import Path
import lightning.pytorch as pl
from HPA_CC.models.train import config_env, get_args, TrainerLogger, find_checkpoint_file, print_with_time
from HPA_CC.models.train import ClassifierLit
from HPA_CC.data.dataset import RefImPseudoDM
from HPA_CC.data.well_normalization import buckets
from HPA_CC.models.dino import DINO, DINO_HPA

log_dirs_home = Path("/data/ishang/pseudotime_pred/")

config_env()
args = get_args()

scope = False
loss_type = "arc" # "reg", "cart", "arc"
project_name = f"conv_{'scope_' if scope else ''}classifier"
print_with_time(f"Running under project {project_name}, press enter to continue...")
input()

config = {
    "alpha": None, # to reweight the focal loss terms per class
    "loss_type": loss_type, # TODO reg above, what is this?
    "batch_size": 8,
    # "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    "devices": [0],
    "num_workers": 1,
    "split": (0.64, 0.16, 0.2), # TODO need to make this so that you send the split per subset and print the actual split and log later
    "lr": 1e-5,
    "gradient_clip_val": 5e5,
    "epochs": args.epochs,
}


##########################################################################################
# Set up data and model
##########################################################################################

print_with_time("Setting up model and data module...")

fucci_path = Path(args.data_dir)
dm = RefImPseudoDM(fucci_path, args.name_data, config["batch_size"], config["num_workers"], config["split"], label="phase", scope=scope)

if args.checkpoint is not None:
    checkpoint_file = find_checkpoint_file(args.checkpoint, log_dirs_home, args.best)
    print_with_time(f"Loading checkpoint from {checkpoint_file}")
    model = ClassifierLit.load_from_checkpoint(checkpoint_file)
else:
    print("Training from scratch")
    model = ClassifierLit(focal=True, alpha=config["alpha"], conv=True)

model.lr = config["lr"]
model.loss_type = config["loss_type"]

##########################################################################################
# Train and test
##########################################################################################

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
    gradient_clip_val=config["gradient_clip_val"],
    callbacks=logger.callbacks,
)

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