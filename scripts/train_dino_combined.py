from glob import glob
from pathlib import Path

import lightning.pytorch as pl
from HPA_CC.models.train import config_env, get_args, TrainerLogger, find_checkpoint_file, print_with_time
from HPA_CC.models.train import CombinedModelLit
from HPA_CC.data.dataset import RefCLSDM
from HPA_CC.data.well_normalization import buckets
from HPA_CC.models.dino import DINO, DINO_HPA

log_dirs_home = Path("/data/ishang/pseudotime_pred/")

config_env()
args = get_args()

HPA = True # use HPA DINO embedding or normal
ref_concat = True
concat_well_stats = False
reweight = True
focal = True
soft = False
scope = True
NUM_CLASSES = 4
loss_type = "arc" # "reg", "cart", "arc"
if not HPA:
    DINO_INPUT = DINO.CLS_DIM
else:
    DINO_INPUT = DINO_HPA.CLS_DIM if not ref_concat else DINO_HPA.CONCAT_CLS_DIM
if concat_well_stats:
    DINO_INPUT += 2 * buckets

project_name = f"{'hpa_' if HPA else ''}dino_{'int_' if concat_well_stats else ''}{'soft_' if soft else ''}{'focal_' if focal else ''}{'scope_' if scope else ''}combined"
print_with_time(f"Running under project {project_name}, press enter to continue...")
input()

config = {
    # input data options
    "HPA": HPA,
    "concat_well_stats": concat_well_stats, # add intensity stats to the input

    # loss specification
    "loss_type": loss_type, # TODO reg above, what is this?
    "reweight_loss": reweight, # TODO maybe start without this?
    "loss_weights": [1.0, 1.0, 1.0],
    # "loss_weights": [1.0, 50.0, 50.0],
    "bins": 6, # How many bins to use for the reweighting
    # "alpha": [1.0, 1.0, 1.0, 1.0],
    "alpha": [0.1, 0.25, 0.4, 0.25],
    "soft": soft,
    "focal": focal,
    "scope": scope,

    # training setup
    "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    # "devices": [0],
    "num_workers": 1,
    "split": (0.64, 0.16, 0.2), # TODO need to make this so that you send the split per subset and print the actual split and log later
    "batch_size": 64,

    # learning alg options
    "lr": 1e-4,
    "gradient_clip_val": 5e5,
    "epochs": args.epochs,

    # model parameters
    "n_hidden": 3,
    "d_hidden": DINO_INPUT * 2,
    "d_repr": 32,
    "dropout": True,
    "batchnorm": True,
    # "dropout": False,
    # "batchnorm": False,
}


##########################################################################################
# Set up data and model
##########################################################################################
print_with_time("Setting up model and data module...")

fucci_path = Path(args.data_dir)
dm = RefCLSDM(fucci_path, args.name_data, config["batch_size"], config["num_workers"], label="all", 
              split=config["split"], hpa=config["HPA"], concat_well_stats=config["concat_well_stats"],
              scope=config["scope"])

if args.checkpoint is not None:
    checkpoint_file = find_checkpoint_file(args.checkpoint, log_dirs_home, args.best)
    print_with_time(f"Loading checkpoint from {checkpoint_file}")
    model = CombinedModelLit.load_from_checkpoint(checkpoint_file)
else:
    print("Training from scratch")
    model = CombinedModelLit(d_input=DINO_INPUT, d_repr=config["d_repr"], d_hidden=config["d_hidden"], n_hidden=config["n_hidden"], 
                               dropout=config["dropout"], batchnorm=config["batchnorm"], lr=config["lr"], loss_weights=config["loss_weights"],
                               loss_type=config["loss_type"], reweight_loss=config["reweight_loss"], bins=config["bins"],
                               soft=config["soft"], focal=config["focal"], alpha=config["alpha"])

model.lr = config["lr"]
model.loss_type = config["loss_type"]
model.reweight_loss = config["reweight_loss"]
model.bins = config["bins"]

# from torchinfo import summary
# print(summary(model, (32, 2048)))

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