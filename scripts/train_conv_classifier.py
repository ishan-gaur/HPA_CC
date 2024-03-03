from pathlib import Path
import lightning.pytorch as pl
from HPA_CC.models.train import config_env, get_args, TrainerLogger, find_checkpoint_file, print_with_time
from HPA_CC.models.train import ClassifierLit
from HPA_CC.data.dataset import RefImDM

log_dirs_home = Path("/data/ishang/pseudotime_pred/")

config_env()
args = get_args()

NUM_CLASSES = 4
scope = True
focal = True
soft = False
project_name = f"conv_{'scope_' if scope else ''}{'soft_' if soft else ''}classifier"
print_with_time(f"Running under project {project_name}, press enter to continue...")
input()

config = {
    "focal": focal,
    "soft": soft,
    "alpha": None, # to reweight the focal loss terms per class
    "gamma": 3.0, # how focal is the focal loss, default is
    "batch_size": 32,
    "devices": [0],
    "devices": [1],
    # "devices": [2],
    "num_workers": 1,
    "split": (0.64, 0.16, 0.2), # TODO need to make this so that you send the split per subset and print the actual split and log later
    "lr": 1e-4,
    "gradient_clip_val": 5e5,
    "epochs": args.epochs,
    "num_classes": NUM_CLASSES,
    "dropout": True,
    "batchnorm": False,
}


##########################################################################################
# Set up data and model
##########################################################################################

print_with_time("Setting up model and data module...")

# import os
# env_cp = os.environ.copy()
# node_rank, local_rank, world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']
# import multiprocessing
# process = multiprocessing.current_process()
# pid = process.pid
# print(f"PID: {pid}")

# is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
# pl_trainer_gpus = env_cp['PL_TRAINER_GPUS']
# print(f"Is in DDP subprocess: {is_in_ddp_subprocess}, node rank: {node_rank}, local rank: {local_rank}, world size: {world_size}, trainer gpus: {pl_trainer_gpus}")
# input("contine?")

fucci_path = Path(args.data_dir)
dm = RefImDM(fucci_path, args.name_data, config["batch_size"], config["num_workers"], config["split"], label="phase", scope=scope)

if args.checkpoint is not None:
    checkpoint_file = find_checkpoint_file(args.checkpoint, log_dirs_home, args.best)
    print_with_time(f"Loading checkpoint from {checkpoint_file}")
    model = ClassifierLit.load_from_checkpoint(checkpoint_file)
else:
    print("Training from scratch")
    model = ClassifierLit(focal=config["focal"], soft=config["soft"], alpha=config["alpha"], gamma=config["gamma"], conv=True, 
                          d_output=config["num_classes"], lr=config["lr"], dropout=config["dropout"], batchnorm=config["batchnorm"])
model.lr = config["lr"]
model.gamma = config["gamma"]
model.alpha = config["alpha"]
model.focus = config["focal"]
model.soft = config["soft"]

# from torchinfo import summary
# print(summary(model, (config["batch_size"], 2, 512, 512)))

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
    # strategy="ddp_spawn",
    # strategy="dp",
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