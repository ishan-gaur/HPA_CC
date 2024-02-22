import os
import time
import torch
import argparse
from glob import glob
from pathlib import Path
from copy import deepcopy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from HPA_CC.models.models import PseudoRegressor, Classifier, ConvClassifier, CombinedModel
from HPA_CC.data.dataset import label_types
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import optim
from lightning import LightningModule
import wandb
from lightning.pytorch.utilities import rank_zero_only
import pandas as pd
from warnings import warn

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
        print(f"Logging to {self.log_folder}")
        wandb_dir = self.log_folder

        # wandb.init(mode="disabled")
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

class PseudoRegressorLit(LightningModule):
    """
    Lightning module for training a regression model
    Supports logging for:
    - MSE loss
    - Histogram of predicted pseudotimes
    - Psuedotime residuals
    """
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 4 * 1024,
        n_hidden: int = 1,
        d_output: int = 1,
        lr: float = 1e-4,
        dropout: bool = False,
        batchnorm: bool = False,
        loss_type: str = "arc",
        reweight_loss: bool = False,
        bins: int = 10,
    ):
        super().__init__()
        # if d_hidden is None:
        #     d_hidden = d_input
        self.save_hyperparameters()
        self.model = PseudoRegressor(d_input=d_input, d_hidden=d_hidden, n_hidden=n_hidden, d_output=d_output, dropout=dropout, 
                                    batchnorm=batchnorm)
        self.model = torch.compile(self.model)
        self.lr = lr
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        self.num_classes = d_output
        self.loss_type = loss_type
        self.reweight_loss = reweight_loss
        self.bins = bins

    def forward(self, x):
        return self.model(x)

    def __shared_step(self, batch, batch_idx, stage):
        x, y = batch
        theta_pred = self(x)
        theta_pred = theta_pred.squeeze()
        y = y.squeeze()
        cart_loss = self.model.cart_loss(theta_pred, y)
        arc_loss = self.model.arc_loss(theta_pred, y)
        reg_loss = self.model.reg_loss(theta_pred, y)
        preds, labels = theta_pred.detach().cpu(), y.detach().cpu()
        self.log(f"{stage}/reg_loss", torch.mean(reg_loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{stage}/cart_loss", torch.mean(cart_loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{stage}/arc_loss", torch.mean(arc_loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        losses = {"cart": cart_loss, "arc": arc_loss, "reg": reg_loss}
        loss = losses[self.loss_type]
        self.log(f"{stage}/loss", torch.mean(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.reweight_loss:
            loss = self.model.bin_reweight(loss, y, self.bins)
            self.log(f"{stage}/loss_reweighted", torch.mean(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss = torch.mean(loss)
        return loss, preds, labels
    
    def __log_image(self, stage, name, ax):
        if type(ax) == sns.JointGrid:
            fig = ax.figure
        else:
            fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/{name}": wandb.Image(fig),
        })

    @rank_zero_only
    def __on_shared_epoch_end(self, preds, labels, stage):
        preds, labels = torch.cat(preds), torch.cat(labels)
        preds, labels = preds.flatten(), labels.flatten()
    
        # plot the intensity kdeplot
        plt.clf()
        plt.title("Predicted Pseudotime Distribution")
        # plt.xlabel("Pseudotime")
        # plt.ylabel("Counts")
        preds_pseudotime = PseudoRegressor.angle_to_pseudo(preds)
        ax = sns.histplot(preds_pseudotime, bins=50)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Counts")
        plt.tight_layout()
        self.__log_image(stage, "pseudotime_hist", ax)
        plt.close()

        plt.clf()
        plt.title("Predicted Raw Angles Distribution")
        # plt.xlabel("Theta")
        # plt.ylabel("Counts")
        ax = sns.histplot(preds, bins=50)
        ax.set_xlabel("Theta")
        ax.set_ylabel("Counts")
        plt.tight_layout()
        self.__log_image(stage, "preds_raw_hist", ax)
        plt.close()

        # plot the residuals
        plt.clf()
        plt.title("Residuals")
        # plt.ylabel("Label - Pred")
        # plt.xlabel("Label Pseudotime")
        residuals = PseudoRegressor.arc_distance(preds, labels)
        label_key, resid_key = "Label Pseudotime", "Label - Pred Residuals"
        df = pd.DataFrame({label_key: labels, resid_key: residuals})
        ax = sns.jointplot(data=df, x=label_key, y=resid_key, kind="hist")
        self.__log_image(stage, "residuals", ax)
        plt.close()

        # create confusion matrix
        bins = torch.linspace(0, 1, self.bins + 1).to(preds.device)
        binned_preds = torch.bucketize(preds_pseudotime, bins)
        binned_preds[binned_preds == 0] = 1
        binned_preds -= 1
        binned_labels = torch.bucketize(labels, bins)
        binned_labels[binned_labels == 0] = 1
        binned_labels -= 1

        # need to add one example of each class prediction pair to the confusion matrix
        preds_lapl = [binned_preds]
        preds_lapl.extend([torch.arange(len(bins) - 1).to(preds.device)] * (len(bins) - 1))
        binned_preds = torch.cat(preds_lapl)
        labels_lapl = [binned_labels]
        labels_lapl.extend([torch.ones(len(bins) - 1).to(preds.device) * i for i in range(len(bins) - 1)])
        binned_labels = torch.cat(labels_lapl)

        cm = confusion_matrix(binned_labels, binned_preds)
        # normalize the rows
        # cm = cm / cm.sum(axis=1, keepdims=True)
        cm = cm / cm.sum(axis=0, keepdims=True)
        # cm = cm / cm.sum()
        # print(cm)
        # ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels))
        plt.clf()
        ax = sns.heatmap(cm, vmin=0, vmax=1.0, annot=True, fmt=".2f")
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        self.__log_image(stage, f"cm_{self.bins}", ax)
        plt.close()

        # create confusion matrix
        eval_bins = 3
        bins = torch.linspace(0, 1, eval_bins + 1).to(preds.device)
        binned_preds = torch.bucketize(preds_pseudotime, bins)
        binned_preds[binned_preds == 0] = 1
        binned_preds -= 1
        binned_labels = torch.bucketize(labels, bins)
        binned_labels[binned_labels == 0] = 1
        binned_labels -= 1

        # need to add one example of each class prediction pair to the confusion matrix
        preds_lapl = [binned_preds]
        preds_lapl.extend([torch.arange(len(bins) - 1).to(preds.device)] * (len(bins) - 1))
        binned_preds = torch.cat(preds_lapl)
        labels_lapl = [binned_labels]
        labels_lapl.extend([torch.ones(len(bins) - 1).to(preds.device) * i for i in range(len(bins) - 1)])
        binned_labels = torch.cat(labels_lapl)

        cm = confusion_matrix(binned_labels, binned_preds)
        # normalize the rows
        # cm = cm / cm.sum(axis=1, keepdims=True)
        cm = cm / cm.sum(axis=0, keepdims=True)
        # cm = cm / cm.sum()
        # print(cm)
        plt.clf()
        # ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels))
        ax = sns.heatmap(cm, vmin=0, vmax=1.0, annot=True, fmt=".2f")
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        self.__log_image(stage, f"cm", ax)
        plt.close()
    
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "train")
        self.train_preds.append(preds)
        self.train_labels.append(labels)
        return loss

    def on_train_epoch_end(self):
        self.__on_shared_epoch_end(self.train_preds, self.train_labels, "train")
        self.train_preds, self.train_labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "validate")
        self.val_preds.append(preds)
        self.val_labels.append(labels)
        return loss

    def on_validation_epoch_end(self):
        self.__on_shared_epoch_end(self.val_preds, self.val_labels, "validate")
        self.val_preds, self.val_labels = [], []
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "test")
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        return loss

    def on_test_epoch_end(self):
        self.__on_shared_epoch_end(self.test_preds, self.test_labels, "test")
        self.test_preds, self.test_labels = [], []

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class ClassifierLit(LightningModule):
    def __init__(self,
        conv: bool = False,
        d_input: int = 1024,
        d_hidden = None,
        n_hidden: int = 0,
        d_output: int = 3,
        lr: float = 5e-5,
        soft: bool = False,
        focal: bool = False,
        alpha = None,
        dropout: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_input
        self.save_hyperparameters()
        if not isinstance(alpha, torch.Tensor) and alpha is not None:
            alpha = torch.Tensor(alpha)
        if conv:
            self.model = ConvClassifier(focal=focal, alpha=alpha, d_output=d_output)
        else:
            self.model = Classifier(d_input=d_input, d_hidden=d_hidden, n_hidden=n_hidden, d_output=d_output, focal=focal,
                                    alpha=alpha, dropout=dropout, batchnorm=batchnorm)
        # self.model = torch.compile(self.model)
        self.lr = lr
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        self.soft = soft
        self.focal = focal
        if self.soft and self.focal:
            warn("Soft and focal loss are both enabled, soft loss will be coerced into regular cross entropy loss")
        self.num_classes = d_output

    def forward(self, x):
        return self.model(x)

    def __shared_step(self, batch, batch_idx, stage):
        x, y = batch
        y_pred = self(x)

        label_y = torch.argmax(y, dim=-1)
        label_loss = self.model.loss(y_pred, label_y, loss_type="cross_entropy" if self.focal else None)
        self.log(f"{stage}/label_loss", label_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # soft_y = torch.exp(y) / torch.sum(torch.exp(y), dim=-1, keepdim=True)
        soft_y = torch.exp(y)
        soft_loss = self.model.loss(y_pred, soft_y)
        self.log(f"{stage}/soft_loss", soft_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss = soft_loss if self.soft else label_loss
        preds = torch.argmax(y_pred, dim=-1)
        labels = label_y

        preds, labels = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss, preds, labels
    
    def __on_shared_epoch_end(self, preds, labels, stage):
        plt.clf()
        if self.num_classes == 3:
            classes = ["G1", "S", "G2"]
        elif self.num_classes == 6:
            classes = ["Stop-G1", "G1", "G1-S", "S-G2", "G2", "G2-M"]
        elif self.num_classes == 4:
            classes = ["M-G1", "G1", "S-G2", "G2"]
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        filler = np.arange(self.num_classes)
        preds = np.concatenate((preds, filler))
        labels = np.concatenate((labels, filler))
        cm = confusion_matrix(labels, preds)
        cm = cm - np.identity(self.num_classes)
        # ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels))
        ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels) / 3)
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels(classes)
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels(classes)
        fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/cm": wandb.Image(fig),
        })

        for i, class_name in enumerate(classes):
            self.log(f"{stage}/accuracy_{class_name}", cm[i, i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "train")
        self.train_preds.append(preds)
        self.train_labels.append(labels)
        return loss

    def on_train_epoch_end(self):
        self.__on_shared_epoch_end(self.train_preds, self.train_labels, "train")
        self.train_preds, self.train_labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "validate")
        self.val_preds.append(preds)
        self.val_labels.append(labels)
        return loss

    def on_validation_epoch_end(self):
        self.__on_shared_epoch_end(self.val_preds, self.val_labels, "validate")
        self.val_preds, self.val_labels = [], []
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "test")
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        return loss

    def on_test_epoch_end(self):
        self.__on_shared_epoch_end(self.test_preds, self.test_labels, "test")
        self.test_preds, self.test_labels = [], []

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class CombinedModelLit(LightningModule):
    """
    Lightning module for training a regression model
    Supports logging for:
    - MSE loss
    - Histogram of predicted pseudotimes
    - Psuedotime residuals
    """
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 4 * 1024,
        n_hidden: int = 1,
        d_repr: int = 1,
        lr: float = 1e-4,
        loss_weights: list = [1, 1, 1],
        dropout: bool = False,
        batchnorm: bool = False,
        loss_type: str = "arc",
        reweight_loss: bool = False,
        bins: int = 10,
        soft: bool = False,
        focal: bool = False,
        alpha = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if not isinstance(alpha, torch.Tensor) and alpha is not None:
            alpha = torch.Tensor(alpha)
        self.model = CombinedModel(d_input=d_input, d_hidden=d_hidden, n_hidden=n_hidden, d_repr=d_repr, dropout=dropout, 
                                    batchnorm=batchnorm, focal=focal, alpha=alpha)
        self.model = torch.compile(self.model)
        self.lr = lr
        self.cache_dict = {l: [] for l in label_types}
        self.train_preds, self.val_preds, self.test_preds = deepcopy(self.cache_dict), deepcopy(self.cache_dict), deepcopy(self.cache_dict)
        self.train_labels, self.val_labels, self.test_labels = deepcopy(self.cache_dict), deepcopy(self.cache_dict), deepcopy(self.cache_dict)
        self.loss_type = loss_type
        self.reweight_loss = reweight_loss
        self.bins = bins
        self.soft = soft
        self.focal = focal
        self.loss_weights = loss_weights
        if self.soft and self.focal:
            warn("Soft and focal loss are both enabled, soft loss will be coerced into regular cross entropy loss")
        self.num_classes = 4

    def forward(self, x):
        return self.model(x)

    def __append_cache(self, preds, labels, label, stage):
        if stage not in ["train", "validate", "test"]:
            raise ValueError(f"Stage {stage} is not valid")
        if stage == "train":
            self.train_preds[label].append(preds)
            self.train_labels[label].append(labels)
        elif stage == "validate":
            self.val_preds[label].append(preds)
            self.val_labels[label].append(labels)
        else:
            self.test_preds[label].append(preds)
            self.test_labels[label].append(labels)

    def __shared_step(self, batch, stage):
        x, pseudo, angle, phase = batch # per the RefCLSDM class "all" label setting
        raw_pseudo_pred, raw_angle_pred, phase_pred = self(x)
        pseudo_loss, pseudo_preds, pseudo_labels = self.__shared_regressor_step(raw_pseudo_pred, pseudo, stage, "pseudotime")
        angle_loss, angle_preds, angle_labels = self.__shared_regressor_step(raw_angle_pred, angle, stage, "angle")
        phase_loss, phase_preds, phase_labels = self.__shared_classifier_step(phase_pred, phase, stage)
        self.__append_cache(pseudo_preds, pseudo_labels, "pseudotime", stage)
        self.__append_cache(angle_preds, angle_labels, "angle", stage)
        self.__append_cache(phase_preds, phase_labels, "phase", stage)
        loss = 0
        for loss_term, weight in zip([pseudo_loss, angle_loss, phase_loss], self.loss_weights):
            loss += weight * loss_term
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def __shared_regressor_step(self, theta_pred, y, stage, label):
        theta_pred = theta_pred.squeeze()
        y = y.squeeze()
        arc_loss = self.model.arc_loss(theta_pred, y)
        preds, labels = theta_pred.detach().cpu(), y.detach().cpu()
        self.log(f"{stage}/{label}_arc_loss", torch.mean(arc_loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if self.reweight_loss:
            loss = self.model.bin_reweight(arc_loss, y, self.bins)
            self.log(f"{stage}/{label}_arc_loss_reweighted", torch.mean(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss = torch.mean(loss)
        return loss, preds, labels

    def __shared_classifier_step(self, y_pred, y, stage):
        label_y = torch.argmax(y, dim=-1)
        label_loss = self.model.phase_loss(y_pred, label_y, loss_type="cross_entropy" if self.focal else None)
        self.log(f"{stage}/xe_loss", label_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        soft_y = torch.exp(y) # y is weighted log-probs for each class
        soft_loss = self.model.phase_loss(y_pred, soft_y)
        self.log(f"{stage}/soft_loss", soft_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss = soft_loss if self.soft else label_loss
        preds = torch.argmax(y_pred, dim=-1)
        labels = label_y

        preds, labels = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
        return loss, preds, labels
    
    def __log_image(self, stage, name, ax):
        if type(ax) == sns.JointGrid:
            fig = ax.figure
        else:
            fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/{name}": wandb.Image(fig),
        })

    def __on_shared_epoch_end(self, preds, labels, stage):
        self.__on_epoch_end_pseudotime(preds["pseudotime"], labels["pseudotime"], stage)
        self.__on_epoch_end_angle(preds["angle"], labels["angle"], stage)
        self.__on_epoch_end_phase(preds["phase"], labels["phase"], stage)

    @rank_zero_only
    def __on_epoch_end_pseudotime(self, preds, labels, stage):
        preds, labels = torch.cat(preds), torch.cat(labels)
        preds, labels = preds.flatten(), labels.flatten()
    
        # plot the intensity kdeplot
        plt.clf()
        plt.title("Predicted Pseudotime Distribution")
        preds_pseudotime = PseudoRegressor.angle_to_pseudo(preds)
        ax = sns.histplot(preds_pseudotime, bins=50)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Counts")
        plt.tight_layout()
        self.__log_image(stage, "pseudotime_hist", ax)
        plt.close()

        plt.clf()
        plt.title("Raw Prediction Distribution")
        ax = sns.histplot(preds, bins=50)
        ax.set_xlabel("Theta")
        ax.set_ylabel("Counts")
        plt.tight_layout()
        self.__log_image(stage, "pseudo_raw_preds_hist", ax)
        plt.close()

        # plot the residuals
        plt.clf()
        plt.title("Residuals")
        residuals = PseudoRegressor.arc_distance(preds, labels)
        label_key, resid_key = "Label Pseudotime", "Label - Pred Residuals"
        df = pd.DataFrame({label_key: labels, resid_key: residuals})
        ax = sns.jointplot(data=df, x=label_key, y=resid_key, kind="hist")
        self.__log_image(stage, "pseudo_residuals", ax)
        plt.close()
    
    @rank_zero_only
    def __on_epoch_end_angle(self, preds, labels, stage):
        preds, labels = torch.cat(preds), torch.cat(labels)
        preds, labels = preds.flatten(), labels.flatten()
    
        # plot the intensity kdeplot
        plt.clf()
        plt.title("Predicted Angular Distribution")
        preds_angle = PseudoRegressor.angle_to_pseudo(preds)
        ax = sns.histplot(preds_angle, bins=50)
        ax.set_xlabel("Angle (Rescaled)")
        ax.set_ylabel("Counts")
        plt.tight_layout()
        self.__log_image(stage, "angular_hist", ax)
        plt.close()

        plt.clf()
        plt.title("Raw Angular Distribution")
        ax = sns.histplot(preds, bins=50)
        ax.set_xlabel("Theta")
        ax.set_ylabel("Counts")
        plt.tight_layout()
        self.__log_image(stage, "angle_raw_preds_hist", ax)
        plt.close()

        # plot the residuals
        plt.clf()
        plt.title("Residuals")
        residuals = PseudoRegressor.arc_distance(preds, labels)
        label_key, resid_key = "Label Pseudotime", "Label - Pred Residuals"
        df = pd.DataFrame({label_key: labels, resid_key: residuals})
        ax = sns.jointplot(data=df, x=label_key, y=resid_key, kind="hist")
        self.__log_image(stage, "angle_residuals", ax)
        plt.close()

    @rank_zero_only
    def __on_epoch_end_phase(self, preds, labels, stage):
        plt.clf()
        if self.num_classes == 3:
            classes = ["G1", "S", "G2"]
        elif self.num_classes == 6:
            classes = ["Stop-G1", "G1", "G1-S", "S-G2", "G2", "G2-M"]
        elif self.num_classes == 4:
            classes = ["M-G1", "G1", "S-G2", "G2"]
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        filler = np.arange(self.num_classes)
        preds = np.concatenate((preds, filler))
        labels = np.concatenate((labels, filler))
        cm = confusion_matrix(labels, preds)
        cm = cm - np.identity(self.num_classes)
        cm = cm / cm.sum(axis=0, keepdims=True)
        cm = np.nan_to_num(cm)
        ax = sns.heatmap(cm, vmin=0, vmax=1.0, annot=True, fmt=".2f")
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels(classes)
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels(classes)
        ax.set_title("Phase Confusion Matrix (Normalized by Predicted Class)")
        fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/cm": wandb.Image(fig),
        })

        for i, class_name in enumerate(classes):
            self.log(f"{stage}/accuracy_{class_name}", cm[i, i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0, 
                     on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        return self.__shared_step(batch, "train")

    def on_train_epoch_end(self):
        self.__on_shared_epoch_end(self.train_preds, self.train_labels, "train")
        self.train_preds, self.train_labels = deepcopy(self.cache_dict), deepcopy(self.cache_dict)

    def validation_step(self, batch, batch_idx):
        return self.__shared_step(batch, "validate")

    def on_validation_epoch_end(self):
        self.__on_shared_epoch_end(self.val_preds, self.val_labels, "validate")
        self.val_preds, self.val_labels = deepcopy(self.cache_dict), deepcopy(self.cache_dict)
    
    def test_step(self, batch, batch_idx):
        return self.__shared_step(batch, "test")

    def on_test_epoch_end(self):
        self.__on_shared_epoch_end(self.test_preds, self.test_labels, "test")
        self.test_preds, self.test_labels = deepcopy(self.cache_dict), deepcopy(self.cache_dict)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)