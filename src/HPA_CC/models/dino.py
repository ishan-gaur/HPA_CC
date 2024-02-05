import os
import yaml
import math
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.nn import DataParallel
from templates import Classifier
from lightning import LightningModule

from DINO4Cells_code.archs import vision_transformer as vits
from DINO4Cells_code.archs.vision_transformer import DINOHead
from DINO4Cells_code.archs import xresnet as cell_models  # (!)
from DINO4Cells_code.utils import utils


class DINO(nn.Module):
    PATCH_SIZE = 14
    CLS_DIM = 1024
    def __init__(self,
        imsize: Tuple[int, int] = (256, 256),
        margin_tolerance: int = -1,
    ):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.crop_slices = DINO.__get_crop_slices(imsize, margin_tolerance, DINO.PATCH_SIZE)

    @torch.no_grad()
    def forward(self, x):
        return self.dinov2(x[..., self.crop_slices[-2], self.crop_slices[-1]])

    def __get_crop_slices(imsize, margin_tolerance, dino_patch_size):
        crop_slices = []
        for image_size in imsize:
            closest_multiple = math.floor(image_size / dino_patch_size)
            margin_size = image_size - closest_multiple * dino_patch_size
            print(f"Margin cropped out to fit DINO patches: {margin_size}")
            if margin_tolerance >= 0:
                assert margin_size <= margin_tolerance, f"Error in creating the crop slices for use with the DINO model. Margin size is {margin_size} but margin tolerance is {margin_tolerance}."
            crop_slice = slice(margin_size // 2, image_size - margin_size // 2)
            cropped_image_size = closest_multiple * dino_patch_size
            assert cropped_image_size == image_size - margin_size == crop_slice.stop - crop_slice.start, \
                f"Error in creating the crop slices for use with the DINO model. {cropped_image_size} {image_size} {margin_size} {crop_slice.stop - crop_slice.start}"
            crop_slices.append(crop_slice)
        assert len(crop_slices) == 2, "Error in creating the crop slices for use with the DINO model. Only 2D images are supported. imsize might have more than 2 elements."
        return crop_slices
        
class DINOClassifier(LightningModule):
    def __init__(self,
        imsize: Tuple[int, int] = 256,
        d_output: int = 3,
        margin_tolerance: int = -1,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dino = DINO(imsize=imsize, margin_tolerance=margin_tolerance)
        self.classifier = Classifier(d_input=DINO.CLS_DIM, d_output=d_output)
        self.imsize = imsize
        self.lr = lr

    def forward(self, x):
        return self.classifier(self.dino(x))

    def loss(self, y_pred, y):
        return self.classifier.loss(y_pred, y)
    
    def __shared_step(self, batch, stage):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log(f'{stage}/loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.__shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.__shared_step(batch, 'validate')
    
    def test_step(self, batch, batch_idx):
        return self.__shared_step(batch, 'test')
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)

class HPA_DINO:
    def __init__(self, imsize, batch_size=100, config_file=Path("dino_config.yaml"),
                 device=("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:

        device = torch.device(device)
        config = yaml.safe_load(open(config_file, "r"))

        if config["model"]["arch"] in vits.__dict__.keys():
            print("config['model']['arch'] {} in vits".format(config["model"]["arch"]))
            model = vits.__dict__[config["model"]["arch"]](
                img_size=[224],
                patch_size=config["model"]["patch_size"],
                num_classes=0,
                in_chans=config["model"]["num_channels"],
            )
            embed_dim = model.embed_dim
        elif config["model"]["arch"] in cell_models.__dict__.keys():
            print(f"config['model']['arch'] {config['model']['arch']} in cell_models")
            model = partial(
                cell_models.__dict__[config["model"]["arch"]],
                c_in=config["model"]["num_channels"],
            )(False)
            embed_dim = model[-1].in_features
            model[-1] = nn.Identity()

        if config["embedding"]["HEAD"] == True:
            model = utils.MultiCropWrapper(
                model,
                DINOHead(
                    embed_dim,
                    config["model"]["out_dim"],
                    config["model"]["use_bn_in_head"],
                ),
            )

        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        model.to(device)
        pretrained_weights = config["embedding"]["pretrained_weights"]
        print(f'loaded {config["embedding"]["pretrained_weights"]}')

        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "teacher" in state_dict:
                teacher = state_dict["teacher"]
                if not config["embedding"]["HEAD"] == True:
                    teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
                    teacher = {
                        k.replace("backbone.", ""): v for k, v in teacher.items()
                    }
                msg = model.load_state_dict(teacher, strict=False)
            else:
                student = state_dict
                if not config["embedding"]["HEAD"] == True:
                    student = {k.replace("module.", ""): v for k, v in student.items()}
                    student = {
                        k.replace("backbone.", ""): v for k, v in student.items()
                    }
                student = {k.replace("0.", ""): v for k, v in student.items()}
                msg = model.load_state_dict(student, strict=False)

            for p in model.parameters():
                p.requires_grad = False
            model = model.cuda()
            model = model.eval()
            model = DataParallel(model)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    pretrained_weights, msg
                )
            )
        else:
            print(
                "Checkpoint file not found at {}. Please check and retry".format(
                    pretrained_weights
                )
            )

        self.model = model
        self.device = device
        self.config = config
        self.imsize = imsize
        self.batch_size = batch_size
        
        self.model.to(self.device)

    def clear_sample(self, x):
        x[:, 2:] = 0
        while x.shape[1] < 4:
            x = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)
        x.to(self.device)
        return x

    def predict_cls_dataset(self, dataset):
        cls_tokens = []
        for i in tqdm(range(0, len(dataset), self.batch_size)):
            sample = dataset[i:i+self.batch_size]
            cleared_sample = self.clear_sample(sample)
            cls = self.model(cleared_sample)
            cls_tokens.append(cls)
        cls_tokens = torch.cat(cls_tokens, dim=0).cpu()
        return cls_tokens

    def predict_cls_ref(self, sample):
        # sample must have batch dimension and only DAPI and TUBL channels in that order
        assert len(sample.shape) == 4
        assert sample.shape[1] == 2

        # model channels are rgby, r is tubl and b is dapi, rest need to be zeros
        input_shape = list(sample.shape)
        input_shape[1] = 4
        input_sample = torch.zeros(input_shape)
        input_sample[:, 0] = sample[:, 1]
        input_sample[:, 2] = sample[:, 0]
        cls_tokens = self.model(input_sample).detach().cpu()
        return cls_tokens

    def predict_cls_ref_concat(self, sample):
        # sample must have batch dimension and only DAPI and TUBL channels in that order
        assert len(sample.shape) == 4
        assert sample.shape[1] == 2

        # model channels are rgby, r is tubl and b is dapi, rest need to be zeros
        input_shape = list(sample.shape)
        input_shape[1] = 4
        input_sample = torch.zeros(input_shape)

        input_sample[:, 0] = sample[:, 1]
        input_sample[:, 1] = sample[:, 0]
        input_sample[:, 2] = sample[:, 0]

        dapi_cls_tokens = self.model(input_sample).detach().cpu()

        input_sample[:, 1] = sample[:, 1]
        tubl_cls_tokens = self.model(input_sample).detach().cpu()

        cls_tokens = torch.cat([dapi_cls_tokens, tubl_cls_tokens], dim=1)

        return cls_tokens