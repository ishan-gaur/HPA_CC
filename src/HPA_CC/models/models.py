import math
import torch
from torch import nn
from HPA_CC.models.utils import FocalLoss
from HPA_CC.models.utils import angle_to_pseudo as angle_conversion
from HPA_CC.models.densenet import DenseNet

class PseudoRegressor(nn.Module):
    """
    Simple feed-forward regression module
    """
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        n_hidden: int = 0,
        d_output: int = 1,
        dropout: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.model = nn.ModuleList()
        self.build_model(d_input, d_hidden, n_hidden, d_output, dropout, batchnorm)
        self.model = nn.Sequential(*self.model)

    def build_model(self, d_input, d_hidden, n_hidden, d_output, dropout, batchnorm):
        # input layer
        # if dropout:
        #     self.model.append(nn.Dropout(0.5))
        self.model.append(nn.Linear(d_input, d_hidden))
        self.model.append(nn.GELU())

        # hidden layers
        for _ in range(n_hidden):
            if batchnorm:
                self.model.append(nn.BatchNorm1d(d_hidden))
            if dropout:
                self.model.append(nn.Dropout(0.5))
            self.model.append(nn.Linear(d_hidden, d_hidden))
            self.model.append(nn.GELU())

        # output layer
        if batchnorm:
            self.model.append(nn.BatchNorm1d(d_hidden))
        if dropout:
            self.model.append(nn.Dropout(0.2))
        self.model.append(nn.Linear(d_hidden, d_output))
        # self.model.append(nn.GELU())
        # self.model.append(nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
        # return self.model(x) * 2 * torch.pi

    def angle_to_pseudo(angle):
        return angle_conversion(angle)

    def cart_distance(theta_pred, y):
        theta = 2 * torch.pi * y
        xy_pred = torch.stack([torch.cos(theta_pred), torch.sin(theta_pred)], dim=-1)
        xy = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return torch.norm(xy_pred - xy, dim=-1)

    def arc_distance(theta_pred, y):
        theta_pred = torch.clone(theta_pred).remainder(2 * torch.pi)
        theta = 2 * torch.pi * y
        return torch.where(
            torch.abs(theta - theta_pred) > torch.pi,
            (2 * torch.pi - torch.abs(theta - theta_pred)) * torch.sign(theta_pred - theta), # flip sign if angle is off by more than pi
            theta - theta_pred
        ) / (2 * torch.pi)

    def reg_distance(theta_pred, y):
        y_pred = theta_pred.remainder(2 * torch.pi)
        y_pred = y_pred / (2 * torch.pi)
        return y_pred - y

    def cart_loss(self, theta_pred, y):
        loss = PseudoRegressor.cart_distance(theta_pred, y)
        return loss

    def arc_loss(self, theta_pred, y):
        loss = PseudoRegressor.arc_distance(theta_pred, y)
        return torch.pow(loss, 2)

    def reg_loss(self, theta_pred, y):
        return torch.pow(PseudoRegressor.reg_distance(theta_pred, y), 2)

    def bin_reweight(self, loss, y, nbin):
        bin_ct = torch.histc(y, nbin)
        bin_ct_avg = torch.mean(bin_ct)
        bin_weights = bin_ct_avg / bin_ct
        bin_weights[bin_weights == float("inf")] = bin_weights[bin_weights != float("inf")].max()
        bins = torch.linspace(y.min(), y.max(), nbin + 1).to(y.device)
        binned_y = torch.bucketize(y, bins)
        binned_y[binned_y == 0] = 1
        binned_y -= 1
        loss_weights = torch.index_select(bin_weights, 0, binned_y)
        loss = loss * loss_weights
        return loss

class Classifier(nn.Module):
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        n_hidden: int = 0,
        d_output: int = 3,
        dropout: bool = False,
        batchnorm: bool = False,
        focal: bool = True,
        alpha = None,
    ):
        super().__init__()
        self.focal = focal
        self.model = nn.ModuleList()
        if batchnorm:
            self.model.append(nn.BatchNorm1d(d_input))
        if dropout:
            self.model.append(nn.Dropout(0.5))
        self.model.append(nn.Linear(d_input, d_hidden))
        self.model.append(nn.GELU())
        for _ in range(n_hidden):
            if dropout:
                self.model.append(nn.Dropout(0.5))
            if batchnorm:
                self.model.append(nn.BatchNorm1d(d_hidden))
            self.model.append(nn.Linear(d_hidden, d_hidden))
            self.model.append(nn.GELU())
        if dropout:
            self.model.append(nn.Dropout(0.2))
        if batchnorm:
            self.model.append(nn.BatchNorm1d(d_hidden))
        self.model.append(nn.Linear(d_hidden, d_output))
        if not self.focal:
            self.model.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*self.model)
        self.loss_fn = nn.CrossEntropyLoss() if not focal else FocalLoss(alpha=alpha)

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y, loss_type=None):
        # x is the output of the model, y is the target
        # x is NOT the INPUT to the model
        if loss_type == "cross_entropy":
            x = nn.Softmax(dim=-1)(x)
            return nn.CrossEntropyLoss()(x, y)
        return self.loss_fn(x, y)

class ConvClassifier(nn.Module):
    def __init__(self, focal: bool = True, alpha = None, d_output: int = 3):
        super().__init__()
        # self.model = DenseNetHPA(d_output=d_output)
        self.model = DenseNet(block_config=(6, 12, 32, 24), growth_rate=32,
            num_classes=4, n_input_channels=2, small_inputs=False, efficient=True)
        self.loss_fn = nn.CrossEntropyLoss() if not focal else FocalLoss(alpha=alpha)

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y, loss_type=None):
        if loss_type == "cross_entropy":
            x = nn.Softmax(dim=-1)(x)
            return nn.CrossEntropyLoss()(x, y)
        return self.loss_fn(x, y)

class CombinedModel(nn.Module):
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        d_repr: int = 32,
        n_hidden: int = 0,
        dropout: bool = False,
        batchnorm: bool = False,
        focal: bool = True,
        alpha = None,
    ):
        super().__init__()
        self.model = nn.ModuleList()
        self.dropout, self.batchnorm = dropout, batchnorm
        self.build_model(d_input, d_hidden, n_hidden, d_repr, self.dropout, self.batchnorm)
        self.model = nn.Sequential(*self.model)

        self.pseudotime = nn.Linear(d_repr, 1)
        self.angle = nn.Linear(d_repr, 1)
        self.phase = nn.ModuleList()
        self.phase.append(nn.Linear(d_repr, 4))
        if not focal:
            self.phase.append(nn.Softmax(dim=-1))
        self.phase = nn.Sequential(*self.phase)
        
        self.focal = focal
        self.phase_loss_fn = nn.CrossEntropyLoss() if not self.focal else FocalLoss(alpha=alpha)

    def build_model(self, d_input, d_hidden, n_hidden, d_repr, dropout, batchnorm):
        # input layer
        if batchnorm:
            self.model.append(nn.BatchNorm1d(d_input))
        if dropout:
            self.model.append(nn.Dropout(0.5))
        self.model.append(nn.Linear(d_input, d_hidden))
        self.model.append(nn.GELU())

        # hidden layers
        for _ in range(n_hidden):
            if batchnorm:
                self.model.append(nn.BatchNorm1d(d_hidden))
            if dropout:
                self.model.append(nn.Dropout(0.5))
            self.model.append(nn.Linear(d_hidden, d_hidden))
            self.model.append(nn.GELU())

        # output layer
        if batchnorm:
            self.model.append(nn.BatchNorm1d(d_hidden))
        if dropout:
            self.model.append(nn.Dropout(0.2))
        self.model.append(nn.Linear(d_hidden, d_repr))
        # self.model = nn.Sequential(*self.model)

    def angle_to_pseudo(angle):
        return angle_conversion(angle)

    def pred_angle(self, x):
        return self.angle(x)

    def pred_pseudotime(self, x):
        return self.pseudotime(x)

    def pred_phase(self, x):
        return self.phase(x)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
        return self.pseudotime(x), self.angle(x), self.phase(x)

    def cart_distance(theta_pred, y):
        theta = 2 * torch.pi * y
        xy_pred = torch.stack([torch.cos(theta_pred), torch.sin(theta_pred)], dim=-1)
        xy = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return torch.norm(xy_pred - xy, dim=-1)

    def arc_distance(theta_pred, y):
        theta_pred = torch.clone(theta_pred).remainder(2 * torch.pi)
        theta = 2 * torch.pi * y
        return torch.where(
            torch.abs(theta - theta_pred) > torch.pi,
            (2 * torch.pi - torch.abs(theta - theta_pred)) * torch.sign(theta_pred - theta), # flip sign if angle is off by more than pi
            theta - theta_pred
        ) / (2 * torch.pi)

    def reg_distance(theta_pred, y):
        y_pred = theta_pred.remainder(2 * torch.pi)
        y_pred = y_pred / (2 * torch.pi)
        return y_pred - y

    def cart_loss(self, theta_pred, y):
        loss = PseudoRegressor.cart_distance(theta_pred, y)
        return loss

    def arc_loss(self, theta_pred, y):
        loss = PseudoRegressor.arc_distance(theta_pred, y)
        return torch.pow(loss, 2)

    def reg_loss(self, theta_pred, y):
        return torch.pow(PseudoRegressor.reg_distance(theta_pred, y), 2)

    def bin_reweight(self, loss, y, nbin):
        bin_ct = torch.histc(y, nbin)
        bin_ct_avg = torch.mean(bin_ct)
        bin_weights = bin_ct_avg / bin_ct
        bin_weights[bin_weights == float("inf")] = bin_weights[bin_weights != float("inf")].max()
        bins = torch.linspace(y.min(), y.max(), nbin + 1).to(y.device)
        binned_y = torch.bucketize(y, bins)
        binned_y[binned_y == 0] = 1
        binned_y -= 1
        loss_weights = torch.index_select(bin_weights, 0, binned_y)
        loss = loss * loss_weights
        return loss

    def phase_loss(self, x, y, loss_type=None):
        # x is the output of the model, y is the target
        # x is NOT the INPUT to the model
        if loss_type == "cross_entropy":
            x = nn.Softmax(dim=-1)(x)
            return nn.CrossEntropyLoss()(x, y)
        return self.phase_loss_fn(x, y)