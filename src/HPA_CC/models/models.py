import torch
from torch import nn
from HPA_CC.models.utils import FocalLoss

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
        mod_angle = angle.remainder(2 * torch.pi)
        pseudo = mod_angle / (2 * torch.pi)
        return pseudo

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
    ):
        super().__init__()
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
        self.model.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*self.model)
        self.loss = nn.CrossEntropyLoss() if not focal else FocalLoss()

    def forward(self, x):
        return self.model(x)

    def loss(self, y_pred, y):
        return self.loss(y_pred, y)