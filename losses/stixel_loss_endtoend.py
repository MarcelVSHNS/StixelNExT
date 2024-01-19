from torch import nn
import torch
import time


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, alpha=False, beta=False, gamma=False, delta=False):
        """Intersection over Union"""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.binary_val = nn.BCELoss(reduction='mean')
        self.continuous_val = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_top = 0.0
        loss_bottom = 0.0
        loss_dist = 0.0
        loss_probab = 0.0
        # 4 dim: yT, yB, dist, probab.
        if self.alpha:
            loss_top = self.continuous_val(inputs[:, 0, :, :], targets[:, 0, :, :])
            loss_top = self.alpha * loss_top
        if self.beta:
            loss_bottom = self.continuous_val(inputs[:, 1, :, :], targets[:, 1, :, :])
            loss_bottom = self.beta * loss_bottom
        if self.gamma:
            loss_probab = self.binary_val(inputs[:, 2, :, :], targets[:, 2, :, :])
            loss_probab = self.gamma * loss_probab
        if self.delta:
            loss_dist = self.continuous_val(inputs[:, 3, :, :], targets[:, 3, :, :])
            loss_dist = self.delta * loss_dist
        return loss_top + loss_bottom + loss_probab + loss_dist
