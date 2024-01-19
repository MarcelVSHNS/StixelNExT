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
        self.existance = nn.BCELoss(reduction='mean')
        self.contin_val = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_pos = 0.0
        loss_bottom = 0.0
        loss_dist = 0.0
        loss_probab = 0.0
        # 4 dim: yT, yB, dist, probab.
        if self.alpha:
            loss_pos = self.contin_val(inputs[:, 0, :, :], targets[:, 0, :, :])
            loss_pos = self.alpha * loss_pos
        if self.beta:
            loss_bottom = self.contin_val(inputs[:, 1, :, :], targets[:, 1, :, :])
            loss_bottom = self.beta * loss_bottom
        if self.gamma:
            loss_dist = self.contin_val(inputs[:, 3, :, :], targets[:, 3, :, :])
            loss_dist = self.gamma * loss_dist
        if self.delta:
            loss_probab = self.existance(inputs[:, 2, :, :], targets[:, 2, :, :])
            loss_probab = self.delta * loss_probab
        return loss_pos + loss_bottom + loss_dist + loss_probab
