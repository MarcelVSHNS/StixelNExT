import torch
from torch import nn


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, threshold=0.5):
        """Intersection over Union"""
        super().__init__()
        self.threshold = threshold
        self.kl_loss = nn.L1Loss(reduction='sum')
        # KLDivLoss(reduction="batchmean")

    def forward(self, inputs, targets):
        # inputs = torch.where(inputs < self.threshold, 0, 1)
        loss_1 = self.kl_loss(inputs, targets)
        return loss_1
