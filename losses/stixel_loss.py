import torch
from torch import nn


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, threshold=0.5):
        """Intersection over Union"""
        super().__init__()
        self.threshold = threshold
        self.loss_1 = nn.KLDivLoss(reduction="mean")

    def forward(self, inputs, targets):
        # Binarize prediction
        inputs = torch.where(inputs < self.threshold, 0, 1)
        batch_size = targets.shape[0]
        # TODO: Apply Loss_1 with KL-Div per row
        # TODO: Enter additional Losses
        """
        intersection = torch.logical_and(inputs, targets)
        intersection = intersection.view(batch_size, -1).sum(-1)
        union = torch.logical_or(inputs, targets)
        union = union.view(batch_size, -1).sum(-1)
        """
        return self.loss_1(inputs, targets)
