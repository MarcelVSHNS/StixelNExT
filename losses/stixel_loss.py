from torch import nn
import torch


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, alpha=1.0, beta=.0, threshold=0.5, offset=10.0):
        """Intersection over Union"""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.offset = offset
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets.squeeze(0))
        # additional penalty term
        mask = inputs[0] > self.threshold
        penalties = mask.sum(dim=0) - self.offset
        penalties = penalties.clamp(min=0)  # set negative values to 0
        loss_penalty = penalties.mean()
        return self.alpha * loss_bce + self.beta * loss_penalty
