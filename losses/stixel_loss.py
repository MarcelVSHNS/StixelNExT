from torch import nn
import torch


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, alpha=False, beta=False, gamma=False, threshold=0.5, offset=10.0):
        """Intersection over Union"""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.offset = offset
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_bce = 0.0
        loss_maximum_cuts = 0.0
        loss_dense_pred = 0.0
        if self.alpha:
            loss_bce = self.bce_loss(inputs, targets.squeeze(0))
            loss_bce = self.alpha * loss_bce
        if self.beta:
            loss_maximum_cuts = torch.mean(inputs[:,1,:,:])
            loss_maximum_cuts = self.beta * loss_maximum_cuts

        return loss_bce + loss_maximum_cuts, loss_bce, loss_maximum_cuts
