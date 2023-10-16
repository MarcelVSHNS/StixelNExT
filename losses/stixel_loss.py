from torch import nn
import torch


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, threshold=0.5):
        """Intersection over Union"""
        super().__init__()
        self.threshold = threshold
        self.kl_loss = nn.BCELoss(reduction='mean')
        # KLDivLoss(reduction="batchmean")

    def forward(self, inputs, targets):
        # inputs = torch.where(inputs < self.threshold, 0, 1)
        print(f"Input: {inputs.shape}")
        print(f"Target: {targets.squeeze().shape}")
        loss_1 = self.kl_loss(inputs, targets.squeeze())
        loss_2 = self.priori_loss_fn(inputs, targets.squeeze())
        return loss_1 + loss_2

    def priori_loss_fn(self, inputs, targets):
        inputs
        threshold = torch.where(inputs < self.threshold, 0, 1)
        return torch.mean(threshold)
