from torch import nn
import torch


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, alpha=1.0, beta=0.0, gamma=0.0, threshold=0.5, offset=10.0):
        """Intersection over Union"""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.offset = offset
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets.squeeze(0))
        loss_maximum_cuts = self.limit_num_cuts(inputs)
        loss_dense_pred = self.avoid_dense_predictions(inputs)
        print(f'loss_bce: {self.alpha * loss_bce}, loss_maximum_cuts: {self.beta * loss_maximum_cuts}, loss_dense_pred: {self.gamma * loss_dense_pred}')
        return self.alpha * loss_bce + self.beta * loss_maximum_cuts + self.gamma * loss_dense_pred

    def limit_num_cuts(self, inputs):
        num_cuts_loss = 0.0
        for sample in inputs:
            mask = sample[0] > self.threshold
            num_cuts_loss += mask.sum()
        return num_cuts_loss

    def avoid_dense_predictions(self, inputs):
        dense_loss = 0.0
        for sample in inputs:
            mask = sample[1] > self.threshold
            for col in range(mask.shape[1]):
                column = mask[:, col]
                i_before = False
                for i in range(len(column) - 1):
                    if column[i]:
                        if not i_before:
                            pass
                        else:
                            distance = i - i_before
                            dense_loss += 1 / (distance ** 3)
                        i_before = i
        return dense_loss
