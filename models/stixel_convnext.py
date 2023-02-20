from torch import nn
from torchvision import models


# Define model
class StixelNExT(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Dont use build-in model due to input shape
        # Doc: https://huggingface.co/docs/transformers/model_doc/convnext
        # Tut: https://towardsdatascience.com/implementing-convnext-in-pytorch-7e37a67abba6
        self.model = models.convnext_tiny(weights='DEFAULT')

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
