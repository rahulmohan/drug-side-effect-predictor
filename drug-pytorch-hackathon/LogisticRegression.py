import torch
import torch.nn as nn


class LogisticRegression(torch.nn.Module):
    def __init__(self, latent_dims, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(latent_dims, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = self.linear(x)
        outputs = self.sigmoid(outputs)
        return outputs

