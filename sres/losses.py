import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss

class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probabiliites):
        loss = -torch.log(probabiliites).sum()
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_loss = MSELoss()
        self.adversarial_loss = AdversarialLoss()
        self.beta = 10e-3

    def forward(self, output, target, probabiliites):
        loss = self.content_loss(output, target) + \
            self.beta * self.adversarial_loss(probabiliites)
        return loss