import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torchvision.models import vgg19


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probabiliites):
        loss = -torch.log(probabiliites).sum()
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, content_loss_type='vgg'):
        """
        args:
            content_loss (str): which content loss to use, options are {vgg, mse}
        """
        super().__init__()
        self.content_losses = {'vgg': VGGLoss, 'mse': MSELoss}
        self.content_loss = self.content_losses[content_loss_type]()
        self.adversarial_loss = AdversarialLoss()
        self.beta = 10e-3

    def forward(self, output, target, probabiliites):
        loss = self.content_loss(output, target) + \
            self.beta * self.adversarial_loss(probabiliites)
        return loss


class VGGLoss(nn.Module):
    def __init__(self, layer=34):
        super().__init__()
        self.vgg19 = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.vgg19.features.children()))[:layer+1]
        for k, v in self.feature_extractor.named_parameters():
            v.requires_grad = False
        self.mse = nn.MSELoss()

    def forward(self, gen_hr, real_hr, beta=0.006):
        gen_hr_feat_map = self.feature_extractor(gen_hr)
        real_hr_feat_map = self.feature_extractor(real_hr)
        loss = beta * self.mse(real_hr_feat_map, gen_hr_feat_map)
        return loss
