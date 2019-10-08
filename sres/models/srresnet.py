import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import *
from functools import partial


class SRResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'SRResNet'
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False)

        self.residual = self._make_layer(ResidualBlock, 16)
        self.sub_pixel_conv1 = SubPixelConv2d(64, 256)
        self.sub_pixel_conv2 = SubPixelConv2d(64, 256)

        self.bn = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()

    def forward(self, x):
        base = self.prelu(self.conv1(x))
        x = self.residual(base)
        x = self.bn(self.conv2(x))
        x = torch.add(x, base)
        x = self.sub_pixel_conv1(x)
        x = self.sub_pixel_conv2(x)
        hres = self.conv3(x)
        return hres

    def _make_layer(self, block, num_blocks):
        layer = [block(64, 64) for _ in range(num_blocks)]
        return nn.Sequential(*layer)


class MSRResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'MSRResNet'
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True)

        res_block = partial(ResidualBlock, use_bn=False, bias=True, shortcut=None)
        self.residual = self._make_layer(res_block, 16)
        self.sub_pixel_conv1 = SubPixelConv2d(64, 256, bias=True)
        self.sub_pixel_conv2 = SubPixelConv2d(64, 256, bias=True)

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        base = x
        x = self.prelu(self.conv1(x))
        x = self.residual(x)
        x = self.sub_pixel_conv1(x)
        x = self.sub_pixel_conv2(x)
        x = self.prelu(self.conv2(x))
        x = self.conv3(x)
        base = F.interpolate(base, scale_factor=4, mode='bicubic')
        base = torch.clamp(base, min=0., max=1.)
        hres = x + base
        return hres

    def _make_layer(self, block, num_blocks):
        layer = [block(64, 64) for _ in range(num_blocks)]
        return nn.Sequential(*layer)