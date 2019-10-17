import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from .submodules import *


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'SRGAN'
        
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
        hres = (self.tanh(hres) + 1) / 2
        return hres

    def _make_layer(self, block, num_blocks):
        layer = [block(64, 64) for _ in range(num_blocks)]
        return nn.Sequential(*layer)


class Discriminator(nn.Module):
    def __init__(self, input_shape, batch_size=16):
        super().__init__()
        self.name = 'SRGAN'
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        strides = [2, 1, 2, 1, 2, 1, 2]
        channels = [(64,64), (64,128), (128,128), (128,256), (256,256), (256,512), (512,512)]
        self.blocks = self._make_layer(DiscriminatorBlock, channels, strides)

        n_downsize = sum([1 for stride in strides if stride == 2])
        output_shape = int(input_shape / (2 ** n_downsize))
        
        if isinstance(input_shape, int):
            output_shape = (output_shape, output_shape)
        n_input = int(np.prod(output_shape) * 512)
        self.affine1 = nn.Linear(n_input, 1024)
        self.affine2 = nn.Linear(1024, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.blocks(x)
        x = x.view(self.batch_size, -1)
        x = self.leakyrelu(self.affine1(x))
        pred = self.sigmoid(self.affine2(x))
        return pred

    def _make_layer(self, block, channels, strides):
        layer = [block(*channel, stride=stride) for channel, stride in zip(channels, strides)]
        return nn.Sequential(*layer)
