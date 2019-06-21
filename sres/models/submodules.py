import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, shortcut=None):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.prelu = nn.PReLU()
        self.shortcut = shortcut

    def forward(self, x):
        input_layer = x
        x = self.bn1(self.conv1(x))
        x = self.prelu(x)
        x = self.bn2(self.conv2(x))
        if self.shortcut:
            input_layer = self.shortcut(input_layer)
        x = torch.add(x, input_layer)
        return x


class RRDB(nn.Module):
    """ Residual-in-Residual Block """
    def __init__(self, input_channels, output_channels, shortcut=None):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.shortcut = shortcut

    def forward(self, x, beta=0.2):
        input_layer = x
        x1 = self.leakyrelu(self.conv1(x))
        x1 = torch.cat((x1, x), dim=1)

        x2 = self.leakyrelu(self.conv2(x1))
        x2 = torch.cat((x2, x1), dim=1)

        x3 = self.leakyrelu(self.conv3(x2))
        x3 = torch.cat((x3, x2), dim=1)

        x4 = self.leakyrelu(self.conv4(x3))
        x4 = torch.cat((x4, x3), dim=1)
        x4 = torch.multiply(x4, beta)

        if self.shortcut:
            input_layer = self.shortcut(input_layer)
        out = torch.add(x4, input_layer)
        return out
        

class DiscriminatorBlock(nn.Module):
    """ A block used in the discriminator network for the GAN """
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.leakyrelu(x)
        return x


class SubPixelConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
