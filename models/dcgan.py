from torch import nn
import torch.nn.functional as F
import math



class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * F.sigmoid(input)

    def __repr__(self):
        return ('{name}()'.format(name=self.__class__.__name__))


class Upsampling(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(Upsampling, self).__init__()
        self.module = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size, 2, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            Swish())

    def forward(self, input):
        return self.module(input)


class Downsampling(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(Downsampling, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size, 2, 1),
            nn.LeakyReLU(0.2, True))

    def forward(self, input):
        return self.module(input)


class Generator(nn.Module):
    def __init__(self, in_channels, output_size, conv_input_size=4):
        super(Generator, self).__init__()
        self.conv_input_size = conv_input_size
        n_conv_blocks = int(math.log(output_size // conv_input_size, 2))
        assert output_size == 2**n_conv_blocks * conv_input_size, 'output_size must be a power of 2'
        self.linear = nn.Sequential(
            nn.Linear(in_channels, in_channels * conv_input_size**2),
            nn.BatchNorm2d(in_channels * conv_input_size**2),
            Swish())
        layers = []
        for i in range(n_conv_blocks):
            layers.append(Upsampling(in_channels // 2**i))
        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels // 2**n_conv_blocks, 1, 7, 1)])
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        output = self.linear(input)
        output = output.view(output.size(0), -1, self.conv_input_size, self.conv_input_size)
        output = self.conv(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, out_channels, input_size, conv_output_size=4):
        super(Discriminator, self).__init__()
        self.conv_output_size = conv_output_size
        n_conv_blocks = int(math.log(input_size // conv_output_size, 2))
        in_channels = out_channels // 2**n_conv_blocks
        assert input_size == 2**n_conv_blocks * conv_output_size, 'input_size must be a power of 2'
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, in_channels, 7, 1),
            nn.LeakyReLU(0.2, True)]
        for i in range(n_conv_blocks):
            layers.append(Downsampling(in_channels * 2**i))
        self.conv = nn.Sequential(*layers)
        self.linear = nn.Linear(in_channels * 2**n_conv_blocks * conv_output_size**2, 1)

    def forward(self, input):
        output = self.conv(input)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output