from torch import nn
from math import log
from src.utils import get_nonlinear_layer, get_norm_layer, get_conv_block
from src.utils import weights_init, View, Identity



class Generator(nn.Module):
    """ Upsampling is fixed and done via 'nearest' """

    def __init__(self, opt):
        super(Generator, self).__init__()

        # Read options
        nonlinear_layer = get_nonlinear_layer(opt.nonlinearity)
        norm_layer = get_norm_layer(opt.norm)
        depth = int(log(opt.image_size // opt.latent_size, 2))

        layers = []

        # Deconde input into latent
        in_channels = len(opt.input_idx.split(','))
        assert opt.in_channels > in_channels, 'in_channels must be > num input_idx'

        out_channels = min(
                opt.num_channels * 2**depth, 
                opt.max_channels)

        bias = norm_layer == Identity or norm_layer == nn.InstanceNorm2d

        layers += [
            nn.Linear(opt.in_channels, out_channels * opt.latent_size**2, bias=bias),
            View(opt.latent_size),
            norm_layer(out_channels),
            nonlinear_layer(True)]

        # Upsampling decoding blocks
        for i in range(depth):
            in_channels = out_channels
            out_channels = min(
                opt.num_channels * 2**(depth-i-1), 
                opt.max_channels)
            layers += get_conv_block(
                in_channels,
                out_channels,
                nonlinear_layer,
                norm_layer, 
                'up', False)

        in_channels = out_channels
        layers += [nn.Conv2d(in_channels, 1, 3, 1, 1)]

        self.block = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, input):

        return self.block(input)
