from torch import nn
from math import log
from src.utils import get_nonlinear_layer, get_norm_layer, get_conv_block
from src.utils import weights_init, View



class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # Read options
        nonlinear_layer = get_nonlinear_layer('LeakyReLU')
        norm = opt.norm if opt.norm_dis else 'None'
        norm_layer = get_norm_layer(norm)
        
        depth = int(log(opt.image_size // opt.latent_size, 2))
        assert depth >= 1, 'image_size must be >= 8'

        in_channels = 1
        out_channels = opt.num_channels

        layers = get_conv_block(
            in_channels,
            out_channels,
            nonlinear_layer, 
            norm_layer,
            'none', False,
            opt.kernel_size)

        # Get all model blocks
        for i in range(depth):

            # Set number of channels for conv
            in_channels = out_channels
            out_channels = min(out_channels * 2, opt.max_channels)

            # Define downsampling block
            layers += get_conv_block(
                in_channels, 
                out_channels,
                nonlinear_layer, 
                norm_layer,
                'down', False,
                opt.kernel_size)

        # Output single number pred
        layers += [
            View(),
            nn.Linear(out_channels * opt.latent_size**2, opt.num_preds)]

        self.block = nn.Sequential(*layers)

        # Initialize weights
        self.apply(weights_init)

    def forward(self, input):

        return self.block(input)
