import torch
from torch import nn
from torch.nn.functional import sigmoid



def get_norm_layer(norm):

    norm = norm.lower()
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'none':
        norm_layer = Identity

    return norm_layer

def get_nonlinear_layer(nonlinearity):

    nonlinearity = nonlinearity.lower()
    if nonlinearity == 'relu':
        nonlinear_layer = nn.ReLU
    elif nonlinearity == 'leakyrelu':
        nonlinear_layer = lambda inplace: nn.LeakyReLU(0.2, inplace)
    elif nonlinearity == 'swish':
        nonlinear_layer = Swish

    return nonlinear_layer

def get_conv_block(
    in_channels, 
    out_channels, 
    nonlinear_layer = nn.ReLU,
    norm_layer = None,
    mode = 'same ',
    sequential = False,
    kernel_size = 3):

    # Prepare layers and parameters
    norm_layer = Identity if norm_layer is None else norm_layer
    bias = norm_layer == Identity or norm_layer == nn.InstanceNorm2d

    layers = []
    
    # If block does upsampling
    if mode == 'up':
        layers += [nn.Upsample(scale_factor=2, mode='nearest')]

    # If block does downsampling
    stride = 2 if mode == 'down' else 1
    if not kernel_size % 2:
        padding = (kernel_size - (stride==2)) // 2
    else:
        padding = kernel_size // 2

    layers += [
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            bias=bias),
        norm_layer(out_channels),
        nonlinear_layer(True)]

    if sequential:
        block = nn.Sequential(*layers)
    else:
        block = layers

    return block

def weights_init(module):
    """ Custom weights initialization """

    classname = module.__class__.__name__

    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * sigmoid(input)

    def __repr__(self):
        
        return ('{name}()'.format(name=self.__class__.__name__))


class Identity(nn.Module):
    def __init__(self, num_channels=None):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):

        return ('{name}()'.format(name=self.__class__.__name__))


class View(nn.Module):

    def __init__(self, size=4):
        super(View, self).__init__()

        self.size = size

    def forward(self, x):

        if len(x.size()) == 2:

            # Input is from linear layer -- unravel it
            x = x.view(x.size(0), -1, self.size, self.size)
        elif x.size(2) == x.size(3) and x.size(2) == self.size:

            # Input from conv layer -- ravel it
            x = x.view(x.size(0), -1)
        else:
            assert False, 'input shape does not match ravel size'

        return x

    def __repr__(self):

        return ('{name}()'.format(name=self.__class__.__name__))
