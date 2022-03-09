from estimate_ranks import estimate_ranks
from tensorly.contrib.sparse.decomposition import partial_tucker
import numpy as np
import torch
from torch import nn

def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer,
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = \
        partial_tucker(layer.weight.data, \
            modes=[0, 1], rank=ranks, init='svd')
    # print(f"core = {core}, first = {first}, last = {last}")

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data
    first = torch.tensor(np.array(first))
    first_layer.weight.data = \
        torch.transpose(input=first, dim0=1, dim1=0).unsqueeze(-1).unsqueeze(-1)
    last = torch.tensor(np.array(last))
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core = torch.tensor(np.array(core))
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    layer = nn.Sequential(*new_layers)
    return layer