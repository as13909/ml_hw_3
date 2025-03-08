import hw3_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels, height, width, use_silu=False, use_layernorm=False):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and
                          the number of channels of conv layers of Block.
            height: height of channel
            width: width of channel
            use_silu: if true, use SiLU, otherwise use ReLU
            use_layernorm: if true, use layer norm with [num_channels, height, width] features,
                           otherwise use batch norm with num_channels features
        """
        super(Block, self).__init__()

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        pass

    def set_param(self, kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_1: a (C, C, 3, 3) tensor, kernels of the first conv layer.
            bn1_weight: if using layer norm, a (C, H, W) tensor; if using batch norm, a (C,) tensor.
            bn1_bias: if using layer norm, a (C, H, W) tensor; if using batch norm, a (C,) tensor.
            kernel_2: a (C, C, 3, 3) tensor, kernels of the second conv layer.
            bn2_weight: if using layer norm, a (C, H, W) tensor; if using batch norm, a (C,) tensor.
            bn2_bias: if using layer norm, a (C, H, W) tensor; if using batch norm, a (C,) tensor.
        """
        pass


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, height, width, use_silu=False, use_layernorm=False, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            height: height of channel
            width: width of channel
            use_silu: if true, use SiLU, otherwise use ReLU
            use_layernorm: if true, use layer norm with [num_channels, (height+1)//2, (width+1)//2] features,
                           otherwise use batch norm with num_channels features
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        pass

    def set_param(self, kernel_0, bn0_weight, bn0_bias,
                  kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias,
                  fc_weight, fc_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_0: a (C, 1, 3, 3) tensor, kernels of the conv layer
                      before the building block.
            bn0_weight: if using layer norm, a (C, H, W) tensor, weight of the layer norm layer
                        before the building block.
                        if using batch norm, a (C,) tensor, weight of the batch norm layer
                        before the building block.
            bn0_bias: if using layer norm, a (C, H, W) tensor, bias of the layer norm layer
                      before the building block.
                      if using batch norm, a (C,) tensor, bias of the batch norm layer
                      before the building block.
            fc_weight: a (10, C) tensor
            fc_bias: a (10,) tensor
        See the docstring of Block.set_param() for the description
        of other arguments.
        """
        pass


def fit_and_validate(net, optimizer, loss_func, train, val, n_epochs=30, batch_size=16):
    """
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param train: a torch.utils.data.Dataset
    @param val: a torch.utils.data.Dataset
    @param n_epochs: the number of epochs over which to do gradient descent
    @param batch_size: the number of samples to use in each batch of gradient descent
    @return train_epoch_loss, validation_epoch_loss: two arrays of length n_epochs+1, containing the mean loss at the beginning of training and after each epoch
    """
    net.eval() #put the net in evaluation mode
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    val_dl = torch.utils.data.DataLoader(val)
    with torch.no_grad():
        # compute the mean loss on the training set at the beginning of iteration
        losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
        train_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        # TODO compute the validation loss and store it in a list
    for _ in range(n_epochs):
        net.train() #put the net in train mode
        # TODO 
        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            # TODO compute the train and validation losses and store it in a list
    return train_el, val_el