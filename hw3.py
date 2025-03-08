import hw3_utils as utils
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

        Arguments:
            num_channels: the number of channels of the input to Block, and
                          the number of channels of conv layers of Block.
            height: height of channel
            width: width of channel
            use_silu: if true, use SiLU, otherwise use ReLU
            use_layernorm: if true, use layer norm with [num_channels, height, width] features,
                           otherwise use batch norm with num_channels features
        """
        super(Block, self).__init__()
        self.use_layernorm = use_layernorm

        # Define convolutional layers
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Define normalization layers
        if use_layernorm:
            self.norm1 = nn.LayerNorm([num_channels, height, width])
            self.norm2 = nn.LayerNorm([num_channels, height, width])
        else:
            self.norm1 = nn.BatchNorm2d(num_channels)
            self.norm2 = nn.BatchNorm2d(num_channels)

        # Define activation function
        self.activation = nn.SiLU() if use_silu else nn.ReLU()

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        # Apply first conv layer, normalization, and activation
        f = self.conv1(x)
        f = self.norm1(f)
        f = self.activation(f)

        # Apply second conv layer and normalization
        f = self.conv2(f)
        f = self.norm2(f)

        # Residual connection and activation
        out = self.activation(x + f)
        return out

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
        # Set parameters for conv layers
        self.conv1.weight = nn.Parameter(kernel_1)
        self.conv2.weight = nn.Parameter(kernel_2)

        # Set parameters for normalization layers
        if self.use_layernorm:
            self.norm1.weight = nn.Parameter(bn1_weight)
            self.norm1.bias = nn.Parameter(bn1_bias)
            self.norm2.weight = nn.Parameter(bn2_weight)
            self.norm2.bias = nn.Parameter(bn2_bias)
        else:
            self.norm1.weight = nn.Parameter(bn1_weight)
            self.norm1.bias = nn.Parameter(bn1_bias)
            self.norm2.weight = nn.Parameter(bn2_weight)
            self.norm2.bias = nn.Parameter(bn2_bias)

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
        self.use_silu = use_silu
        self.use_layernorm = use_layernorm

        # Initial convolutional layer
        self.conv0 = nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Normalization layer selection
        if use_layernorm:
            self.norm0 = nn.LayerNorm([num_channels, (height + 1) // 2, (width + 1) // 2])
        else:
            self.norm0 = nn.BatchNorm2d(num_channels)
        
        # Activation function selection
        self.activation = nn.SiLU() if use_silu else nn.ReLU()

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Residual block
        self.block = Block(num_channels, (height + 1) // 4, (width + 1) // 4, use_silu=use_silu, use_layernorm=use_layernorm)

        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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
        # Set parameters for the initial conv layer
        self.conv0.weight = nn.Parameter(kernel_0)

        # Set parameters for normalization layer
        if self.use_layernorm:
            self.norm0.weight = nn.Parameter(bn0_weight)
            self.norm0.bias = nn.Parameter(bn0_bias)
        else:
            self.norm0.weight = nn.Parameter(bn0_weight)
            self.norm0.bias = nn.Parameter(bn0_bias)

        # Set parameters for the block
        self.block.set_param(kernel_1, bn1_weight, bn1_bias, kernel_2, bn2_weight, bn2_bias)

        # Set parameters for the fully connected layer
        self.fc.weight = nn.Parameter(fc_weight)
        self.fc.bias = nn.Parameter(fc_bias)

def fit_and_validate(net, optimizer, loss_func, train, val, n_epochs=30, batch_size=16):
    """
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param train: a torch.utils.data.Dataset
    @param val: a torch.utils.data.Dataset
    @param n_epochs: the number of epochs over which to do gradient descent
    @param batch_size: the number of samples to use in each batch of gradient descent
    @return train_epoch_loss, validation_epoch_loss: two arrays of length n_epochs+1, 
            containing the mean loss at the beginning of training and after each epoch
    """
    # Data loaders
    train_dl = torch.utils.data.DataLoader(train, batch_size, shuffle=False)
    val_dl = torch.utils.data.DataLoader(val, batch_size, shuffle=False)

    # Initial evaluation
    net.eval()
    with torch.no_grad():
        # Compute initial train loss
        losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
        train_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]

        # Compute initial validation loss
        losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
        val_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]

    # Training loop
    for _ in range(n_epochs):
        net.train()
        for X, Y in train_dl:
            loss, _ = utils.loss_batch(net, loss_func, X, Y, optimizer)

        # Evaluate at the end of each epoch
        net.eval()
        with torch.no_grad():
            # Compute train loss
            losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
            train_epoch_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))

            # Compute validation loss
            losses, nums = zip(*[utils.loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
            val_epoch_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))

    return train_epoch_loss, val_epoch_loss
