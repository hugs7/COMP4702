"""
CNN Model class
"""

import torch.nn as nn
from typing import List
from colorama import Fore, Style


class CNN(nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        num_in_channels: int,
        hidden_channels: List[int],
        num_out_channels: int,
        kernel_sizes: List[int],
        pooling_kernel_sizes: List[int],
        stride: int,
        padding: int,
        num_fc_layers: int,
        num_classes: int,
    ):
        super(CNN, self).__init__()

        self.check_input_layeres(hidden_channels, kernel_sizes, pooling_kernel_sizes)

        self.conv_layers = nn.ModuleList()

        for i in range(num_conv_layers):
            kernel_size = kernel_sizes[i]
            num_hidden_channels = hidden_channels[i]

            if i == 0:
                conv = nn.Conv2d(
                    in_channels=num_in_channels,
                    out_channels=num_hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            else:
                in_channels = hidden_channels[i - 1]

                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )

            self.conv_layers.append(conv)

            # Activation function
            activation_func = nn.GELU()
            self.conv_layers.append(activation_func)

            # Pooling layer
            # Note pooling does not change the number of channels
            pooling_kernel_size = pooling_kernel_sizes[i]
            pooling_layer = nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=2)
            self.conv_layers.append(pooling_layer)

        # Flatten the output into one channel
        self.flatten = nn.Flatten()

        # # Fully connected linear layers
        # fc_layers = []

        # for i in range(num_fc_layers):
        #     if i == 0:
        #         fc = nn.Linear(32 * 7 * 7, 64)
        #     else:
        #         fc = nn.Linear(64, num_classes)

        #     fc_layers.append(fc)

        #     # Activation function
        #     activation_func = nn.GELU()
        #     fc_layers.append(activation_func)

        # self.fc1 = nn.Linear(32 * 7 * 7, 64)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(64, 10)

        # self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = []

        print(f"{Fore.LIGHTBLUE_EX}Input shape: {x.shape}{Style.RESET_ALL}")

        for i, layer in enumerate(self.conv_layers):
            print(f"{Fore.LIGHTMAGENTA_EX}Layer operation: {layer}{Style.RESET_ALL}")
            conv_output = layer(x)
            print(
                f"{Fore.LIGHTGREEN_EX}Layer {i} output shape: {conv_output.shape}{Style.RESET_ALL}"
            )

            x_conv.append(conv_output)
            x = conv_output

        print("Convolutional layers output shape", x_conv[-1].shape)
        x = self.flatten(x_conv[-1])

        print("Flattened shape", x.shape)

        return x

    def check_input_layeres(self, hidden_channels, kernel_sizes, pooling_kernel_sizes):
        """
        Checks that the number of hidden channels, kernel sizes, and pooling kernel sizes are the same
        As they are used in the same loop
        """
        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError("Hidden channels and kernel sizes must be the same length")

        if len(hidden_channels) != len(pooling_kernel_sizes):
            raise ValueError(
                "Hidden channels and pooling kernel sizes must be the same length"
            )

        if len(kernel_sizes) != len(pooling_kernel_sizes):
            raise ValueError(
                "Kernel sizes and pooling kernel sizes must be the same length"
            )
