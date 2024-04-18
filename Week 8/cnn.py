"""
CNN Model class
"""

import torch.nn as nn
from typing import List
from colorama import Fore, Style
from utils import prod


class CNN(nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        input_dimension_x: int,
        input_dimension_y: int,
        num_in_channels: int,
        hidden_channels: List[int],
        kernel_sizes: List[int],
        pooling_kernel_sizes: List[int],
        stride: int,
        padding: int,
        num_fc_layers: int,
        fc_channels: List[int],
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
        self.fc_layers = nn.ModuleList()

        if num_fc_layers != len(fc_channels):
            raise ValueError(
                "Number of fully connected layers must be the same as the number of channels"
            )

        for i in range(num_fc_layers):
            out_channels = fc_channels[i]

            if i == 0:
                final_conv_layer_channels = hidden_channels[-1]
                final_conv_layer_x = int(
                    (input_dimension_x * input_dimension_y)
                    / (prod(pooling_kernel_sizes) * num_conv_layers * stride)
                )
                print(
                    f"{Fore.LIGHTYELLOW_EX}Final conv layer x: {final_conv_layer_x}{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.LIGHTYELLOW_EX}Final conv layer channels: {final_conv_layer_channels}{Style.RESET_ALL}"
                )
                in_channels = int(final_conv_layer_channels * final_conv_layer_x)
                in_channels = 392
                print(
                    f"{Fore.LIGHTYELLOW_EX}In channels: {in_channels}{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.LIGHTYELLOW_EX}Out channels: {out_channels}{Style.RESET_ALL}"
                )
                fc = nn.Linear(in_channels, out_channels)
            else:
                in_channels = fc_channels[i - 1]
                fc = nn.Linear(in_channels, out_channels)

            self.fc_layers.append(fc)

            # Activation function
            activation_func = nn.GELU()
            self.fc_layers.append(activation_func)

        # Final output layer
        self.output_layer = nn.Linear(fc_channels[-1], num_classes)

    def forward(self, x):
        x_conv = []

        print(f"{Fore.LIGHTBLUE_EX}Input shape: {x.shape}{Style.RESET_ALL}")

        for i, layer in enumerate(self.conv_layers):
            print(f"{Fore.LIGHTMAGENTA_EX}Layer operation: {layer}{Style.RESET_ALL}")
            x = layer(x)
            print(
                f"{Fore.LIGHTGREEN_EX}Layer {i} output shape: {x.shape}{Style.RESET_ALL}"
            )
            x_conv.append(x)

        print("Convolutional layers output shape", x_conv[-1].shape)

        # Flatten the output of the convolutional layers
        x = self.flatten(x_conv[-1])

        print("Flattened shape", x.shape)

        # Fully connected linear layers
        x_linear = []
        for i, layer in enumerate(self.fc_layers):
            print(
                f"{Fore.LIGHTGREEN_EX}Fully connected layer {i} operation: {layer}{Style.RESET_ALL}"
            )
            x = layer(x)
            print(
                f"{Fore.LIGHTGREEN_EX}Fully connected layer {i} output shape: {x.shape}{Style.RESET_ALL}"
            )

            x_linear.append(x)

        # Output layer
        x = self.output_layer(x_linear[-1])

        print(f"{Fore.LIGHTRED_EX}Output shape: {x.shape}{Style.RESET_ALL}")

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
