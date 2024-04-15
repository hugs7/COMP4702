"""
Main file for week 7 prac
11/04/2024
"""

import os
import torch
import sys
import numpy as np
import torchvision

from plot import plot_gelu, plot_sigmoid
from classification import classification_model
import train
from welcome import welcome


def main():
    welcome()

    # Check length of command line arguments
    if len(sys.argv) > 1:
        print("Arguments: ", sys.argv[1:])

        if sys.argv[1] == "plot":
            # Activation function

            input = plot_gelu()
            plot_sigmoid(input)

        exit(0)

    # --- Model Creation ---

    dim_in = 32*32*3
    normalising_factor = 255.0
    dim_out = 10
    hidden_layer_dims = [100, 100]
    classification_model("CIFAR10", dim_in, dim_out,
                         hidden_layer_dims, normalising_factor)


if __name__ == "__main__":
    main()
