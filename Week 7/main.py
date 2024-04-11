"""
Main file for week 7 prac
11/04/2024
"""

import torch
import sys
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def lineplot(x_label: str, y_label: str, x: np.ndarray, y: np.ndarray):
    """
    Plot a line graph
    """

    sb.set_context("talk")
    sb.set_style("dark")

    sb.lineplot(x=x, y=y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def main():
    print("Week 7: PyTorch")

    print("Torch version: ", torch.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available")
        print("Device count: ", torch.cuda.device_count())
        print("Current device: ", torch.cuda.current_device())
        print("Device name: ", torch.cuda.get_device_name())
    else:
        print("CUDA is not available")

    print("Python version: ", sys.version)

    # --- Define a Model ---

    # Activation function

    geLu = torch.nn.GELU()

    input = torch.arange(-6, 6, step=0.1)
    output = geLu(input)

    lineplot("X", "GeLU(X)", input, output)

    # Sigmoid activation function

    sigmoid = torch.nn.Sigmoid()

    # Input remains the same
    output = sigmoid(input)

    lineplot("X", "Sigmoid(X)", input, output)


if __name__ == "__main__":
    main()
