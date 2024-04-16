
from typing import Tuple
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import torch


def lineplot(x_label: str, y_label: str, *args: Tuple[np.ndarray, np.ndarray]):
    """
    Plot a line graph
    """

    sb.set_context("talk")
    sb.set_style("dark")

    for i, (x, y) in enumerate(args):
        sb.lineplot(x=x, y=y, label=f"Line {i+1}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_gelu():
    geLu = torch.nn.GELU()

    input = torch.arange(-6, 6, step=0.1)
    output = geLu(input)

    lineplot("X", "GeLU(X)", (input, output))

    return input


def plot_sigmoid(input):
    # Sigmoid activation function

    sigmoid = torch.nn.Sigmoid()

    # Input remains the same
    output = sigmoid(input)

    lineplot("X", "Sigmoid(X)", (input, output))
