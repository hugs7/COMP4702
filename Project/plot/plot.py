from typing import Tuple
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import torch


def lineplot(x_label: str, y_label: str, *args: Tuple[np.ndarray, np.ndarray, str]):
    """
    Plots a line plot of the given data.

    Parameters:
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - args (Tuple[np.ndarray, np.ndarray, str]): The data to plot. Each tuple contains 
        - x (np.ndarray): The x values.
        - y (np.ndarray): The y values.
        - label (str): The label for the line.
    """

    sb.set_context("talk")
    sb.set_style("dark")

    for (x, y, label) in (args):
        sb.lineplot(x=x, y=y, label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_gelu():
    geLu = torch.nn.GELU()

    input = torch.arange(-6, 6, step=0.1)
    output = geLu(input)

    lineplot("X", "GeLU(X)", (input, output, "GeLU"))

    return input


def plot_sigmoid(input):
    # Sigmoid activation function

    sigmoid = torch.nn.Sigmoid()

    # Input remains the same
    output = sigmoid(input)

    lineplot("X", "Sigmoid(X)", (input, output, "Sigmoid"))
