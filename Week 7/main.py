"""
Main file for week 7 prac
11/04/2024
"""

import os
import torch
import sys
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import torchvision


def welcome():
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
    print("Numpy version: ", np.__version__)
    print("Seaborn version: ", sb.__version__)

    print("-" * 40)


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


def plot_gelu():
    geLu = torch.nn.GELU()

    input = torch.arange(-6, 6, step=0.1)
    output = geLu(input)

    lineplot("X", "GeLU(X)", input, output)


def plot_sigmoid():
    # Sigmoid activation function

    sigmoid = torch.nn.Sigmoid()

    # Input remains the same
    output = sigmoid(input)

    lineplot("X", "Sigmoid(X)", input, output)


def create_sequential_model(dim_input: int, dim_output: int, hidden_layer_dims: List[int]) -> torch.nn.Sequential:
    """
    Create a sequential model

    Args:
        dim_input: Dimension of the input
        dim_output: Dimension of the output
        hidden_layers: List of hidden layers

    Returns:
        Sequential model
    """

    print("Input dimension: ", dim_input)

    hiddens = [dim_input, *hidden_layer_dims]

    torch_layers = []

    # Create a linear layer and feed it through the activation function
    for i in range(len(hiddens) - 1):
        # Create linear layer from i to i+1
        linear_layer = torch.nn.Linear(hiddens[i], hiddens[i+1])

        # Create activation function
        activation = torch.nn.GELU()

        # Add to the list
        torch_layers.append(linear_layer)
        torch_layers.append(activation)

    # Add the final output layer
    final_hidden_layer = hiddens[-1]
    output_layer = torch.nn.Linear(final_hidden_layer, dim_output)
    torch_layers.append(output_layer)

    # Turn the list into a sequential model
    sequential_model = torch.nn.Sequential(*torch_layers)

    return sequential_model


def main():
    welcome()

    # Check length of command line arguments
    if len(sys.argv) > 1:
        print("Arguments: ", sys.argv[1:])

        if sys.argv[1] == "plot":
            # Activation function

            plot_gelu()
            plot_sigmoid()

    # --- Model Creation ---

    dim_in = 2
    dim_out = 1
    hidden_layer_dims = [4, 8, 16]
    sequential_model = create_sequential_model(
        dim_in, dim_out, hidden_layer_dims)

    print(sequential_model)

    # --- Loss Function ---
    # For classification problems, usually use cross entropy loss
    # For regression problems, usually use mean squared error

    # Both accuracy and loss are important

    # --- Optimiser ---

    # Stochastic gradient descent is the most popular optimiser when
    # training a neural network

    # We only compute the gradient on a subset of the data because this
    # is far faster than computing the gradient on the entire dataset.
    # Given a large enough subset, the gradient will be a good approximation
    # of the true gradient.

    # --- Dataset ---

    folder_of_script = os.path.dirname(__file__)
    data_folder = os.path.join(folder_of_script, "data")

    # Create data folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    CIFAR10_data_folder = os.path.join(data_folder, "CIFAR10")

    # Check if data already exists
    is_downloaded = os.path.exists(CIFAR10_data_folder)
    download = not is_downloaded

    CIFAR10_train = torchvision.datasets.CIFAR10(
        CIFAR10_data_folder, download=download, train=True, transform=True)
    CIFAR10_validation = torchvision.datasets.CIFAR10(
        CIFAR10_data_folder, download=download, train=False, transform=True)

    print("Training data stats")
    print("Shape:", CIFAR10_train.data.shape)
    print("Classes:", CIFAR10_train.classes)

    # Flatten the dataset and normalise it
    # We flatten the dataset because we need to pass in a 1D array
    # not a 3D array (which is (X, Y, Channels))
    train_data = (CIFAR10_train.data.reshape(
        (-1, 32*32*3))/255.0).astype(np.float32)
    train_labels = np.asarray(CIFAR10_train.targets)

    validation_data = (CIFAR10_validation.data.reshape(
        (-1, 32*32*3))/255.0).astype(np.float32)
    validation_labels = np.asarray(CIFAR10_validation.targets)

    print("Training data shape:", train_data.shape)
    print("Validation data shape:", validation_data.shape)

    # Classification problem so define cross entropy loss
    # and stochastic gradient descent optimiser

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    learning_rate = 1e-3
    optimiser = torch.optim.SGD(
        sequential_model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    main()
