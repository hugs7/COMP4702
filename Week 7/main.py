"""
Main file for week 7 prac
11/04/2024
"""

import os
import torch
import sys
import seaborn as sb
import numpy as np
from typing import List
import torchvision

from plot import lineplot, plot_gelu, plot_sigmoid
import model
import train


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


def show_training_results(metrics):
    metrics = np.asarray(metrics)

    # Training Loss
    training_loss_x = metrics[:, 0]
    training_loss_y = metrics[:, 2]

    # Validation Loss
    validation_loss_x = metrics[:, 0]
    validation_loss_y = metrics[:, 3]

    lineplot("Epoch", "Loss", (training_loss_x, training_loss_y),
             (validation_loss_x, validation_loss_y))


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
    dim_out = 10
    hidden_layer_dims = [100, 100]
    sequential_model = model.create_sequential_model(
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

    # --- Training Loop ---

    batch_size = 256                   # Data points per batch to train on
    optimisation_steps = int(1e4)      # Number of batches to train on

    metrics = []
    for i in range(optimisation_steps):
        metrics = train.nn_train(i, train_data, train_labels, batch_size,
                                 sequential_model, criterion, optimiser, optimisation_steps, metrics)


if __name__ == "__main__":
    main()
