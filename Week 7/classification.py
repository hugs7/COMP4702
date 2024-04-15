
from typing import List, Tuple
import model
from colorama import Fore, Style
import os
import numpy as np
import torch
import torchvision
import train
from results import show_training_results


def preprocess_data(train_data: np.ndarray, validation_data: np.ndarray, dim_input: int,
                    normalising_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    return_data = []

    for data in [train_data, validation_data]:
        new_data = (data.data.reshape((-1, dim_input)) /
                    normalising_factor).astype(np.float32)
        labels = np.asarray(data.targets)

        return_data.append(new_data)
        return_data.append(labels)

    return return_data


def classification_model(dataset_name: str, dim_input: int, dim_output: int, hidden_layer_dims: List[int],
                         normalising_factor: float) -> None:
    print(f"{Fore.GREEN}Creating model for {dataset_name} dataset{Style.RESET_ALL}")

    sequential_model = model.create_sequential_model(
        dim_input, dim_output, hidden_layer_dims)

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

    model_data_folder = os.path.join(data_folder, dataset_name)

    # Check if data already exists
    is_downloaded = os.path.exists(model_data_folder)
    download = not is_downloaded

    dataset_map = {
        "CIFAR10": torchvision.datasets.CIFAR10,
        "MNIST": torchvision.datasets.MNIST,
        "FashionMNIST": torchvision.datasets.FashionMNIST,
        "KMNIST": torchvision.datasets.KMNIST,
        "EMNIST": torchvision.datasets.EMNIST,
        "QMNIST": torchvision.datasets.QMNIST,
    }

    if dataset_name not in dataset_map:
        raise ValueError(
            f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")

    # Get the dataset
    mappped_dataset = dataset_map[dataset_name]

    train_data = mappped_dataset(
        model_data_folder, download=download, train=True, transform=True)
    validation_data = mappped_dataset(
        model_data_folder, download=download, train=False, transform=True)

    print("Training data stats")
    print("Shape:", train_data.data.shape)
    print("Classes:", train_data.classes)

    # Flatten the dataset and normalise it
    # We flatten the dataset because we need to pass in a 1D array
    # not a 3D array (which is (X, Y, Channels))
    train_data, train_labels, validation_data, validation_labels = preprocess_data(
        train_data, validation_data, dim_input, normalising_factor)

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

    show_training_results(metrics)
