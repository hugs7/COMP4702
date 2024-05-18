"""
Driver script for Neural Network model
06/05/2024
Hugo Burton
"""

import os
from typing import List
import torch
import torch.optim as optim
import numpy as np

import results
import file_helper
from logger import *

from nn import train, nn_model
from nn.train import CUDA, CPU


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to tensor with float32 data type.

    Args:
    - data (ndarray): The numpy array to convert.

    Returns:
    - torch.Tensor: The converted tensor.
    """
    return torch.tensor(data, dtype=torch.float32)


def run_nn_model(
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_labels: List[str],
    y_labels: List[List[str]],
    unique_classes: List[List[str]],
    num_classes_in_vars: List[int],
    nn_folder_path: str,
    plots_folder_path: str = None,
) -> None:
    """
    Driver script for Neural Network model. Takes in training, test data along with labels and trains
    a neural network model on the data.

    Args:
    - dataset_name (str): The name of the dataset.
    - X_train (ndarray): Training data features.
    - y_train (ndarray): Training data target variable.
    - X_validation (ndarray): Validation data features.
    - y_validation (ndarray): Validation data target variable.
    - X_labels (List[str]): The names of the (input) features.
    - y_labels (List[List[str]]): The names of each class within each target variable. Of which there can be multiple
    - unique_classes (List[List[str]]): The unique classes in each target variable.
    - num_classes_in_vars (List[int]): The number of classes in each target variable.
    - nn_folder_path (str): The path of the neural network model.
    - plots_folder_path (str): The path to save the plots to.
    """

    log_title("Start of nn model driver...")

    log_info(f"Configuring folders")

    final_model_folder = os.path.join(nn_folder_path, "final_model")
    checkpoints_folder = os.path.join(nn_folder_path, "checkpoints")
    file_helper.make_folder_if_not_exists(final_model_folder)
    file_helper.make_folder_if_not_exists(checkpoints_folder)

    log_info(f"Folders configured")

    log_info(
        f"Clearing contents of folders {final_model_folder} and {checkpoints_folder}")
    file_helper.delete_folder_contents(final_model_folder)
    file_helper.delete_folder_contents(checkpoints_folder)

    log_info(f"Folders cleared")
    log_line(level="INFO")

    log_info(
        f"Number of classes in each output variable: {num_classes_in_vars}")

    # Ouptut dimension is sum of classes in each output variable
    # because of one hot encoding. Flatten the list of classes
    dim_output_flattened = sum(num_classes_in_vars)
    # Although this is different from the number of output variables, we will need to
    # recover the true classification from the one-hot encoding by reshaping the output.

    # Assume the train and test data have the same dimension along axis 1
    dim_input = X_train.shape[1]
    normalising_factor = 1.0
    hidden_layer_dims = [100, 150, 100]

    # Hyperparameters
    epochs = int(2e4)
    batch_size = 1000
    learning_rate = 2e-4
    weight_decay = 0.01

    loss_weights = [1.0 for _ in range(len(num_classes_in_vars))]

    log_title("Convert data to tensors...")

    X_train = to_tensor(X_train)
    y_train = to_tensor(y_train)

    X_validation = to_tensor(X_validation)
    y_validation = to_tensor(y_validation)

    log_info("Data converted to tensors")

    # ----- Device ------
    # Move model to GPU if available
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_validation = X_validation.to(device)
    y_validation = y_validation.to(device)

    log_info("Training data shape:", X_train.shape, "x", y_train.shape)
    log_info("Validation data shape:",
             X_validation.shape, "x", y_validation.shape)

    # Instantiate the model and move it to the specified device
    sequential_model = nn_model.create_sequential_model(
        dim_input, dim_output_flattened, hidden_layer_dims).to(device)

    log_info(f"Model: \n{sequential_model}\n")

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

    # Classification problem so define cross entropy loss
    # and stochastic gradient descent optimiser

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    # optimiser = torch.optim.SGD(
    #     sequential_model.parameters(), lr=learning_rate)
    optimiser = optim.Adam(sequential_model.parameters(),
                           lr=learning_rate, weight_decay=weight_decay)

    # --- Training Loop ---

    metrics = []
    for i in range(int(epochs)):
        metrics = train.nn_train(
            i,
            X_train,
            y_train,
            X_validation,
            y_validation,
            batch_size,
            sequential_model,
            criterion,
            optimiser,
            epochs,
            metrics,
            num_classes_in_vars,
            checkpoints_folder,
            loss_weights,
        )

    results.show_training_results(metrics)
