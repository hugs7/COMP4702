"""
Driver script for Neural Network model
06/05/2024
Hugo Burton
"""

import os
from typing import List, Tuple
import torch
import torch.optim as optim
import numpy as np

import results
import file_helper
import decode_data
from logger import *

import model.classifier as classifier

from plot.plot import plot_multivar_decision_regions

from nn import train, nn_model
from nn.train import CUDA, CPU
from nn.model_handler import save_model, read_model, NN_MODEL_NAME
from nn.predict import make_predictions, compute_accuracy


NN_MODEL_NORMALISING_FACTOR = 1.0
NN_MODEL_HIDDEN_LAYER_DIMS = [100, 150, 100]


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to tensor with float32 data type.

    Args:
    - data (ndarray): The numpy array to convert.

    Returns:
    - torch.Tensor: The converted tensor.
    """
    return torch.tensor(data, dtype=torch.float32)


def get_device() -> torch.device:
    """
    Get the device to train the model on.

    Returns:
    - torch.device: The device to train the model on.
    """

    log_trace("Fetching device...")
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    log_debug(f"Device fetched {device}")
    return device


def get_loss_weights(num_classes_in_vars: List[int]) -> List[float]:
    """
    Get the loss weights for the model.

    Args:
    - num_classes_in_vars (List[int]): The number of classes in each output variable.

    Returns:
    - List[float]: The loss weights.
    """

    log_trace("Calculating loss weights...")

    # If there is only one output variable, the loss weight is 1.0
    if len(num_classes_in_vars) == 1:
        log_debug("Only one output variable. Loss weight is 1.0")
        return [1.0]

    # Calculate the loss weight for each output variable
    loss_weights = [1.0 / num_classes for num_classes in num_classes_in_vars]
    log_debug("Loss weights calculated", loss_weights)
    return loss_weights


def train_val_data_to_tensor_and_device(device: torch.device, X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts data to tensor and moves it to the device.

    Args:
    - device (torch.device): The device to move the data to.
    - X_train (ndarray): Training data features.
    - y_train (ndarray): Training data target variable.
    - X_validation (ndarray): Validation data features.
    - y_validation (ndarray): Validation data target variable.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The training and validation data features and target variable as tensors.
    """

    log_title("Convert data to tensors...")

    X_train = to_tensor(X_train)
    y_train = to_tensor(y_train)

    X_validation = to_tensor(X_validation)
    y_validation = to_tensor(y_validation)

    log_info("Data converted to tensors")

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_validation = X_validation.to(device)
    y_validation = y_validation.to(device)

    log_info("Training data shape:", X_train.shape, "x", y_train.shape)
    log_info("Validation data shape:",
             X_validation.shape, "x", y_validation.shape)

    return X_train, y_train, X_validation, y_validation


def test_data_to_tensor_and_device(device: torch.device, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts test data to tensor and moves it to the device.

    Args:
    - device (torch.device): The device to move the data to.
    - X_test (ndarray): The test data features.
    - y_test (ndarray): The test data target variable.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: The test data features and target variable as tensors.
    """

    log_title("Convert test data to tensors...")

    X_test = to_tensor(X_test)
    y_test = to_tensor(y_test)

    log_info("Test data converted to tensors")

    # Move data to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    log_info("Test data shape:", X_test.shape, "x", y_test.shape)

    return X_test, y_test


def get_model_save_folders(nn_folder_path: str) -> Tuple[str, str]:
    """
    Get the paths of the final model and checkpoints folders.

    Args:
    - nn_folder_path (str): The path of the neural network model.

    Returns:
    - Tuple[str, str]: The paths of the final model and checkpoints folders.
    """

    final_model_folder = os.path.join(nn_folder_path, "final_model")
    checkpoints_folder = os.path.join(nn_folder_path, "checkpoints")
    return final_model_folder, checkpoints_folder


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

    log_title(f"Start of nn model driver for dataset: {dataset_name}...")

    log_info(f"Configuring folders")

    final_model_folder, checkpoints_folder = get_model_save_folders(
        nn_folder_path)
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

    # Hyperparameters
    epochs = int(2e4)
    batch_size = 1000
    learning_rate = 2e-4
    weight_decay = 0.01
    loss_weights = get_loss_weights(num_classes_in_vars)

    device = get_device()

    X_train, y_train, X_validation, y_validation = train_val_data_to_tensor_and_device(
        device, X_train, y_train, X_validation, y_validation)

    # Instantiate the model and move it to the specified device
    sequential_model = nn_model.create_sequential_model(
        dim_input, dim_output_flattened, NN_MODEL_HIDDEN_LAYER_DIMS).to(device)

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

    # Save the final model
    log_info("Saving final model...")
    save_model(final_model_folder, sequential_model,  metrics)

    log_info("Final Model saved")
    log_line(level="INFO")

    # Show training results
    log_title("Training Results")
    results.show_training_results(metrics)

    # Decision boundary plots


def run_saved_nn_model(
    nn_folder_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_labels: List[str],
    y_labels: List[List[str]],
    unique_classes: List[List[str]],
    num_classes_in_vars: List[int],
    final_model: bool = True,
    checkpoint_num: int = None,
) -> None:
    """
    Run a saved neural network model on the test data.

    Args:
    - nn_folder_path (str): The path of the neural network model.
    - X_test (ndarray): Test data features.
    - y_test (ndarray): Test data target variable.
    - X_labels (List[str]): The names of the (input) features.
    - y_labels (List[List[str]]): The names of each class within each
                                    target variable. Of which there can be multiple                 
    - unique_classes (List[List[str]]): The unique classes in each target variable.
    - num_classes_in_vars (List[int]): The number of classes in each target variable.
    - final_model (bool): If true the final model will be used, otherwise the checkpoint model will be used. If true, ignore checkpoint_num.
    - checkpoint_num (int): The checkpoint number to use. If None, the final model will be used. Should not be None in conjunction with final_model=False.
    """

    if not final_model and checkpoint_num is None:
        log_error(
            "Checkpoint number is None and final model is False. Please provide a checkpoint number or specify to use the final model.")
        return

    final_model_folder, checkpoints_folder = get_model_save_folders(
        nn_folder_path)

    save_folder = final_model_folder if final_model else checkpoints_folder

    log_debug(f"Save folder: {save_folder}")
    file_helper.show_files_in_folder(save_folder, level="DEBUG")

    model_save_path = None
    if final_model:
        model_save_path = os.path.join(
            save_folder, f"{NN_MODEL_NAME}_final_model.pt")
    else:
        # Load the checkpoint model
        model_save_path = os.path.join(
            save_folder, f"{NN_MODEL_NAME}_checkpoint_{checkpoint_num}.pt")

    log_debug(f"Checking model exists at path: {model_save_path}")

    # Check if the model file exists
    if not file_helper.file_exists(model_save_path):
        log_error(
            f"Model file does not exist: {model_save_path}. The files in the folder are:")
        file_helper.show_files_in_folder(save_folder)
        return

    log_debug("Reading model...")
    model_obj = read_model(model_save_path)

    # Load the model state dict
    state_dict = model_obj["model_state_dict"]
    log_debug(f"State dict: {state_dict}")

    # Load data

    device = get_device()

    X_test, y_test = test_data_to_tensor_and_device(
        device, X_test, y_test)

    # Ouptut dimension is sum of classes in each output variable
    # because of one hot encoding. Flatten the list of classes
    dim_output_flattened = sum(num_classes_in_vars)
    # Although this is different from the number of output variables, we will need to
    # recover the true classification from the one-hot encoding by reshaping the output.

    dim_input = X_test.shape[1]  # Should be the same as the training data
    normalising_factor = 1.0
    loss_weights = get_loss_weights(num_classes_in_vars)

    # Instantiate the model and move it to the specified device
    sequential_model = nn_model.create_sequential_model(
        dim_input, dim_output_flattened, NN_MODEL_HIDDEN_LAYER_DIMS).to(device)

    log_info(f"Loading model state dict into model...")
    sequential_model.load_state_dict(state_dict)

    log_info(f"Model state dict loaded from {model_save_path}")

    log_info(
        f"Making predictions on test data. Test data shape: {X_test.shape}")

    test_preds = make_predictions(
        X_test, sequential_model, num_classes_in_vars, classifier.TESTING)

    log_debug(f"Predictions made on test data", test_preds)
    log_info(f"Predictions shape: {test_preds.shape}")

    # Compute the accuracy of the model
    num_output_vars = len(num_classes_in_vars)
    train_accuracy = compute_accuracy(
        num_output_vars, y_test, test_preds, classifier.TESTING)

    # Decode the one-hot encoded test predictions
    test_preds_arg_max = decode_data.decode_one_hot_encoded_data(test_preds)

    log_info(f"Accuracy of the model: {train_accuracy}")

    # Decision boundary plots

    for i, var_y in enumerate(y_labels):
        log_title(f"Output variable {i}: {var_y}")

        test_preds_var = test_preds_arg_max[i]
        var_classes = unique_classes[i]

        log_info(
            f"Unique classes for output variable {i}: {var_classes}")
        log_debug(
            f"Test predictions for output variable {i}:\n{test_preds_var}")

        # plot_multivar_decision_regions()
