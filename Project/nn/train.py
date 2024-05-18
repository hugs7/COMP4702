import numpy as np
from typing import List, Tuple
import torch
import utils

from logger import *

from nn.model_handler import save_model
from nn.predict import compute_accuracy, make_predictions

CUDA = "cuda"
CPU = "cpu"


def compute_loss(
    reshaped_preds: torch.Tensor,
    y_true: torch.Tensor,
    criterion: torch.nn.CrossEntropyLoss,
    num_classes_in_vars: List[int],
    loss_weights: List[float] = None,
    train_data: bool = True,
) -> torch.Tensor:
    """
    Computes the loss for the model

    Args:
    - reshaped_preds (torch.Tensor): Predictions from the model
    - y_true (torch.Tensor): True labels
    - criterion (torch.nn.CrossEntropyLoss): Loss function
    - num_classes_in_vars (List[int]): The number of classes in each output variable
    - loss_weights (List[float]): Weights for the loss function. If not provided, defaults to 1.0 for each variable
    - train_data (bool): Whether the data is training or not. If false, data is validation.

    Returns:
    - torch.Tensor: The total computed loss for the model as a tensor
    """

    data_type = "training" if train_data else "validation"

    log_debug(f"Computing loss for {data_type} data...")

    # Compute the loss for each recovered variable
    losses = []
    num_output_vars = len(num_classes_in_vars)

    if loss_weights is None:
        loss_weights = [1.0 for _ in range(num_output_vars)]
    else:
        assert len(loss_weights) == num_output_vars, (
            "Loss weights must be provided for each variable. Received: ",
            len(loss_weights),
            "Expected: ",
            num_output_vars,
        )

    # Compute the loss for each variable
    for var in range(num_output_vars):
        log_debug(f"Variable: {var}")
        y_true_var = y_true[:, var]
        y_pred_var = reshaped_preds[:, var]

        log_trace("Y True Var: ", y_true_var)
        log_trace("Y Pred Var: ", y_pred_var)

        log_debug("Y True Var shape: ", y_true_var.shape)
        log_debug("Y Pred Var shape: ", y_pred_var.shape)

        log_debug("Y True Var dtype: ", y_true_var.dtype)
        log_debug("Y Pred Var dtype: ", y_pred_var.dtype)

        loss_var = criterion(y_pred_var, y_true_var)

        log_debug("Loss: ", loss_var)
        log_debug("Loss shape: ", loss_var.shape)
        log_debug("Loss dtype: ", loss_var.dtype)

        loss_weighted = loss_var * loss_weights[var]
        losses.append(loss_weighted)

    log_line(level="DEBUG")

    log_debug("Losses:", losses)

    total_loss_tensor = sum(losses)

    # Loss is the sum of all losses
    log_debug(f"Total Loss ({data_type}): {total_loss_tensor}")

    return total_loss_tensor


def nn_train(
    epoch: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    batch_size: int,
    sequential_model: torch.nn.Sequential,
    criterion: torch.nn.CrossEntropyLoss,
    optimiser: torch.optim.SGD,
    optimisation_steps: int,
    metrics: List[float],
    num_classes_in_vars: int,
    checkpoints_folder: str,
    loss_weights: List[float] = None,
) -> List[Tuple[int, float, float, float, float]]:
    """
    One epoch of training

    Args:
    - epoch (int): Current epoch
    - X_train (torch.Tensor): Training data
    - y_train (torch.Tensor): Training labels. These should already be flattened into [batch size, num_variables, num_classes for each variable]
    - X_validation (torch.Tensor): Validation data
    - y_validation (torch.Tensor): Validation labels
    - batch_size (int): Batch size
    - sequential_model (torch.nn.Sequential): Model object
    - criterion (torch.nn.CrossEntropyLoss): Loss function
    - optimiser (torch.optim.SGD): Optimiser
    - optimisation_steps (int): Number of steps to train
    - metrics (int): List of metrics
    - num_classes_in_vars (List[int]): The number of classes in each output variable - used for recovering the true classification from the flattened output of the model
    - checkpoints_folder (str): The folder to save the model checkpoints
    - loss_weights (List[float]): Weights for the loss function. If not provided, defaults to 1.0 for each variable

    Returns:
    - List[Tuple]: List of metrics with the new metrics appended. Each tuple contains:
        - epoch (int): Current epoch
        - train_loss (float): Training loss
        - train_accuracy (float): Training accuracy
        - validation_loss (float): Validation loss
        - validation_accuracy (float): Validation accuracy
    """

    # Select a random batch of data
    indices = np.random.randint(0, X_train.shape[0], size=batch_size)

    # Obtain the data and labels for the batch
    X = X_train[indices, :]

    log_trace("Data: ", X)
    log_debug("Shape of x: ", X.shape)
    log_line(level="TRACE")
    log_trace("Labels: ", y_train[indices])
    log_debug("Shape of labels: ", y_train[indices].shape)

    train_preds = make_predictions(
        X, sequential_model, num_classes_in_vars, train_data=True)

    # True labels
    y_true = y_train[indices]

    log_trace("Y True Labels: ", y_true)
    log_debug("Y True shape: ", y_true.shape)

    # Compute the loss of the training data
    train_loss_tensor = compute_loss(
        train_preds, y_true, criterion, num_classes_in_vars, loss_weights, train_data=True)

    # Zero the gradients
    log_debug("Zeroing gradients...")
    optimiser.zero_grad()

    # Compute the gradients
    log_trace("Computing gradients...")
    train_loss_tensor.backward()

    # Update the weights
    log_trace("Optimising...")
    optimiser.step()

    log_line(level="DEBUG")

    num_output_vars = len(num_classes_in_vars)
    if epoch % 100 == 0 or epoch == optimisation_steps - 1:
        train_loss_cpu = utils.tensor_to_cpu(train_loss_tensor, detach=True)

        log_debug("Train loss: ", train_loss_cpu)

        log_debug("Making predictions on validation data...")
        val_preds = make_predictions(
            X_validation, sequential_model, num_classes_in_vars, train_data=False)
        validation_loss_tensor = compute_loss(
            val_preds, y_validation, criterion, num_classes_in_vars, loss_weights, train_data=False)
        validation_loss_cpu = utils.tensor_to_cpu(
            validation_loss_tensor, detach=True)

        log_debug("Validation loss: ", validation_loss_cpu)

        train_accuracy = compute_accuracy(
            num_output_vars, y_true, train_preds, train_data=True)
        validation_accuracy = compute_accuracy(
            num_output_vars, y_validation, val_preds, train_data=False)

        opt_steps_digits = len(str(optimisation_steps))

        log_info(
            f"Epoch: {epoch:<{opt_steps_digits}} / {optimisation_steps:{opt_steps_digits}}, Train accuracy: {train_accuracy:.4f}, " +
            f"Loss: {train_loss_cpu:.4f}, Validation accuracy: {validation_accuracy:.4f}, Loss: {validation_loss_cpu:.4f}")

        metrics.append([epoch, train_loss_cpu, train_accuracy,
                       validation_loss_cpu, validation_accuracy])

        if epoch % 1000 == 0 or epoch == optimisation_steps - 1:
            # Save model checkpoint
            save_model(checkpoints_folder, sequential_model,
                       metrics, epoch // 1000)

    return metrics
