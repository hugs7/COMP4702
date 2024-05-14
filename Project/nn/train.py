import numpy as np
from typing import List
import torch

from logger import *

CUDA = "cuda"
CPU = "cpu"


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
    loss_weights: List[float] = None,
) -> List:
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
    - num_classes_in_vars (List[int]): The number of classes in each output variable - used for recovering the true classification from the flattened output of the model
    - metrics (int): List of metrics
    - loss_weights (List[float]): Weights for the loss function. If not provided, defaults to 1.0 for each variable

    Returns:
    - List: List of metrics with the new metrics appended
    """

    # Select a random batch of data
    indices = np.random.randint(0, X_train.shape[0], size=batch_size)

    # Obtain the data and labels for the batch
    x = X_train[indices, :]

    log_trace("Data: ", x)
    log_debug("Shape of x: ", x.shape)
    log_line(level="TRACE")
    log_trace("Labels: ", y_train[indices])
    log_debug("Shape of labels: ", y_train[indices].shape)
    # NEED TO FIX LABELS dimensionality

    # Make predictions
    y_pred = sequential_model(x)

    log_trace("Predictions: ", y_pred)
    log_debug("Shape of predictions: ", y_pred.shape)

    # As we (potentially) have multidimensional output, we need to split
    # the output of the model into (still one-hot encoded) predictions for EACH variable

    reshaped_preds = []
    num_output_vars = len(num_classes_in_vars)
    max_classes = max(num_classes_in_vars)
    cum_index_offset = 0
    for i, num_classes in enumerate(num_classes_in_vars):
        log_debug(f"Variable Index: {i}, recovered_var_dim: {num_classes}")

        reshaped_pred = y_pred[:, cum_index_offset : cum_index_offset + num_classes]

        # Pad the reshaped prediction with zeros to match the maximum number of classes
        reshaped_pred = torch.nn.functional.pad(reshaped_pred, (0, max_classes - num_classes))

        log_trace(f"Reshaped prediction: ", reshaped_pred)
        reshaped_preds.append(reshaped_pred)

        cum_index_offset += num_classes

    # Convert reshaped_preds to a tensor
    reshaped_preds = torch.stack(reshaped_preds, dim=1)

    log_line(level="DEBUG")
    log_trace("Reshaped predictions: ", reshaped_preds)
    log_debug("Shape of reshaped predictions: ", reshaped_preds.shape)

    # True labels
    y_true = y_train[indices]

    log_trace("Y True Labels: ", y_true)
    log_debug("Y True shape: ", y_true.shape)

    # Compute the loss
    log_debug("Computing loss...")

    # Compute the loss for each recovered variable
    losses = []
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

    # Loss is the sum of all losses
    total_loss_tensor = sum(losses)
    log_debug("Total Loss: ", total_loss_tensor)

    # Zero the gradients
    log_debug("Zeroing gradients...")
    optimiser.zero_grad()

    # Compute the gradients
    log_trace("Computing gradients...")
    total_loss_tensor.backward()

    # Update the weights
    log_trace("Optimising...")
    optimiser.step()

    log_line(level="DEBUG")

    if epoch % 100 == 0 or epoch == optimisation_steps - 1:
        # Find argument which maximises the prediction value
        # Again we need to reshape the predictions and take argmax of each recovered variable
        accuracies = []
        for var in range(num_output_vars):
            log_debug(f"Output Variable: {var}")

            y_true_var = y_true[:, var]
            y_pred_var = reshaped_preds[:, var]

            log_trace("Y True Var: ", y_true_var)
            log_trace("Y Pred Var: ", y_pred_var)

            log_debug("Y True Var shape: ", y_true_var.shape)
            log_debug("Y Pred Var shape: ", y_pred_var.shape)

            # Compute the argmax of the predictions
            argmax = y_pred_var.argmax(dim=1)

            # Recover the true class from the one-hot encoding of y_true_var to obtain the index of the true class
            y_true_var = y_true_var.argmax(dim=1)

            log_debug("Argmax (predictions): ", argmax)
            log_debug("True class: ", y_true_var)

            # Comparison
            comparison = argmax == y_true_var

            log_debug("Comparison: ", comparison)

            train_accuracy = torch.mean((comparison).float())
            train_accuracy_value = train_accuracy.cpu().numpy()
            accuracies.append(train_accuracy_value)

        # Take average of all accuracies
        train_accuracy = sum(accuracies) / len(accuracies)
        log_info(f"Epoch: {epoch} / {optimisation_steps}, Train accuracy: {train_accuracy}, Loss: {loss_var.item()}")

        metrics.append([epoch, loss_var.item(), train_accuracy])

    return metrics
