"""
Handles making predictions and computing accuracy for the neural network model.
Hugo Burton
"""

import torch
from typing import List

from logger import *
import utils


def make_predictions(
    X: torch.Tensor,
    sequential_model: torch.nn.Sequential,
    num_classes_in_vars: List[int],
    train_data: bool = True,
) -> torch.Tensor:
    """
    Makes predictions given an X tensor and model. Reshapes the predictions
    to match the number of classes in each output variable.

    Args:
    - X (torch.Tensor): Input tensor
    - sequential_model (torch.nn.Sequential): Model object
    - num_classes_in_vars (List[int]): The number of classes in each output variable
    - train_data (bool): Whether the data is training or not. If false, data is validation.
                         Only affects headings in log output

    Returns:
    - torch.Tensor: Reshaped Predictions
    """

    data_type = "training" if train_data else "validation"

    # Make predictions
    y_pred = sequential_model(X)

    log_trace(f"Predictions of {data_type}: {y_pred}")
    log_debug(f"Shape of predictions: {y_pred.shape}")

    # As we (potentially) have multidimensional output, we need to split
    # the output of the model into (still one-hot encoded) predictions for EACH variable

    reshaped_preds = []
    max_classes = max(num_classes_in_vars)
    cum_index_offset = 0
    for i, num_classes in enumerate(num_classes_in_vars):
        log_debug(f"Variable Index: {i}, recovered_var_dim: {num_classes}")

        reshaped_pred = y_pred[:,
                               cum_index_offset: cum_index_offset + num_classes]

        # Pad the reshaped prediction with zeros to match the maximum number of classes
        reshaped_pred = torch.nn.functional.pad(
            reshaped_pred, (0, max_classes - num_classes))

        log_trace(f"Reshaped prediction: ", reshaped_pred)
        reshaped_preds.append(reshaped_pred)

        cum_index_offset += num_classes

    # Convert reshaped_preds to a tensor
    reshaped_preds = torch.stack(reshaped_preds, dim=1)

    log_line(level="DEBUG")
    log_trace("Reshaped predictions: ", reshaped_preds)
    log_debug("Shape of reshaped predictions: ", reshaped_preds.shape)

    return reshaped_preds


def compute_accuracy(num_output_vars: int, y_true: torch.Tensor, train_preds: torch.Tensor, train_data: bool) -> float:
    """
    Computes the accuracy given the true labels and the predictions

    Args:
    - num_output_vars (int): Number of output variables
    - y_true (torch.Tensor): True labels
    - train_preds (torch.Tensor): Predictions
    - train_data (bool): Whether the data is training or not. If false, data is validation.

    Returns:
    - float: The accuracy
    """

    data_type = "training" if train_data else "validation"

    log_debug(f"Computing accuracy for {data_type} data...")

    # Find argument which maximises the prediction value
    # Again we need to reshape the predictions and take argmax of each recovered variable
    var_accuracies = []
    for var in range(num_output_vars):
        log_debug(f"Output Variable: {var}")

        y_true_var = y_true[:, var]
        y_pred_var = train_preds[:, var]

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

        overall_accuracy = torch.mean((comparison).float())
        var_accuracy_val = utils.tensor_to_cpu(overall_accuracy, detach=False)
        var_accuracies.append(var_accuracy_val)

    # Take average of all accuracies
    overall_accuracy = sum(var_accuracies) / len(var_accuracies)

    log_debug(f"Overall accuracy ({data_type}): {overall_accuracy}")

    return overall_accuracy
