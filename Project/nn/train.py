import numpy as np
from typing import List
import torch

from logger import *

CUDA = "cuda"
CPU = "cpu"


def nn_train(
    epoch: int,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    batch_size: int,
    sequential_model: torch.nn.Sequential,
    criterion: torch.nn.CrossEntropyLoss,
    optimiser: torch.optim.SGD,
    optimisation_steps: int,
    metrics: List[float],
    classes_in_output_vars: int,
) -> List:
    """
    One epoch of training

    Args:
    - epoch (int): Current epoch
    - train_data (torch.Tensor): Training data
    - train_labels (torch.Tensor): Training labels
    - batch_size (int): Batch size
    - sequential_model (torch.nn.Sequential): Model object
    - criterion (torch.nn.CrossEntropyLoss): Loss function
    - optimiser (torch.optim.SGD): Optimiser
    - optimisation_steps (int): Number of steps to train
    - classes_in_output_vars (List[float]): number of classes in each output variable - used for reshaping the output
    - metrics (int): List of metrics

    Returns:
    - List: List of metrics with the new metrics appended
    """

    # Select a random batch of data
    indices = np.random.randint(0, train_data.shape[0], size=batch_size)

    # Obtain the data and labels for the batch
    x = train_data[indices, :]

    log_trace("Shape of x: ", x.shape)

    # Make predictions
    y_pred = sequential_model(x)

    # As we (potentially) have multidimensional output, we need to reshape the predictions
    # or unflatten them based on classes_in_output_vars

    reshaped_preds = []
    sample = 5
    cum = 0
    for i, recovered_var_dim in enumerate(classes_in_output_vars):
        previous = classes_in_output_vars[i - 1] if i > 0 else 0
        cum += previous

        reshaped_pred = y_pred[:, previous : cum + recovered_var_dim]

        reshaped_preds.append(reshaped_pred)

    log_trace("Predictions: ", y_pred)
    log_trace("Shape of predictions: ", y_pred.shape)

    # True labels

    log_trace("True labels:")

    train_labels_batch = train_labels[indices]

    log_trace(train_labels_batch)
    log_trace("Shape of labels: ", train_labels_batch.shape)

    y_true = train_labels_batch

    # Convert to long tensor
    # y_true = y_true.long()

    log_trace("True labels tensor: ", y_true)
    log_trace("Shape of true labels tensor: ", y_true.shape)

    # Compute the loss

    # log_trace("y_pred: ", y_pred)
    log_trace("y_resh: ", reshaped_preds)
    log_trace("y_true: ", y_true)

    # Compute the loss for each recovered variable
    losses = []
    for dim, recovered_var_dim in enumerate(reshaped_preds):
        y_true_dim = y_true[:, dim].long()  # convert to int64

        log_trace("dim: ", dim, "recov: ", recovered_var_dim, "true: ", y_true_dim)
        log_trace(recovered_var_dim.shape, y_true_dim.shape)
        log_trace(recovered_var_dim.dtype, y_true_dim.dtype)
        loss = criterion(recovered_var_dim, y_true_dim)
        losses.append(loss)

    log_trace(loss)

    # Loss is the sum of all losses
    loss = sum(losses)

    # Zero the gradients
    optimiser.zero_grad()

    # Compute the gradients
    loss.backward()

    # Update the weights
    optimiser.step()

    if epoch % 100 == 0 or epoch == optimisation_steps - 1:
        # Find argument which maximises the prediction value
        # Again we need to reshape the predictions and take argmax of each recovered variable
        accuracies = []
        for dim, recovered_var_dim in enumerate(reshaped_preds):

            log_debug(f"Dim {dim}, Reshaped prefs: ", recovered_var_dim, recovered_var_dim.shape, recovered_var_dim.dtype)
            y_true_dim = y_true[:, dim].long()
            argmax = recovered_var_dim.argmax(dim=1)

            log_trace("Argmax: ", argmax)

            # Comparison
            comparison = argmax == y_true_dim

            log_trace("Comparison: ", comparison)

            train_accuracy = torch.mean((comparison).float())
            train_accuracy_value = train_accuracy.cpu().numpy()
            accuracies.append(train_accuracy_value)

        # Take average of all accuracies
        train_accuracy = sum(accuracies) / len(accuracies)
        log_info(f"Epoch: {epoch} / {optimisation_steps}, Train accuracy: {train_accuracy}, Loss: {loss.item()}")

        metrics.append([epoch, loss.item(), train_accuracy])

    return metrics
