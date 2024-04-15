import numpy as np
from typing import List
import torch


def nn_train(epoch: int, train_data: torch.Tensor, train_labels: torch.Tensor, batch_size: int,
             sequential_model: torch.nn.Sequential, criterion: torch.nn.CrossEntropyLoss,
             optimiser: torch.optim.SGD, optimisation_steps: int, metrics: List[float], log: bool = False) -> List:
    """
    One epoch of training

    Args:
        epoch: Current epoch
        train_data: Training data
        train_labels: Training labels
        batch_size: Batch size
        sequential_model: Model
        criterion: Loss function
        optimiser: Optimiser
        optimisation_steps: Number of steps to train
        metrics: List of metrics

    Returns:
        New list of metrics
    """
    # Select a random batch of data
    indices = np.random.randint(0, train_data.shape[0], size=batch_size)

    # Obtain the data and labels for the batch
    x = train_data[indices, :]
    if log:
        print(x)
        print("Shape of x: ", x.shape)
        print()

    # Make predictions
    y_pred = sequential_model(x)
    if log:
        print("Predictions: ", y_pred)
        print("Shape of predictions: ", y_pred.shape)
        print()

    # True labels
    if log:
        print("True labels:")
    train_labels_batch = train_labels[indices]
    if log:
        print(train_labels_batch)
        print("Shape of labels: ", train_labels_batch.shape)
        print()
    y_true = train_labels_batch
    # Convert to long tensor
    # y_true = y_true.long()
    if log:
        print("True labels tensor: ", y_true)
        print("Shape of true labels tensor: ", y_true.shape)
        print()

    # Compute the loss
    loss = criterion(y_pred, y_true)

    # Zero the gradients
    optimiser.zero_grad()

    # Compute the gradients
    loss.backward()

    # Update the weights
    optimiser.step()

    if epoch % 100 == 0:
        if log:
            print(f"Epoch: {epoch} / {optimisation_steps}")
            print("Loss: ", loss.item())

        train_accuracy = torch.mean((y_pred.argmax(dim=1) == y_true).float())

        metrics.append([epoch, loss.item(), train_accuracy.cpu().numpy()])

    return metrics
