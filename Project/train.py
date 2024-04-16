import numpy as np
from typing import List
import torch


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
    log: bool = False,
) -> List:
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
        classes_in_output_vars: List[int]: number of classes in each output variable - used for reshaping the output
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

    # As we (potentially) have multidimensional output, we need to reshape the predictions
    # or unflatten them based on classes_in_output_vars

    reshaped_preds = []
    sample = 5
    cum = 0
    for i, recovered_var_dim in enumerate(classes_in_output_vars):
        previous = classes_in_output_vars[i - 1] if i > 0 else 0
        cum += previous

        reshaped_pred = y_pred[:, previous:cum+recovered_var_dim]

        reshaped_preds.append(reshaped_pred)

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
    if log:
        # print("y_pred: ", y_pred)
        print("y_resh: ", reshaped_preds)
        print("y_true: ", y_true)

    # Compute the loss for each recovered variable
    losses = []
    for dim, recovered_var_dim in enumerate(reshaped_preds):
        y_true_dim = y_true[:, dim].long()  # convert to int64
        if log:
            print("dim: ", dim, "recov: ",
                  recovered_var_dim, "true: ", y_true_dim)
            print(recovered_var_dim.shape, y_true_dim.shape)
            print(recovered_var_dim.dtype, y_true_dim.dtype)
        loss = criterion(recovered_var_dim, y_true_dim)
        losses.append(loss)
        if log:
            print(loss)

    # Loss is the sum of all losses
    loss = sum(losses)

    # Zero the gradients
    optimiser.zero_grad()

    # Compute the gradients
    loss.backward()

    # Update the weights
    optimiser.step()

    if epoch % 100 == 0 or epoch == optimisation_steps - 1:
        if log:
            print(f"Epoch: {epoch} / {optimisation_steps}")
            print("Loss: ", loss.item())

        # Find argument which maximises the prediction value
        # Again we need to reshape the predictions and take argmax of each recovered variable
        accuracies = []
        for dim, recovered_var_dim in enumerate(reshaped_preds):
            if log:
                print(f"Dim {dim}, Reshaped prefs: ",
                      recovered_var_dim, recovered_var_dim.shape, recovered_var_dim.dtype)
            y_true_dim = y_true[:, dim].long()
            argmax = recovered_var_dim.argmax(dim=1)
            if log:
                print("Argmax: ", argmax)

            # Comparison
            comparison = argmax == y_true_dim
            if log:
                print("Comparison: ", comparison)

            train_accuracy = torch.mean((comparison).float())
            train_accuracy_value = train_accuracy.cpu().numpy()
            accuracies.append(train_accuracy_value)

        # Take average of all accuracies
        train_accuracy = sum(accuracies) / len(accuracies)
        print("Train accuracy: ", train_accuracy)

        metrics.append([epoch, loss.item(), train_accuracy])

    return metrics
