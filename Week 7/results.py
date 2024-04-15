from plot import lineplot
import numpy as np


def show_training_results(metrics):

    metrics = np.asarray(metrics)

    # Training Loss
    training_loss_x = metrics[:, 0]
    training_loss_y = metrics[:, 2]
    train_metrics = (training_loss_x, training_loss_y)

    # Validation Loss
    # validation_loss_x = metrics[:, 0]
    # validation_loss_y = metrics[:, 3]
    # validation_metrics = (validation_loss_x, validation_loss_y)

    lineplot("Epoch", "Loss", train_metrics)
