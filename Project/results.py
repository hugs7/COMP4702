from plot.plot import lineplot
import numpy as np

from logger import *


def show_training_results(metrics):
    log_title("Plotting training results")
    metrics = np.asarray(metrics)

    # Training Loss
    training_accuracy_x = metrics[:, 0]
    training_accuracy_y = metrics[:, 2]
    train_metrics = (training_accuracy_x, training_accuracy_y)

    # Validation Loss
    # validation_loss_x = metrics[:, 0]
    # validation_loss_y = metrics[:, 3]
    # validation_metrics = (validation_loss_x, validation_loss_y)

    lineplot("Epoch", "Accuracy", train_metrics)
