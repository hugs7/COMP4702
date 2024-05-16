from plot.plot import lineplot
import numpy as np

from logger import *


def show_training_results(metrics):
    log_title("Plotting training results")
    metrics = np.asarray(metrics)

    # Training Loss
    epochs_x = metrics[:, 0]
    training_accuracy_y = metrics[:, 2]
    val_accuracy_y = metrics[:, 4]

    train_metrics = (epochs_x, training_accuracy_y, "Training")
    val_metrics = (epochs_x, val_accuracy_y, "Validation")

    lineplot("Epoch", "Accuracy", train_metrics, val_metrics)
