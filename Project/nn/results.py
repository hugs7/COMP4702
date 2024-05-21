from typing import List
from plot.plot import lineplot
import numpy as np
import os

from logger import *


def show_training_results(metrics: List[List[float]], plots_save_path: str) -> None:
    """
    Plots the training results

    Args:
    - metrics (List[List[float]]): The training metrics
    - plots_save_path (str): The path to save the plots to

    Returns:
    - None
    """

    log_title("Plotting training results")
    metrics = np.asarray(metrics)

    # Training Loss
    epochs_x = metrics[:, 0]
    training_accuracy_y = metrics[:, 2]
    val_accuracy_y = metrics[:, 4]

    train_metrics = (epochs_x, training_accuracy_y, "Training")
    val_metrics = (epochs_x, val_accuracy_y, "Validation")

    plot_file_name = f"nn_train_val_accuracies.png"
    plot_file_path = os.path.join(plots_save_path, plot_file_name)

    lineplot("Epoch", "Accuracy", "Neural Network Train & Validation Accuracies",
             plot_file_path, train_metrics, val_metrics)
