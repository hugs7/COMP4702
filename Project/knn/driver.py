""""
Driver script for knn classification model
Hugo Burton
06/05/2024
"""

import itertools
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math

from knn import knn_model
from knn import variable_importance
from logger import *


def run_knn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_labels: List[str],
    y_labels: List[List[str]],
    num_classes_in_vars: List[int],
    k: int = 5,
) -> None:
    """
    Driver script for k-nearest neighbours classification model. Takes in training, test data along with labels and trains
    a k-nearest neighbours model on the data.

    Args:
    - X_train (ndarray): Training data features.
    - y_train (ndarray): Training data target variable (not one-hot-encoded).
    - X_test (ndarray): Testing data features.
    - y_test (ndarray): Testing data target variable (not one-hot-encoded).
    - X_labels (List[str]): The names of the (input) features.
    - y_labels (List[List[str]]): The names of each class within each target variable. Of which there can be multiple
    - num_classes_in_vars (List[int]): The number of classes in each target variable.
    """

    log_title("Start of knn model driver...")

    log_info(f"Number of classes in each output variable: {num_classes_in_vars}")

    # For multi-variable classification, we need to create a knn classifier for each output variable
    # These are independent of each other.

    log_debug(f"Creating a knn classifier for each of the {len(y_labels)} output variables")

    knn_classifiers = []

    # Dictionary of output variable: (y_test_true, test_predictions, train_accuracy, test_accuracy)
    results = {}

    for i, var_y_labels in enumerate(y_labels):
        # ======== Train KNN classifier for this output variable ========
        log_title(f"Output variable {i} classes: {var_y_labels}")

        # Get slice of y_train and y_test for this output variable
        var_y_train = y_train[:, i]
        var_y_test = y_test[:, i]

        log_trace(f"y_train_var:\n{var_y_train}")
        log_trace(f"y_test_var:\n{var_y_test}")

        log_debug(f"y_train_var shape: {var_y_train.shape}")
        log_debug(f"y_test_var shape: {var_y_test.shape}")

        knn_classifier = knn_model.KNNClassify(X_train, var_y_train, X_test, var_y_test, X_labels, var_y_labels, k=k)

        # Add the classifier to the list
        knn_classifiers.append(knn_classifier)

        log_debug(f"KNN classifier for output variable {i} created")

        log_info(f"Training KNN classifier for output variable {i}...")

        # ======== Obtain results from test and train data ========

        test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

        log_info(f"KNN classifier for output variable {i} trained")

        results[i] = (var_y_test, test_preds, train_accuracy, test_accuracy)

        log_debug(f"Output variable {i} results: {results[i]}")

        log_line(level="DEBUG")

        # ======== Compute variable importance ========

        log_title(f"Computing variable importance for output variable {i}...")

        # Calculate permutation importance
        var_importance = variable_importance.compute_average_feature_importance(X_test, 10, k, 100)

        log_info(f"Variable importance for output variable {i}: {var_importance}")

        sorted_importance = sorted(enumerate(var_importance), key=lambda x: x[1], reverse=True)
        log_debug(f"Sorted variable importance: {sorted_importance}")

        for idx, importance in sorted_importance:
            log_debug(f"Feature {idx}: Importance {importance}")

        # ======== Plot Decision Boundaries ========

        # Plot decision regions for each pair of features that are considered important by our threshold, alpha

        # Calculate the total number of plots
        total_plots = len(list(itertools.combinations(range(X_train.shape[1]), 2)))

        # Determine the number of rows and columns for the square grid
        num_plots_per_row = math.ceil(math.sqrt(total_plots))
        num_plots_per_col = math.ceil(total_plots / num_plots_per_row)

        # Create a square grid of subplots
        fig, axs = plt.subplots(num_plots_per_row, num_plots_per_col, figsize=(15, 15))

        # Flatten the axs array to iterate over it easily
        axs = axs.flatten()

        # Iterate over each pair of input variables
        plot_index = 0
        feature_pairs = itertools.combinations(range(X_train.shape[1]), 2)
        feature_pairs = list(feature_pairs)
        log_debug(f"Feature pairs: {feature_pairs}")
        num_feature_pairs = len(feature_pairs)
        log_info(f"Number of feature pairs: {num_feature_pairs}")

        for i, feature_pair in enumerate(feature_pairs):
            log_info(f"Plotting decision boundary for feature pair ({feature_pair}). Progress: {i} / {num_feature_pairs}")

            # Get the current axes
            plt.sca(axs.flatten()[plot_index])

            # Generate and plot decision regions for the current pair of input variables
            subplot = knn_classifier.plot_decision_regions(test_preds, feature_pair, show_plot=False, resolution=1)
            # Set title for each subplot
            subplot.set_title(f"Decision Boundary for Feature Pair {feature_pair}")

            # Add subplot to the list of plots
            axs[plot_index] = subplot

            # Increment plot index
            plot_index += 1

        log_info("All decision boundary plots generated")

        # Hide empty subplots
        for j in range(total_plots, len(axs)):
            axs[j].axis("off")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the decision boundary plots for the current KNN classifier
        plt.show()

    return results
