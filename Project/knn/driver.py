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
import utils


def run_knn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_labels: List[str],
    y_labels: List[List[str]],
    unique_classes: List[List[str]],
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
    - unique_classes (List[List[str]]): The unique classes in each target variable.
    - num_classes_in_vars (List[int]): The number of classes in each target variable.
    """

    log_title("Start of knn model driver...")

    log_debug(f"Number of classes in each output variable: {num_classes_in_vars}")

    # For multi-variable classification, we need to create a knn classifier for each output variable
    # These are independent of each other.

    log_debug(f"Creating a knn classifier for each of the {len(y_labels)} output variables")

    knn_classifiers = []

    # Dictionary of output variable: (y_test_true, test_predictions, train_accuracy, test_accuracy)
    results = {}

    for i, var_y_labels in enumerate(y_labels):
        # ======== Train KNN classifier for this output variable ========
        log_title(f"Output variable {i}: {var_y_labels}")
        y_var_unique_classes = unique_classes[i]
        log_info(f"Unique classes for output variable {i}: {y_var_unique_classes}")

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

        log_debug(f"Obtaining unique classes for output variable {i}...")

        log_debug(f"Training KNN classifier for output variable {i}...")

        # ======== Obtain results from test and train data ========

        test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

        log_debug(f"KNN classifier for output variable {i} trained")

        results[i] = (var_y_test, test_preds, train_accuracy, test_accuracy)

        log_trace(f"Output variable {i} results: {results[i]}")

        log_line(level="DEBUG")

        # ======== Compute variable importance ========

        log_info(f"Computing variable importance for output variable {i}...")

        # Calculate permutation importance
        var_importance = variable_importance.compute_average_feature_importance(X_test, 5, k, 50)
        sorted_importance = sorted(enumerate(var_importance), key=lambda x: x[1], reverse=True)

        log_info(f"Variable importance for output variable {i}")

        max_feature_name_length = max([len(feature_name) for feature_name in X_labels])
        for idx, importance in sorted_importance:
            feature_name = X_labels[idx]
            log_info(f"    Feature {idx:<2} {feature_name:<{max_feature_name_length+3}}: Importance {importance:.4f}")

        # ======== Plot Decision Boundaries ========

        log_info(f"Plotting decision boundaries for output variable {i}...")

        delta = 6
        delta = min(delta, len(X_labels))
        # Plot decision regions for the top delta features
        top_delta_feature_idxs = [idx for idx, _ in sorted_importance[:delta]]
        top_5_feature_cols = [X_labels[idx] for idx in top_delta_feature_idxs]
        log_info(f"Top {delta} feature indices: {top_delta_feature_idxs}")

        # Calculate the total number of plots
        feature_combinations = list(itertools.combinations(top_delta_feature_idxs, 2))
        log_trace(f"Feature combinations: {feature_combinations}")
        num_feature_pairs = len(feature_combinations)
        log_debug(f"Total number of plots: {num_feature_pairs}")

        # Determine the number of rows and columns for the square grid
        num_plots_per_row = math.ceil(math.sqrt(num_feature_pairs))
        num_plots_per_col = math.ceil(num_feature_pairs / num_plots_per_row)

        # Create a square grid of subplots
        fig, axs = plt.subplots(num_plots_per_row, num_plots_per_col, figsize=(10, 6))

        # Flatten the axs array to iterate over it easily
        # Flatten only if there is more than one row
        if num_plots_per_row > 1:
            axs = axs.flatten()

        # Iterate over each pair of input variables
        plot_index = 0

        for i, feature_pair in enumerate(feature_combinations):
            log_info(f"Plotting decision boundary for feature pair {feature_pair}. Progress: {i} / {num_feature_pairs}")

            # Get the current axes
            # If only 1 plot, axs is not an array
            if num_plots_per_row > 1:
                plt.sca(axs[plot_index])
            else:
                plt.sca(axs)

            # Generate and plot decision regions for the current pair of input variables
            subplot = knn_classifier.plot_decision_regions(
                test_preds, feature_pair, X_labels, y_var_unique_classes, show_plot=False, resolution=1
            )
            # Set title for each subplot
            feature_label_x = X_labels[feature_pair[0]]
            feature_label_y = X_labels[feature_pair[1]]
            subplot.set_title(f"DB for features {feature_label_x} and {feature_label_y}")
            subplot.set_xlabel(feature_label_x)
            subplot.set_ylabel(feature_label_y)

            # Add subplot to the list of plots
            if num_plots_per_row > 1:
                axs[plot_index] = subplot
            else:
                axs = subplot

            # Increment plot index
            plot_index += 1

        log_debug("All decision boundary plots generated")

        # Hide empty subplots
        num_axes = len(axs) if isinstance(axs, np.ndarray) else 1
        for j in range(num_feature_pairs, num_axes):
            axs[j].axis("off")

        log_line(level="DEBUG")
        log_debug("X Test points:")
        X_test_important_features = X_test[:, top_delta_feature_idxs]
        log_debug(utils.np_to_pd(X_test_important_features, top_5_feature_cols))
        log_line(level="DEBUG")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the decision boundary plots for the current KNN classifier
        plt.show()

    return results
