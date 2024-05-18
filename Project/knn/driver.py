""""
Driver script for knn classification model
Hugo Burton
06/05/2024
"""

from typing import List
import numpy as np


from knn import knn_model
from logger import *


def run_knn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_labels: List[str],
    y_labels: List[List[str]],
    unique_classes: List[List[str]],
    num_classes_in_vars: List[int],
    ordered_predictor_indicies: np.ndarray,
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
    - ordered_predictor_indicies (ndarray): 2D array of indices of the predictors in descending order of importance. Rows are output
                                            variables and columns are the ordered indices of the predictors for that output variable.
    - k (int): The number of neighbours to consider.
    """

    log_title("Start of knn model driver...")

    log_debug(
        f"Number of classes in each output variable: {num_classes_in_vars}")

    # For multi-variable classification, we need to create a knn classifier for each output variable
    # These are independent of each other.

    log_debug(
        f"Creating a knn classifier for each of the {len(y_labels)} output variables")

    knn_classifiers = []

    # Dictionary of output variable: (y_test_true, test_predictions, train_accuracy, test_accuracy)
    results = {}

    for i, var_y in enumerate(y_labels):
        # ======== Train KNN classifier for this output variable ========
        log_title(f"Output variable {i}: {var_y}")
        y_var_unique_classes = unique_classes[i]
        log_info(
            f"Unique classes for output variable {i}: {y_var_unique_classes}")

        # Get slice of y_train and y_test for this output variable
        var_y_train = y_train[:, i]
        var_y_test = y_test[:, i]

        log_trace(f"y_train_var:\n{var_y_train}")
        log_trace(f"y_test_var:\n{var_y_test}")

        log_debug(f"y_train_var shape: {var_y_train.shape}")
        log_debug(f"y_test_var shape: {var_y_test.shape}")

        knn_classifier = knn_model.KNNClassify(
            X_train, var_y_train, X_test, var_y_test, X_labels, y_var_unique_classes, k=k)

        # Add the classifier to the list
        knn_classifiers.append(knn_classifier)

        log_debug(f"KNN classifier for output variable {i} ({var_y}) created")

        log_debug(f"Training KNN classifier for output variable {i}...")

        # ======== Obtain results from test and train data ========

        test_preds, train_accuracy, test_accuracy = knn_classifier.classify()
        results[i] = (var_y_test, test_preds, train_accuracy, test_accuracy)

        log_debug(f"KNN classifier for output variable {i} ({var_y}) trained")

        log_trace(f"Output variable {i} results: {results[i]}")

        log_line(level="DEBUG")

        # ======== Variable importance ========

        log_info(f"Variable importance for output variable {i}")

        max_feature_name_length = max(
            [len(feature_name) for feature_name in X_labels])
        var_y_predictor_importance = ordered_predictor_indicies[i, :]
        log_info(f"  Predictors ordered by importance:",
                 var_y_predictor_importance)
        for feature_indx in var_y_predictor_importance:
            feature_name = X_labels[feature_indx]
            log_info(
                f"    Feature {feature_indx:<2} {feature_name:<{max_feature_name_length+3}}")

        # ======== Plot Decision Boundaries ========

        log_info(f"Plotting decision boundaries for output variable {i}...")

        knn_classifier.plot_multivar_decision_regions(
            var_y, test_preds, var_y_predictor_importance, y_var_unique_classes, 4)

    return results
