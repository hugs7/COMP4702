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

    # Dictionary of output variable: (predictions, train_accuracy, test_accuracy)
    results = {}

    for i, var_y_labels in enumerate(y_labels):
        log_title(f"Output variable {i} classes: {var_y_labels}")

        # Get slice of y_train and y_test for this output variable
        var_y_train = y_train[:, i]
        var_y_test = y_test[:, i]

        log_trace(f"y_train_var:\n{var_y_train}")
        log_trace(f"y_test_var:\n{var_y_test}")

        log_debug(f"y_train_var shape: {var_y_train.shape}")
        log_debug(f"y_test_var shape: {var_y_test.shape}")
        log_line(level="DEBUG")

        knn_classifier = knn_model.KNNClassify(X_train, var_y_train, X_test, var_y_test, X_labels, var_y_labels, k=k)

        # Add the classifier to the list
        knn_classifiers.append(knn_classifier)

        log_debug(f"KNN classifier for output variable {i} created")

        log_info(f"Training KNN classifier for output variable {i}...")

        test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

        log_info(f"KNN classifier for output variable {i} trained")

        results[i] = (test_preds, train_accuracy, test_accuracy)

        log_debug(f"Output variable {i} results: {results[i]}")

        log_line(level="DEBUG")

    return results
