"""
Classification Model Class
Hugo Burton
"""

from typing import List
import numpy as np
from sklearn.base import BaseEstimator

from logger import *

from model.base_model import Model


TRAINING = "training"
VALIDATION = "validation"
TESTING = "testing"

DATA_TYPES = {TRAINING, VALIDATION, TESTING}


def check_data_type(data_type: str) -> int:
    """
    Checks if the data type is valid

    Args:
    - data_type (str): The data type

    Returns:
    - int: Error code for invalid data type. 0 if valid.
    """

    if data_type not in DATA_TYPES:
        log_error(f"Invalid data type. Must be one of: {' '.join(DATA_TYPES)}")
        return 1

    return 0


def aggregate_fold_results(fold_results: list[tuple[np.ndarray, np.ndarray, float, float]]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Aggregates the results from the folds in a cross-validation run into a single set of results.

    Args:
    - fold_results (list[tuple[np.ndarray, np.ndarray, float, float]]): The results from each fold

    Returns:
    - tuple[np.ndarray, np.ndarray, float, float]: A tuple containing the aggregated test labels, test predictions, mean train accuracy, and mean test accuracy.
    """
    y_tests, test_preds, train_accuracies, test_accuracies = zip(*fold_results)
    aggregated_y_tests = np.concatenate(y_tests)
    aggregated_test_preds = np.concatenate(test_preds)
    mean_train_accuracy = np.mean(train_accuracies)
    mean_test_accuracy = np.mean(test_accuracies)
    return (aggregated_y_tests, aggregated_test_preds, mean_train_accuracy, mean_test_accuracy)


class Classifier(Model):
    def __init__(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_labels: list[str],
        y_labels: list[list[str]],
    ) -> None:
        super().__init__(X_train, y_train, X_test, y_test, X_labels, y_labels)

        self.model = model
