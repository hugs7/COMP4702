"""
Driver for decision tree model
Hugo Burton
"""

from typing import List
import numpy as np

from dt import decision_tree
from logger import *
from utils import accuracy_to_error


def run_dt_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_labels: List[str],
    y_labels: List[List[str]],
    unique_classes: List[List[str]],
    num_classes_in_vars: List[int],
    max_tree_depth: int = 5,
    variable_importance_only: bool = False,
) -> np.ndarray:
    """
    Question 1
    Fit a single Decision Tree classifier to the training data
    with default parameters and calculate E_{train} and
    E_{hold-out}

    Parameters:
    - X_train (ndarray): The training data features.
    - y_train (ndarray): The training data target variable.
    - X_test (ndarray): The testing data features.
    - y_test (ndarray): The testing data target variable.
    - X_labels (List[str]): The names of the features.
    - y_labels (List[List[str]]): The names of each class within each target variable. Of which there can be multiple
    - unique_classes (List[List[str]]): The unique classes in each target variable.
    - num_classes_in_vars (List[int]): The number of classes in each target variable.
    - max_tree_depth (int): The maximum depth of the decision tree.
    - variable_importance_only (bool): Flag for whether we just want to compute variable importance (and not show plots).

    Returns:
    - Ranking of variables by importance. Format is a descending ordered list of the indices of the predictors.
    """

    decision_tree_model = decision_tree.DTClassifier(X_train, y_train, X_test, y_test, X_labels, y_labels, max_tree_depth=max_tree_depth)

    test_preds, train_accuracy, test_accuracy = decision_tree_model.classify()

    log_debug(f"Test predictions: {test_preds}")

    log_info(f"Train accuracy: {train_accuracy}")
    log_info(f"Test accuracy: {test_accuracy}")

    log_line(level="INFO")

    # Variable importance
    log_title("Variable importance:")
    variable_importance = decision_tree_model.model.feature_importances_
    log_info(f"Variable importance: {variable_importance}")

    # Convert variable importance to a list of indices in descending order
    predictors_ordered = np.argsort(variable_importance)[::-1]
    log_info(f"Predictors ordered by importance: {predictors_ordered}")

    predictors_ordered_names = [X_labels[i] for i in predictors_ordered]
    log_info(f"Predictors ordered by importance (names): {predictors_ordered_names}")

    if not variable_importance_only:
        log_title("Plotting decision regions...")

        exit()

        # decision_tree_model.plot_decision_regions(test_preds, , resolution=0.02)

    return predictors_ordered
