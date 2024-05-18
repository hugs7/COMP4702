"""
Driver for decision tree model
Hugo Burton
"""

from typing import List
import numpy as np

from plot.plot import plot_multivar_decision_regions

from dt import decision_tree

from logger import *


def run_dt_model(
    dataset_name: str,
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
    plots_folder_path: str = None,
) -> np.ndarray:
    """
    Question 1
    Fit a single Decision Tree classifier to the training data
    with default parameters and calculate E_{train} and
    E_{hold-out}

    Parameters:
    - dataset_name (str): The name of the dataset.
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
    - plots_folder_path (str): The path to save the plots to.

    Returns:
    - (nparray) Ranking of variables by importance in a 2D array with each row in descending ordered of indices of the
                predictors for that row's variable.
    """

    num_input_vars = len(X_labels)
    num_output_vars = len(y_labels)
    predictors_ordered = np.zeros((num_output_vars, num_input_vars), dtype=int)

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

        decision_tree_model = decision_tree.DTClassifier(
            X_train, var_y_train, X_test, var_y_test, X_labels, y_var_unique_classes, max_tree_depth=max_tree_depth
        )

        log_debug(
            f"Decision tree model for output variable {i} ({var_y}) created")

        log_debug(
            f"Training decision tree classifier for output variable {i}...")

        # ======== Obtain results from test and train data ========

        test_preds, train_accuracy, test_accuracy = decision_tree_model.classify()
        results[i] = (var_y_test, test_preds, train_accuracy, test_accuracy)

        log_debug(
            f"Decision tree classifier for output variable {i} ({var_y}) trained")

        log_trace(f"Output variable {i} results: {results[i]}")

        log_line(level="DEBUG")

        # Variable importance
        log_title("Fetching variable importance...")
        variable_importance = decision_tree_model.model.feature_importances_
        log_debug(f"Variable importance: {variable_importance}")

        # Convert variable importance to a list of indices in descending order
        predictors_ordered_var = np.argsort(variable_importance)[::-1]
        log_debug(f"Predictors ordered by importance: {predictors_ordered}")

        predictors_ordered_names = [X_labels[i]
                                    for i in predictors_ordered_var]
        log_info(
            f"Predictors ordered by importance (names): {predictors_ordered_names}")

        if not variable_importance_only:
            log_title("Plotting decision regions...")

            predict_callback = decision_tree_model.model.predict
            delta = 4

            plot_multivar_decision_regions(
                var_y, test_preds, predictors_ordered_var, y_var_unique_classes,
                decision_tree_model.X_test, decision_tree_model.X_labels, predict_callback, delta,
                model_name="dt", dataset_name=dataset_name, plots_folder_path=plots_folder_path)

        predictors_ordered[i] = predictors_ordered_var

    return predictors_ordered
