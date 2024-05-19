""""
Driver script for knn classification model
Hugo Burton
06/05/2024
"""

from typing import List
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from model.classifier import aggregate_fold_results
from plot.plot import plot_multivar_decision_regions

from knn import knn_model

from logger import *


def run_knn_model(
    dataset_name: str,
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
    n_splits: int = 5,
    plots_folder_path: str = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Driver script for k-nearest neighbours classification model. Takes in training, test data along with labels and trains
    a k-nearest neighbours model on the data.

    Args:
    - dataset_name (str): The name of the dataset.
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
    - n_splits (int): The number of splits to use in cross-validation.
    - plots_folder_path (str): The path to save the plots to.

    Returns:
    - dict[int, tuple[np.ndarray, np.ndarray, float, float]]: Dictionary of output variable: (y_test_true, test_predictions, train_accuracy, test_accuracy)
    """

    log_title("Start of knn model driver...")

    log_debug(
        f"Number of classes in each output variable: {num_classes_in_vars}")

    # For multi-variable classification, we need to create a knn classifier for each output variable
    # These are independent of each other.

    log_debug(
        f"Creating a knn classifier for each of the {len(y_labels)} output variables")

    # Dictionary of output variable: (y_test_true, test_predictions, train_accuracy, test_accuracy)
    results = {}

    for i, var_y in enumerate(y_labels):
        # ======== Train KNN classifier for this output variable ========
        log_title(f"Output variable {i}: {var_y}")
        y_var_unique_classes = unique_classes[i]
        log_info(
            f"Unique classes for output variable {i}: {y_var_unique_classes}")

        # Get slice of y_train and y_test for this output variable
        var_y_train = y_train[:, i]     # To be split up into
        var_y_test = y_test[:, i]

        log_trace(f"y_train_var:\n{var_y_train}")
        log_trace(f"y_test_var:\n{var_y_test}")

        log_debug(f"y_train_var shape: {var_y_train.shape}")
        log_debug(f"y_test_var shape: {var_y_test.shape}")

        log_debug(
            f"Defining k-fold cross-validation with {n_splits} splits...")
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=42)  # Because why not
        log_trace(f"KFold object: {kf}")

        fold_results = []

        for fold, (train_index, validation_index) in enumerate(kf.split(X_train, var_y_train)):
            log_info(f"Fold {fold+1}/{n_splits}...")

            X_train_fold, X_validation_fold = X_train[train_index], X_train[validation_index]
            y_train_fold, y_validation_fold = var_y_train[train_index], var_y_train[validation_index]

            knn_classifier = knn_model.KNNClassify(
                X_train_fold, y_train_fold, X_validation_fold, y_validation_fold, X_labels, y_var_unique_classes, k=k)

            log_trace(
                f"KNN classifier for output variable {i} ({var_y}) created")

            log_debug(f"Training KNN classifier for output variable {i}...")

            # ======== Obtain results from test and train data ========

            test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

            log_trace(
                f"KNN classifier for output variable {i} ({var_y}) trained")

            test_preds, train_accuracy, test_accuracy = knn_classifier.classify()
            fold_results.append(
                (y_test, test_preds, train_accuracy, test_accuracy))

            log_trace(
                f"Output variable {i}, fold {fold}, results: {fold_results[fold]}")

            log_line(level="DEBUG")

        log_debug(
            f"Cross validation training complete for output variable {i} ({var_y}). Aggregating results...")

        cv_results = aggregate_fold_results(fold_results)

        log_debug(
            f"Aggregated results: {cv_results}")

        results[i] = cv_results

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

        predict_callback = knn_classifier.model.predict
        delta = 4

        plot_multivar_decision_regions(
            var_y, test_preds, var_y_predictor_importance, y_var_unique_classes, knn_classifier.X_test,
            knn_classifier.X_labels, predict_callback, delta, "knn", dataset_name, plots_folder_path)

    return results
