""""
Driver script for knn classification model
Hugo Burton
06/05/2024
"""

import sys
from typing import List
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from plot.plot import plot_knn_accuracies, plot_multivar_decision_regions

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
    k_range: range,
    n_splits: int = 5,
    plots_folder_path: str = None,
) -> None:
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
    - k_range (range): The range of k to perform cross validation over.
    - n_splits (int): The number of splits to divide up the training set into to perform cross validation. Defaults to 5. The test
                      set is reserved for the final classifier.
    - plots_folder_path (str): The path to save the plots to.
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

        # Cross Validation Setup
        log_info(
            f"Performing Cross validation with n_splits: {n_splits}, k_range: {k_range}...")
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

        cv_knn_classifier = knn_model.KNNClassify(
            X_train, var_y_train, X_test, var_y_test, X_labels, y_var_unique_classes, k=5)
        cv_knn_classifier_model = cv_knn_classifier.model

        param_grid = {"n_neighbors": k_range}
        grid_search = GridSearchCV(
            cv_knn_classifier_model, param_grid, cv=kf, scoring="accuracy")

        grid_search.fit(X_train, var_y_train)

        best_k = grid_search.best_params_["n_neighbors"]
        cv_results = grid_search.cv_results_
        log_info(f"Best k for output variable {i}: {best_k}")

        # Assuming you have a grid search object named cv_results
        accuracies_by_k = {}

        # Iterate over each parameter setting in the grid search
        for params, accuracy in zip(cv_results["params"], cv_results["mean_test_score"]):
            # Extract the value of k from the parameters
            log_trace(f"Params: {params}, accuracy: {accuracy}")

            k_value = params["n_neighbors"]

            # Check if the k_value already exists in the accuracies dictionary
            if k_value in accuracies_by_k:
                log_warning(
                    f"Duplicate accuracy for k value found: {k_value}. Continuing...")
                continue

            # Append the accuracy to the list corresponding to the k_value
            accuracies_by_k[k_value] = accuracy

        log_debug(f"Accuracies: {accuracies_by_k}")

        # Plot the accuracies for each k as a line plot
        log_debug(
            f"Plotting accuracies for each k as a line plot for output variable {i}...")
        plot_knn_accuracies(accuracies_by_k, dataset_name,
                            var_y, plots_folder_path)

        # Final Classifier trained on best_k
        log_info(f"Fitting Final Classifier trained on best k: {best_k}")
        cv_knn_classifier = knn_model.KNNClassify(
            X_train, var_y_train, X_test, var_y_test, X_labels, y_var_unique_classes, k=best_k)

        # Add the classifier to the list
        knn_classifiers.append(cv_knn_classifier)

        log_debug(f"KNN classifier for output variable {i} ({var_y}) created")

        log_debug(f"Training KNN classifier for output variable {i}...")

        # ======== Obtain results from test and train data ========

        test_preds, train_accuracy, test_accuracy = cv_knn_classifier.classify()
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

        predict_callback = cv_knn_classifier.model.predict
        delta = 4

        plot_multivar_decision_regions(
            var_y,
            test_preds,
            var_y_predictor_importance,
            y_var_unique_classes,
            cv_knn_classifier.X_test,
            cv_knn_classifier.X_labels,
            predict_callback,
            delta,
            "knn",
            dataset_name,
            plots_folder_path,
        )

    return results
