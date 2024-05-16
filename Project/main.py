"""
Main Driver file for project
"""

import os
from typing import List
from colorama import Fore, Style
import sys
import numpy as np

from welcome import welcome, available_items
from dataset import DATASET_MAPPING
from load_data import load_data
import process_data
from logger import *
import correlation
from check_log_level import set_log_level

from nn.driver import run_nn_model
from knn.driver import run_knn_model
from dt.driver import run_dt_model


def dt_variable_ranking(dataset_file_path, X_vars, y_vars, test_train_ratio) -> np.ndarray:
    """
    Fits a decision tree model to the data and returns the ranking of the variables by importance.

    Parameters:
    - dataset_file_path (str): The path to the dataset file.
    - X_vars (List[str]): The names of the predictor variables.
    - y_vars (List[str]): The names of the target variables.
    - test_train_ratio (float): The ratio of test to train data.

    Returns:
    - ndarray: The ranking of the variables by importance as a 2D array. Rows are output
               variables and columns are the indices of the predictors in descending order of importance.
    """

    X_train, y_train, X_test, y_test, unique_classes, num_classes_in_vars = process_data.process_classification_data(
        dataset_file_path, X_vars, y_vars, False, False, test_train_ratio
    )

    max_tree_depth = 6

    predictors_ordered = run_dt_model(
        X_train,
        y_train,
        X_test,
        y_test,
        X_vars,
        y_vars,
        unique_classes,
        num_classes_in_vars,
        max_tree_depth=max_tree_depth,
        variable_importance_only=True,
    )

    return predictors_ordered


def main():
    set_log_level()
    welcome()

    models = {"knn": "k Nearest Neighbours", "dt": "Decision Tree", "rf": "Random Forest", "nn": "Neural Network"}

    if len(sys.argv) < 3:
        if len(sys.argv) == 2:
            arg = sys.argv[1]
            if arg == "--help":
                available_items("models", models)
                available_items("datasets", DATASET_MAPPING.keys())
                sys.exit(0)

        print(
            f"{Fore.LIGHTRED_EX}Usage: {Fore.LIGHTCYAN_EX}python main.py {Fore.LIGHTMAGENTA_EX}<model_name> <dataset_name>{Style.RESET_ALL}\n"
        )
        available_items("models", models)

        available_items("datasets", DATASET_MAPPING.keys())

        sys.exit(1)

    # Check model
    first_arg = sys.argv[1].lower()
    if first_arg == "corr":
        pass
    else:
        model_name = first_arg
        if model_name not in models.keys():
            log_error(f"Model {model_name} not found")
            available_items("models", models)
            sys.exit(1)

    # Check dataset
    dataset_name = sys.argv[2]
    if dataset_name not in DATASET_MAPPING:
        log_error(f"Dataset {dataset_name} not found")
        available_items("datasets", DATASET_MAPPING.keys())
        sys.exit(1)

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")

    dataset_file_name, columns = DATASET_MAPPING[dataset_name]
    columns: List[str]

    folder_of_script = os.path.dirname(__file__)
    data_folder = os.path.join(folder_of_script, "data")

    # Create data folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    dataset_file_path = os.path.join(data_folder, dataset_file_name)

    # === Column Labels ===

    # Specify the indices of the columns that are the variables we are predicting
    y_col_indices = [9]
    x_col_names = ["Thorax_length", "wing_loading"]
    x_col_indices = [columns.index(col) for col in x_col_names]
    exclude_col_indices = [2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17]

    # Derive the indices of the x variables by removing the y indices
    x_col_indices = [i for i in range(len(columns)) if i not in y_col_indices and i not in exclude_col_indices]

    # Obtain the vars of the x and y variables
    X_vars = [columns[i] for i in x_col_indices]
    y_vars = [columns[i] for i in y_col_indices]

    log_info(f"X vars: {X_vars}")
    log_info(f"y vars: {y_vars}")
    log_line(level="INFO")

    # === Correlation Matrix ===

    if first_arg == "corr":
        log_info("Plotting the correlation matrix of the data")
        data = load_data(dataset_file_path)
        X = data[X_vars]
        title = f"Correlation matrix of predictor variables from {dataset_name} dataset"
        correlation.plot_correlation_matrix(X, title)
        sys.exit(0)

    # === Model ===

    log_info(f"Creating {model_name} model for {dataset_name} dataset...")

    # --- Dataset ---

    # Load and pre-process the dataset
    test_train_ratio = 0.3

    if model_name == "knn":
        predictors_ordered = dt_variable_ranking(dataset_file_path, X_vars, y_vars, test_train_ratio)

        X_train, y_train, X_test, y_test, unique_classes, num_classes_in_vars = process_data.process_classification_data(
            dataset_file_path, X_vars, y_vars, False, True, test_train_ratio
        )

        k = 30

        run_knn_model(X_train, y_train, X_test, y_test, X_vars, y_vars, unique_classes, num_classes_in_vars, predictors_ordered, k=k)
    elif model_name == "dt":
        X_train, y_train, X_test, y_test, unique_classes, num_classes_in_vars = process_data.process_classification_data(
            dataset_file_path, X_vars, y_vars, False, False, test_train_ratio
        )

        max_tree_depth = 6

        predictors_ordered = run_dt_model(
            X_train, y_train, X_test, y_test, X_vars, y_vars, unique_classes, num_classes_in_vars, max_tree_depth=max_tree_depth
        )
    elif model_name == "rf":
        raise NotImplementedError("Random forest not implemented")
    elif model_name == "nn":
        # Grid search cv function

        X_train, y_train, X_test, y_test, unique_classes, num_classes_in_vars = process_data.process_classification_data(
            dataset_file_path, X_vars, y_vars, True, False, test_train_ratio
        )
        run_nn_model(X_train, y_train, X_test, y_test, X_vars, y_vars, unique_classes, num_classes_in_vars)


if __name__ == "__main__":
    main()
