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
import utils
import file_helper
import encode_data

from nn.driver import run_nn_model, run_saved_nn_model
from knn.driver import run_knn_model
from dt.driver import run_dt_model


def dt_variable_ranking(
    dataset_name: str, dataset_file_path: str, X_vars: List[str], y_vars: List[str], test_train_ratio: float
) -> np.ndarray:
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
        dataset_name,
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

    models = {"knn": "k Nearest Neighbours",
              "dt": "Decision Tree", "nn": "Neural Network"}

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
    model_name = None
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
        raise ValueError(
            f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")

    dataset_file_name, columns = DATASET_MAPPING[dataset_name]
    columns: List[str]
    log_info(f"Dataset columns: {columns}")

    folder_of_script = os.path.dirname(__file__)
    data_folder = os.path.join(folder_of_script, "data")
    plots_folder = os.path.join(folder_of_script, "plots")

    # Create data folder if it does not exist
    file_helper.make_folder_if_not_exists(data_folder)
    file_helper.make_folder_if_not_exists(plots_folder)

    dataset_file_path = os.path.join(data_folder, dataset_file_name)

    # Check for third argument
    no_save_arg = ""
    if len(sys.argv) > 3:
        no_save_arg = sys.argv[3]

    save_plots = True if no_save_arg != "nosave" else False
    if not save_plots:
        # Set the plots folder to None
        plots_folder = None

    # === Correlation Matrix ===

    if first_arg == "corr":
        log_info("Plotting the correlation matrix of the data")
        data = load_data(dataset_file_path)
        data = encode_data.encode_non_numeric_data(data)
        title = f"Correlation matrix of predictor variables from {dataset_name} dataset"
        corr_plot_save_path = os.path.join(
            plots_folder, f"{dataset_name}_corr_matrix.png")
        file_helper.remove_file_if_exist(corr_plot_save_path)
        log_info("Columns in data: ", data.columns)
        correlation.plot_correlation_matrix(data, title, corr_plot_save_path)
        sys.exit(0)

    # === Column Labels ===

    # Specify the indices of the columns that are the variables we are predicting
    y_col_names = ["Species", "Population"]
    y_col_indices = utils.col_names_to_indices(columns, y_col_names)

    x_exclude_names = ["Year_start", "Year_end", "Latitude", "Longitude"]
    x_exclude_indices = utils.col_names_to_indices(columns, x_exclude_names)

    if model_name:
        if model_name == "nn":
            # Use all columns for neural network
            x_col_indices = [i for i in range(
                len(columns)) if i not in y_col_indices and i not in x_exclude_indices]
        else:
            x_col_names = ["Thorax_length", "Replicate", "Vial",
                           "Temperature", "Sex", "w1", "w2", "w3", "wing_loading"]
            x_col_indices = utils.col_names_to_indices(columns, x_col_names)

        # Obtain the vars of the x and y variables
        X_vars = [columns[i] for i in x_col_indices]
        y_vars = [columns[i] for i in y_col_indices]

        log_info(f"X vars: {X_vars}")
        log_info(f"y vars: {y_vars}")
        log_line(level="INFO")
    else:
        log_error("Model not found")
        sys.exit(1)

    # === Model ===

    log_info(f"Creating {model_name} model for {dataset_name} dataset...")

    # --- Dataset ---

    # Load and pre-process the dataset
    test_train_ratio = 0.3

    if model_name == "knn":
        predictors_ordered = dt_variable_ranking(
            dataset_name, dataset_file_path, X_vars, y_vars, test_train_ratio)

        X_train, y_train, X_test, y_test, unique_classes, num_classes_in_vars = process_data.process_classification_data(
            dataset_file_path, X_vars, y_vars, False, True, test_train_ratio
        )

        k_range = range(3, 100, 1)
        n_splits = 5

        run_knn_model(
            dataset_name,
            X_train,
            y_train,
            X_test,
            y_test,
            X_vars,
            y_vars,
            unique_classes,
            num_classes_in_vars,
            predictors_ordered,
            k_range=k_range,
            n_splits=n_splits,
            plots_folder_path=plots_folder,
        )
    elif model_name == "dt":
        X_train, y_train, X_test, y_test, unique_classes, num_classes_in_vars = process_data.process_classification_data(
            dataset_file_path,
            X_vars,
            y_vars,
            False,
            False,
            test_train_ratio,
        )

        max_tree_depth = 10

        predictors_ordered = run_dt_model(
            dataset_name,
            X_train,
            y_train,
            X_test,
            y_test,
            X_vars,
            y_vars,
            unique_classes,
            num_classes_in_vars,
            max_tree_depth=max_tree_depth,
            plots_folder_path=plots_folder,
        )
    elif model_name == "nn":
        third_arg = sys.argv[3] if len(sys.argv) > 3 else None
        checkpoint_num = sys.argv[4] if len(sys.argv) > 4 else None
        nn_folder_path = os.path.join(folder_of_script, "nn")

        # Load data
        X_train, y_train, X_test, y_test, unique_classes, num_classes_in_vars = process_data.process_classification_data(
            dataset_file_path, X_vars, y_vars, True, False, test_train_ratio
        )

        save_model = True
        if third_arg:
            if third_arg == "read":
                predictors_ordered = dt_variable_ranking(
                    dataset_name, dataset_file_path, X_vars, y_vars, test_train_ratio)

                # Read from saved final model
                log_info("Reading from saved final model")
                final_model = False if checkpoint_num else True
                model_str = "final" if final_model else f"checkpoint {checkpoint_num}"
                log_info(f"Using {model_str} model")

                run_saved_nn_model(
                    dataset_name,
                    X_test,
                    y_test,
                    X_vars,
                    y_vars,
                    unique_classes,
                    num_classes_in_vars,
                    predictors_ordered,
                    nn_folder_path,
                    final_model,
                    checkpoint_num=checkpoint_num,
                    plots_folder_path=plots_folder,
                )

                sys.exit(0)

            elif third_arg == "nosave":
                log_warning("Not saving model checkpoints or final")
                save_model = False

            else:
                log_error("Invalid third argument")
                sys.exit(1)

        # Grid search cv function (maybe)
        run_nn_model(
            dataset_name,
            X_train,
            y_train,
            X_test,
            y_test,
            X_vars,
            y_vars,
            unique_classes,
            num_classes_in_vars,
            nn_folder_path=nn_folder_path,
            plots_folder_path=plots_folder,
            is_save_model=save_model,
        )


if __name__ == "__main__":
    main()
