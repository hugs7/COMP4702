"""
Main Driver file for project
"""

import os
from colorama import Fore, Style
import sys

from welcome import welcome, available_items
from dataset import DATASET_MAPPING
import process_data
from logger import *

from nn.driver import run_nn_model
from knn.driver import run_knn_model


def main():
    welcome()

    dataset_name = "Thorax"

    models = {"knn": "k Nearest Neighbours", "dt": "Decision Tree",
              "rf": "Random Forest", "nn": "Neural Network"}

    if len(sys.argv) < 3:
        print(
            f"{Fore.LIGHTRED_EX}Usage: {Fore.LIGHTCYAN_EX}python main.py {Fore.LIGHTMAGENTA_EX}<model_name> <dataset_name>{Style.RESET_ALL}\n"
        )
        available_items("models", models)

        available_items("datasets", DATASET_MAPPING.keys())

        sys.exit(1)

    # Check dataset
    dataset_name = sys.argv[2]
    if dataset_name not in DATASET_MAPPING:
        log_error(f"Dataset {dataset_name} not found")
        available_items("datasets", DATASET_MAPPING.keys())
        sys.exit(1)

    # Check model
    model_name = sys.argv[1].lower()
    if model_name not in models.keys():
        log_error(f"Model {model_name} not found")
        available_items("models", models)
        sys.exit(1)

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(
            f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")

    dataset_file_name, columns = DATASET_MAPPING[dataset_name]

    folder_of_script = os.path.dirname(__file__)
    data_folder = os.path.join(folder_of_script, "data")

    # Create data folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    dataset_file_path = os.path.join(data_folder, dataset_file_name)

    log_info(f"Creating {model_name} model for {dataset_name} dataset...")

    # Specify the indices of the columns that are the variables we are predicting
    y_col_indices = [0, 1]

    # Derive the indices of the x variables by removing the y indices
    x_col_indices = [i for i in range(len(columns)) if i not in y_col_indices]

    # Obtain the labels of the x and y variables

    X_labels = [columns[i] for i in x_col_indices]
    y_labels = [columns[i] for i in y_col_indices]

    log_info(f"X labels: {X_labels}")
    log_info(f"y labels: {y_labels}")

    # --- Dataset ---

    # Load and pre-process the dataset
    test_train_ratio = 0.3
    X_train, y_train, X_test, y_test = process_data.process_classification_data(dataset_file_path, X_labels, y_labels, test_train_ratio)

    if model_name == "knn":
        run_knn_model(dataset_file_path, columns)
        raise NotImplementedError("KNN not implemented")
    elif model_name == "dt":
        raise NotImplementedError("Decision tree not implemented")
    elif model_name == "rf":
        raise NotImplementedError("Random forest not implemented")
    elif model_name == "nn":
        run_nn_model(X_train, y_train, X_test, y_test, X_labels, y_labels)


if __name__ == "__main__":
    main()
