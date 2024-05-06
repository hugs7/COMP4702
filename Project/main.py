"""
Main Driver file for project
"""

import os
from colorama import Fore, Style
import sys

from welcome import welcome, available_items
from dataset import DATASET_MAPPING

from nn.driver import run_nn_model



def main():
    welcome()

    dataset_name = "Thorax"

    models = {"knn": "k Nearest Neighbours", "dt": "Decision Tree", "nn": "Neural Network"}

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
        print(f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")
        available_items("datasets", DATASET_MAPPING.keys())
        sys.exit(1)

    # Check model
    model_name = sys.argv[1].lower()
    if model_name not in models.keys():
        print(f"{Fore.RED}Model {model_name} not found{Style.RESET_ALL}")
        available_items("models", models)
        sys.exit(1)

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")

    dataset_file_name, columns = DATASET_MAPPING[dataset_name]

    folder_of_script = os.path.dirname(__file__)
    data_folder = os.path.join(folder_of_script, "data")

    # Create data folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    dataset_file_path = os.path.join(data_folder, dataset_file_name)

    print(
        f"{Fore.GREEN}Creating {model_name} model for {dataset_name} dataset{Style.RESET_ALL}"
    )

    if model_name == "knn":
        raise NotImplementedError("KNN not implemented")
    elif model_name == "dt":
        raise NotImplementedError("Decision tree not implemented")
    elif model_name == "nn":
        run_nn_model(dataset_file_path, columns)


if __name__ == "__main__":
    main()
