"""
Driver file for week 6 practical class
28/03/2024
"""

import os
from q1 import q1
from q2 import q2
from q3 import q3
import sys
from colorama import Fore, Style
from load_data import load_data, tag_data


def preload_data(data_folder: str):
    train_datafile_name = "ann-train.data"
    test_datafile_name = "ann-test.data"

    train_datafile_path = os.path.join(data_folder, train_datafile_name)
    test_datafile_path = os.path.join(data_folder, test_datafile_name)

    # Load the data
    train_data = load_data(train_datafile_path)
    test_data = load_data(test_datafile_path)

    # Tag the data
    columns = [f"X{i}" for i in range(1, train_data.shape[1])] + ["Y"]
    print(f"{Fore.LIGHTMAGENTA_EX}Data Columns: {Style.RESET_ALL}{columns}\n")
    train_data = tag_data(train_data, columns)
    test_data = tag_data(test_data, columns)

    print(f"{Fore.GREEN}Train data:{Style.RESET_ALL}")
    print(train_data.head())

    print(f"{Fore.GREEN}Test data:{Style.RESET_ALL}")
    print(test_data.head())

    classes = ["Normal", "Hyperthyroid", "Hypothyroid"]

    return train_data, test_data, classes


def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    if len(sys.argv) > 1:
        if sys.argv[1] in ["1", "2"]:
            # Preload the data for q1, q2
            train_data, test_data, classes = preload_data(data_folder)

            if sys.argv[1] == "1":
                q1(train_data, test_data, classes)
            elif sys.argv[1] == "2":
                q2(train_data, test_data, classes)

        elif sys.argv[1] == "3":
            q3(data_folder)
        else:
            raise ValueError("Invalid question number")
    else:
        # Run all questions

        # Preload the data for q1, q2
        train_data, test_data, classes = preload_data(data_folder)

        # Question 1
        q1(train_data, test_data, classes)

        # Question 2
        q2(train_data, test_data, classes)

        # Question 3
        q3(data_folder)


if __name__ == "__main__":
    main()
