"""
Driver file for week 6 practical class
28/03/2024
"""

from load_data import load_data
import os
from colorama import Fore, Style

def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    train_datafile_name = "ann-train.data"
    test_datafile_name = "ann-test.data"

    train_datafile_path = os.path.join(data_folder, train_datafile_name)
    test_datafile_path = os.path.join(data_folder, test_datafile_name)

    # Load the data
    train_data = load_data(train_datafile_path)
    test_data = load_data(test_datafile_path)

    print(f"{Fore.GREEN}Train data:{Style.RESET_ALL}")
    print(train_data.head())

    print(f"{Fore.GREEN}Test data:{Style.RESET_ALL}")
    print(test_data.head())

    exit(0)


if __name__ == "__main__":
    main()