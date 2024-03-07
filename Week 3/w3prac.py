"""
Week 3 Practical
Hugo Burton
07/03/2024
"""

import os

import load_data
import scatterplot
import knn
from colorama import Fore


def classification(data_folder):
    """
    Classification task
    """

    # Load the data
    file_path = os.path.join(data_folder, "w3classif.csv")

    data = load_data.load_data(file_path)

    data = load_data.tag_data(data, ["X1", "X2", "Y"])

    # Show the data as scatterplot
    scatterplot.scatterplot(data)

    # Randomise the data
    data_randomised = knn.shuffle_data(data)

    # Split the data into training and testing data
    X = data_randomised.loc[:, ["X1", "X2"]]
    y = data_randomised.loc[:, "Y"]
    ratio = 0.3

    X_train, X_test, y_train, y_test = knn.test_train_split(X, y)

    # Apply the knn classifier

    knn_classifier, test_preds, train_accuracy, test_accuracy = knn.knn(
        X_train, X_test, y_train, y_test, k=3
    )

    print(f"{Fore.LIGHTMAGENTA_EX}Training accuracy: {Fore.BLACK}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing accuracy: {Fore.BLACK}", test_accuracy)
    print("")
    print(f"{Fore.LIGHTMAGENTA_EX}Training MCR: {Fore.BLACK}", 1 - train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing MCR: {Fore.BLACK}", 1 - test_accuracy)

    # Plot decision regions

    knn.plot_decision_regions(X_test, test_preds, knn_classifier, resolution=0.02)


def regression(data_folder):
    """
    Regression task
    """

    file_path = os.path.join(data_folder, "w3regr.csv")

    data = load_data.load_data(file_path)

    data = load_data.tag_data(data, ["X", "Y"])


def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    # classification(data_folder)

    regression(data_folder)


if __name__ == "__main__":
    main()
