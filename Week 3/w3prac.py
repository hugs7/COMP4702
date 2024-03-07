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
    scatterplot.scatterplot_with_colour(data, "X1", "X2", "Y")

    # Randomise the data
    data_randomised = knn.shuffle_data(data)

    feature_names = ["X1", "X2"]
    # Split the data into training and testing data
    X = data_randomised.loc[:, feature_names]
    y = data_randomised.loc[:, "Y"]
    ratio = 0.3

    X_train, X_test, y_train, y_test = knn.test_train_split(X, y)

    # Apply the knn classifier

    knn_classifier, test_preds, train_accuracy, test_accuracy = knn.classify(
        X_train, X_test, y_train, y_test, feature_names, k=3
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
    print(f"{Fore.RED} Data Loaded {Fore.WHITE}")
    # Plot the data
    scatterplot.scatterplot_2d(data)

    feature_names = ["X"]

    X = data.loc[:, feature_names]
    y = data.loc[:, "Y"]

    X_train, X_test, y_train, y_test = knn.test_train_split(X, y)

    # Apply the knn regressor
    knn_regressor, test_preds, train_accuracy, test_accuracy = knn.regress(
        X_train, X_test, y_train, y_test, feature_names, k=3
    )

    print(f"{Fore.LIGHTMAGENTA_EX}Training R2: {Fore.WHITE}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing R2: {Fore.WHITE}", test_accuracy)

    # Plot the regression line
    knn.plot_regression_line(X_test, test_preds, knn_regressor)


def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    if False:
        classification(data_folder)

    regression(data_folder)


if __name__ == "__main__":
    main()
