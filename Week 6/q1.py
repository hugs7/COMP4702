"""
Implements question 1 from Week 6 Prac
"""

from load_data import split_feature_response
from colorama import Fore, Style
from pandas import DataFrame
from knn_helper import knn_classify
from conf_matrix import conf_matrix


def q1(train_data: DataFrame, test_data: DataFrame, classes: list[str]):
    """
    Train a k-NN model (choose some reasonable value for k) on the training set and calculate a 
    confusion matrix for the hold-out validation set.
    """

    # train_data will be used as training set.
    # test_data will be used as final test set (called hold-out in the question).

    # Extract the feature names
    feature_names = train_data.columns[:-1]
    X_train, y_train = split_feature_response(train_data)
    X_test, y_test = split_feature_response(test_data)

    # Apply the knn classifier to the training data
    test_predictions, train_accuracy, test_accuracy = knn_classify(
        X_train, y_train, X_test, y_test, feature_names, k=5
    )

    print(f"{Fore.LIGHTGREEN_EX}Train Accuracy: {Style.RESET_ALL}{train_accuracy}")
    print(f"{Fore.LIGHTGREEN_EX}Test Accuracy: {Style.RESET_ALL}{test_accuracy}")

    # Calculate the confusion matrix
    print(f"{Fore.LIGHTGREEN_EX}Confusion Matrix: {Style.RESET_ALL}")

    conf_matrix(y_test, test_predictions, classes)

    return conf_matrix
