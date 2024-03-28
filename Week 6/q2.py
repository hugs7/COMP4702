"""
Implements question 2 from Week 6 Prac
"""

from load_data import split_feature_response
from colorama import Fore, Style
from pandas import DataFrame
from logistic_helper import logistic_fit
from conf_matrix import conf_matrix


def q2(train_data: DataFrame, test_data: DataFrame, classes: list[str]):
    """
    Attempt to reproduce Example 4.5 from the Lindholm et al. textbook. You will need to: 
        - Convert the data into a binary classification problem.
        - Train a logistic regression model on the training data.
        - Evaluate the trained model to calculate a confusion matrix.
        - Vary the decision threshold for the model as done in Example 4.5 and 
          recalculate the confusion matrix. 
    """

    # Now the problem is binary classification

    # Extract the feature names
    feature_names = train_data.columns[:-1]
    X_train, y_train = split_feature_response(train_data)
    X_test, y_test = split_feature_response(test_data)

    # Classes "Hyperthyroid" and "Hypothyroid" will be combined into a single class called "Abnormal"
    def amalgamate_abnormal_classes(y: DataFrame):
        return y.apply(lambda r: "Abnormal" if r != 1 else classes[1-1])

    y_train = amalgamate_abnormal_classes(y_train)
    y_test = amalgamate_abnormal_classes(y_test)

    new_classes = [classes[1-1], "Abnormal"]

    print(f"{Fore.LIGHTGREEN_EX}Train Data levels:{Style.RESET_ALL}")
    print(y_train.value_counts())
    print(f"{Fore.LIGHTGREEN_EX}Test data after amalgamation:{Style.RESET_ALL}")
    print(y_test.value_counts())

    # Apply the logistic regression model to the training data
    test_predictions, train_accuracy, test_accuracy = logistic_fit(
        X_train, y_train, X_test, y_test, feature_names, new_classes, threshold=0.01
    )

    print(test_predictions)
    print(f"{Fore.LIGHTGREEN_EX}Train Accuracy: {Style.RESET_ALL}{train_accuracy}")
    print(f"{Fore.LIGHTGREEN_EX}Test Accuracy: {Style.RESET_ALL}{test_accuracy}")

    conf_matrix(y_test, test_predictions, new_classes)
