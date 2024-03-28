"""
Driver file for week 6 practical class
28/03/2024
"""

from load_data import load_data, load_and_process_data, split_feature_response, tag_data, test_train_split
import os
from colorama import Fore, Style
from pandas import DataFrame
from knn_helper import knn_classify
from logistic_helper import logistic_fit
from linear_helper import linear_fit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def conf_matrix(y_true, y_pred, classes):
    """
    Calculate the confusion matrix for the given data and
    display it using ConfusionMatrixDisplay
    """

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    conf_matrix_disp = ConfusionMatrixDisplay(
        conf_matrix, display_labels=classes)
    conf_matrix_disp.plot(cmap="Blues")
    plt.show()

    return conf_matrix


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


def q3(data_folder: str):
    """
    In Prac W4 we applied linear regression to a pokemon dataset, where the loss function was sum 
    of squares (or mean squared) error. Revisit this task but add (a) L2; (b) L1 regularisation to
    the loss function, with some suitable value for the regularization hyperparameter (see Section 
    5.3 of the textbook). Compare the coefficient values from your different trained models. 
    """

    pokemon_data = "pokemonregr.csv"
    pokemon_data_path = os.path.join(data_folder, pokemon_data)

    # Load the data
    pokemon_data = load_and_process_data(
        pokemon_data_path, replace_null=True, header="infer")

    X_data, y_data = split_feature_response(pokemon_data)
    feature_names = X_data.columns
    print(f"{Fore.LIGHTMAGENTA_EX}Feature names: {Style.RESET_ALL}{feature_names}")

    X_train, y_train, X_test, y_test = test_train_split(X_data, y_data)

    penalties = ["l1", "l2", "elasticnet"]

    for penalty in penalties:
        print(f"{'-'*100}\n{Fore.LIGHTRED_EX}Penalty: {Style.RESET_ALL}{penalty}")

        if penalty == "elasticnet":
            linear_fit(X_train, y_train, X_test, y_test,
                       feature_names, penalty=penalty, l1_ratio=0.5)
        else:
            linear_fit(X_train, y_train, X_test, y_test,
                       feature_names, penalty=penalty)


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

    # Question 1
    # q1(train_data, test_data, classes)

    # Question 2
    # q2(train_data, test_data, classes)

    # Question 3
    q3(data_folder)

    exit(0)


if __name__ == "__main__":
    main()
