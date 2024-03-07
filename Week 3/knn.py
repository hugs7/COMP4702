"""
Applies knn classifier to data

"""

from colorama import Fore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pandas import DataFrame


def shuffle_data(data: DataFrame) -> DataFrame:
    """
    Shuffles the data randomly
    """

    return data.sample(frac=1).reset_index(drop=True)


def test_train_split(
    X: DataFrame, y: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits the data into training and testing data.

    Parameters:
    - X (DataFrame): The input features.
    - y (DataFrame): The target variable.

    Returns:
    - X_train (DataFrame): The training data features.
    - X_test (DataFrame): The testing data features.
    - y_train (DataFrame): The training data target variable.
    - y_test (DataFrame): The testing data target variable.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def knn(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
    k: int = 3,
) -> tuple[np.ndarray, float, float]:
    """
    Performs k-nearest neighbors classification on the given training and test data.

    Parameters:
        X_train (DataFrame): The feature matrix of the training data.
        y_train (DataFrame): The target values of the training data.
        X_test (DataFrame): The feature matrix of the test data.
        y_test (DataFrame): The target values of the test data.
        k (int, optional): The number of neighbors to consider. Defaults to 3.

    Returns:
        tuple[np.ndarray, float, float]: A tuple containing the test predictions, train accuracy, and test accuracy.
    """

    model = KNeighborsClassifier(n_neighbors=k)

    print(
        f"{Fore.RED}X_train dim: {X_train.shape}, y_train dim: {y_train.shape}{Fore.WHITE}"
    )
    model.fit(X_train, y_train)

    # Get results
    train_accuracy = model.score(X_train, y_train)

    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return test_predictions, train_accuracy, test_accuracy
