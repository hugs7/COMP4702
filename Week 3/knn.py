"""
Applies knn classifier to data

"""

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


def knn():
    return 0
