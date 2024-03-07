"""
Helper file to load data using pandas
"""

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    df = None

    # If Excel file
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    # If CSV file
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    return df


def tag_data(data: DataFrame, cols: list[str]) -> DataFrame:
    """
    Tags the data as first column X, second column Y and third column class
    """

    data.columns = cols

    return data


def shuffle_data(data: DataFrame) -> DataFrame:
    """
    Shuffles the data randomly.

    Parameters:
    - data: A pandas DataFrame containing the data to be shuffled.

    Returns:
    - A new pandas DataFrame with the data shuffled randomly.
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
    - y_train (DataFrame): The training data target variable.
    - X_test (DataFrame): The testing data features.
    - y_test (DataFrame): The testing data target variable.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, y_train, X_test, y_test
