"""
Processing data helper
"""

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Union, List
import load_data
import os
import numpy as np
import encode_data


def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Parameters:
    - file_path (str): The file path.

    Returns:
    - bool: True if the file exists, False otherwise.
    """

    return os.path.exists(file_path)


def process_classification_data(
    data_file_path: str,
    X_feature_names: List[str],
    y_feature_names: List[str],
    test_train_split_ratio: float = 0.3,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Check if the file exists
    if not file_exists(data_file_path):
        raise FileNotFoundError(f"File {data_file_path} not found.")

    # Load the data
    data = load_data.load_data(data_file_path)

    all_feature_names = y_feature_names + X_feature_names

    data = load_data.tag_data(data, all_feature_names)

    # Randomise the data
    data_randomised = load_data.shuffle_data(data)

    # Split the data into training and testing data
    X = data_randomised.loc[:, X_feature_names]
    y = data_randomised.loc[:, y_feature_names]

    # X data is mostly numeric but in the case it isn't, encode it
    X = encode_data.encode_non_numeric_data(X)

    # y data is mostly categorical but in the case it isn't, encode it
    y = encode_data.encode_non_numeric_data(y)

    print("X", X)

    print("-" * 50)

    print("y", y)

    # Convert data to numpy arrays
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)

    X_train, y_train, X_test, y_test = test_train_split(
        X, y, ratio=test_train_split_ratio
    )

    return data, X_train, y_train, X_test, y_test


def test_train_split(
    X: np.ndarray, y: np.ndarray, ratio: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training and testing data.

    Parameters:
    - X (DataFrame): The input features.
    - y (DataFrame): The target variable.
    - ratio (float): The ratio of the testing data.

    Returns:
    - X_train (DataFrame): The training data features.
    - y_train (DataFrame): The training data target variable.
    - X_test (DataFrame): The testing data features.
    - y_test (DataFrame): The testing data target variable.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)

    return X_train, y_train, X_test, y_test
