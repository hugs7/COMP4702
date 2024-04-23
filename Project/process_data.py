"""
Processing data helper
"""

from sklearn.model_selection import train_test_split
from typing import List, Tuple
import load_data
import numpy as np
import encode_data
import torch


def process_classification_data(
    data_file_path: str,
    X_feature_names: List[str],
    y_feature_names: List[str],
    test_train_split_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    # Check if the file exists
    if not load_data.file_exists(data_file_path):
        raise FileNotFoundError(f"File {data_file_path} not found.")

    # Load the data
    data = load_data.load_data(data_file_path)

    all_feature_names = y_feature_names + X_feature_names

    data = load_data.tag_data(data, all_feature_names)

    # Get the values of the target variables
    y_classes = []
    for y_feature_name in y_feature_names:
        y_values = data.loc[:, y_feature_name].values

        # Get only the unique values

        y_values = list(np.unique(y_values))

        # Append the array to the y_classes array (2D array)
        y_classes.append(y_values)

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

    return data, X_train, y_train, X_test, y_test, y_classes


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


def preprocess_data(
    train_data: np.ndarray,
    validation_data: np.ndarray,
    dim_input: int,
    normalising_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    return_data = []

    for data in [train_data, validation_data]:
        new_data = torch.as_tensor(
            data.data.reshape((-1, dim_input)) / normalising_factor, dtype=torch.float32
        )
        labels = torch.as_tensor(data.targets)

        return_data.append(new_data)
        return_data.append(labels)

    return return_data
