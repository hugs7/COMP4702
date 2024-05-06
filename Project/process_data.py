"""
Processing data helper
"""

from sklearn.model_selection import train_test_split
from typing import List, Tuple
from colorama import Fore, Style
import torch

import load_data
import numpy as np
import encode_data
from logger import *


def process_classification_data(
    data_file_path: str,
    X_feature_names: List[str],
    y_feature_names: List[str],
    test_train_split_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses data ready for supervised learning by picking the X and y columns as passed,
    randomising the data, and encoding non-numeric data. Finally, the data is converted to
    numpy arrays, with dtype float32, then split into training and testing sets.

    This function assumes the data includes labels in the first row.

    Args:
    - data_file_path (str): The path to the data file.
    - X_feature_names (List[str]): The names of the features.
    - y_feature_names (List[str]): The names of the target variable.
    - test_train_split_ratio (float): The ratio of the testing data.

    Returns:
    - X_train (np.ndarray): The training data features.
    - y_train (np.ndarray): The training data target variable.
    - X_test (np.ndarray): The testing data features.
    - y_test (np.ndarray): The testing data target variable.
    """

    log_title(f"Pre-processing data...")

    # Check if the file exists
    if not load_data.file_exists(data_file_path):
        raise FileNotFoundError(f"File {data_file_path} not found.")

    # Load the data
    data = load_data.load_data(data_file_path)
    log_debug(f"Data sample\n{data.head()}")

    # Randomise the data
    data_randomised = load_data.shuffle_data(data)

    # Split the data into training and testing data
    X = data_randomised[X_feature_names]
    y = data_randomised[y_feature_names]

    log_info(f"Data pre-processed")

    log_title(f"Encoding data...")

    X = encode_data.encode_non_numeric_data(X)
    y = encode_data.encode_non_numeric_data(y)

    log_line()
    log_info(f"Data sample X:")
    print(X.head())

    log_info(f"Data sample y:")
    print(y.head())
    log_line()

    log_info(f"Data encoded")

    log_title(f"Converting data to numpy arrays...")
    # Convert data to numpy arrays
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)

    log_info(f"Data converted to numpy arrays")

    log_title(f"Splitting data into training and testing data...")

    X_train, y_train, X_test, y_test = test_train_split(X, y, ratio=test_train_split_ratio)

    log_info(f"Data split into training and testing data")
    log_debug(f"X_train shape: {X_train.shape},\ny_train shape: {y_train.shape}")

    return X_train, y_train, X_test, y_test


def test_train_split(X: np.ndarray, y: np.ndarray, ratio: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        new_data = torch.as_tensor(data.data.reshape((-1, dim_input)) / normalising_factor, dtype=torch.float32)
        labels = torch.as_tensor(data.targets)

        return_data.append(new_data)
        return_data.append(labels)

    return return_data
