"""
Processing data helper
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from colorama import Fore, Style
import torch

import load_data
import numpy as np
import encode_data
from logger import *


SAMPLE_SIZE = 5


def process_classification_data(
    data_file_path: str,
    X_feature_names: List[str],
    y_feature_names: List[str],
    one_hot_encode: bool,
    normalise_data: bool,
    test_train_split_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[str]], List[int]]:
    """
    Preprocesses data ready for supervised learning by picking the X and y columns as passed,
    randomising the data, and encoding non-numeric data. Finally, the data is converted to
    numpy arrays, with dtype float32, then split into training and testing sets.

    This function assumes the data includes labels in the first row.

    Args:
    - data_file_path (str): The path to the data file.
    - X_feature_names (List[str]): The names of the features.
    - y_feature_names (List[str]): The names of the target variable.
    - one_hot_encode (bool): Whether to one-hot encode the target variable.
    - normalise_data (bool): Whether to normalise the data.
    - test_train_split_ratio (float): The ratio of the testing data.

    Returns:
    - X_train (ndarray): The training data features.
    - y_train (ndarray): The training data target variable.
    - X_test (ndarray): The testing data features.
    - y_test (ndarray): The testing data target variable.
    - unique_classes (List[np.ndarray]): The unique classe names in each target variable.
    - num_classes_in_vars (List[int]): The number of classes in each target variable.
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

    y_unique_classes = [list(y[col].unique()) for col in y.columns]
    log_debug(f"Unique classes in each target variable: {y_unique_classes}")

    # Normalise the data
    if normalise_data:
        log_title(f"Normalising X data...")
        X = encode_data.normalise_data(X)
        log_info(f"Data normalised")

    log_title(f"Encoding data...")

    X = encode_data.encode_non_numeric_data(X)
    y = encode_data.encode_non_numeric_data(y)

    # For any missing values, fill with the mean of the column
    X = X.fillna(X.mean())

    # For any values which are 0, replace with the mean of the column. Ensure this is done BEFORE one-hot encoding
    X = X.replace(0, X.mean())

    # Print all rows
    pd.set_option('display.max_rows', None)
    log_trace(f"Data all encoded: \n{X}")

    log_line()
    log_info(f"Data sample X:")
    print(X.head())

    log_info(f"Data sample y:")
    print(y.head())
    log_line()

    # Convert data to numpy arrays
    log_title(f"Data converted to numpy arrays")
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)
    log_info(f"Converted data to numpy arrays")

    # Calculate number of classes in each target variable
    num_output_vars = y.shape[1]
    unique_classes = [np.unique(y[:, col]) for col in range(num_output_vars)]
    num_classes_in_vars = [len(classes) for classes in unique_classes]

    if one_hot_encode:
        log_title(f"One-hot encoding target variables...")

        # One hot encode the data with padding
        y = encode_data.one_hot_encode_separate_output_vars(y)

        log_info(f"Data sample y:")
        log_debug(y[:SAMPLE_SIZE])
        log_line()

    log_info(f"Data encoded")

    log_title(f"Splitting data into training and testing data...")

    X_train, y_train, X_test, y_test = test_train_split(
        X, y, ratio=test_train_split_ratio)

    log_info(f"Data split into training and testing data")
    log_debug(
        f"X_train shape: {X_train.shape},\ny_train shape: {y_train.shape}\n")
    log_debug(f"X_test shape: {X_test.shape},\ny_test shape: {y_test.shape}")

    log_line()

    return X_train, y_train, X_test, y_test, y_unique_classes, num_classes_in_vars


def test_train_split(X: np.ndarray, y: np.ndarray, ratio: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training and testing data.

    Parameters:
    - X (ndarray): The input features.
    - y (ndarray): The target variable.
    - ratio (float): The ratio of the testing data.

    Returns:
    - X_train (ndarray): The training data features.
    - y_train (ndarray): The training data target variable.
    - X_test (ndarray): The testing data features.
    - y_test (ndarray): The testing data target variable.
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
        new_data = torch.as_tensor(data.data.reshape(
            (-1, dim_input)) / normalising_factor, dtype=torch.float32)
        labels = torch.as_tensor(data.targets)

        return_data.append(new_data)
        return_data.append(labels)

    return return_data
