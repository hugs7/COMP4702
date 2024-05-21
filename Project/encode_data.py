"""
Helper script to encode non-numeric data as numeric
"""

import numpy as np
import pandas as pd

from logger import *


def encode_non_numeric_data(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the data
    data = data.copy()

    for column in data.columns:
        sample_data_point = data[column].iloc[0]
        log_trace(
            f"Column: {column}, Sample Data Point: {sample_data_point}, Data Type: {data[column].dtype}")
        if data[column].dtype == str or data[column].dtype == object:
            # Replace with mode
            data[column] = data[column].astype("category")
            data[column] = data[column].cat.codes

    return data


def normalise_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises the data by subtracting the mean and dividing by the standard deviation for each column.

    Parameters:
    - data (DataFrame): The data to be normalised.

    Returns:
    - normalised_data (DataFrame): The normalised data.
    """

    # Some helper functions
    def normalise_column(column: pd.Series) -> pd.Series:
        column_mean = column.mean()
        column_std = column.std()
        log_trace(f"Column Mean: {column_mean}, Column Std: {column_std}")

        normalised_column = (column - column_mean) / column_std
        log_trace(f"Normalised Column Sample: \n{normalised_column.head()}")

        return normalised_column

    # Create a copy of the data
    normalised_data = data.copy()

    # Iterate over each column in the data
    for column in normalised_data.columns:
        # Check if the column contains numerical data
        column_data = normalised_data[column]

        if np.issubdtype(column_data.dtype, np.number):
            log_debug(
                f"Column: {column} is of type {column_data.dtype}. Normalising...")

            # Normalise the current column
            normalised_data[column] = normalise_column(column_data)

        elif np.issubdtype(column_data.dtype, np.object_):
            log_debug(
                f"Column: {column} is of type {column_data.dtype}. Skipping normalisation")

        else:
            log_debug(
                f"Column: {column} is of type {column_data.dtype}. Skipping normalisation...")

    return normalised_data


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    """
    One-hot encodes the target variable.

    Parameters:
    - y (ndarray): The target variable.

    Returns:
    - y_encoded (ndarray): The one-hot encoded target variable.
    """

    # Initialize an empty list to store the encoded arrays for each variable
    encoded_arrays = []

    # Get the number of columns in y. Axis 0 is the number of samples in the dataset
    num_cols = y.shape[1]

    # Iterate over each variable in y
    for col in range(num_cols):
        # Get the unique values for the current variable
        unique_values = np.unique(y[:, col])

        # Initialize an empty array to store the encoded values for the current variable
        encoded_array = np.zeros((len(y), len(unique_values)), dtype=int)

        # Encode the current variable
        for i, value in enumerate(unique_values):
            encoded_array[:, i] = (y[:, col] == value).astype(int)

        # Append the encoded array for the current variable to the list
        encoded_arrays.append(encoded_array)

    # Concatenate the encoded arrays along the second axis to form the final encoded array
    y_encoded = np.concatenate(encoded_arrays, axis=1)

    return y_encoded


def one_hot_encode_separate_output_vars(y: np.ndarray) -> np.ndarray:
    """
    One-hot encodes the target variable where each column in y represents classes for a single variable.
    In the output, each variable's one-hot encoding is represented on a separate axis.

    Parameters:
    - y (ndarray): The target variable where each column contains classes for a single variable.

    Returns:
    - y_encoded (ndarray): The one-hot encoded target variable with each variable's one-hot encoding
                              represented on a separate axis.
    """

    log_info(f"y shape: {y.shape}")
    sample = 5

    encoded_data = []

    num_variables = y.shape[1]

    # Determine the number of classes for each variable
    num_classes = [len(np.unique(y[:, col])) for col in range(num_variables)]
    max_num_classes = max(num_classes)

    # Iterate over each variable in y
    for col in range(num_variables):
        log_info(f"Column: {col}")
        # Get the unique values for the current variable
        unique_values = np.unique(y[:, col])
        num_samples = len(y)
        log_debug(f"Unique Values: {unique_values}")

        # Initialize an empty array to store the encoded values for the current variable
        padded_encoded_array = np.zeros(
            (num_samples, max_num_classes), dtype=int)

        # For the current column, one-hot encode the values
        for i, value in enumerate(y[:, col]):
            one_hot = (unique_values == value).astype(int)
            padded_encoded_array[i, : len(one_hot)] = one_hot

        log_debug(
            f"Padded Encoded Array Sample: \n{padded_encoded_array[:sample]}")
        log_debug("")
        log_info(f"Padded Encoded Array Shape: {padded_encoded_array.shape}")

        # Append the encoded array for the current variable to the list
        encoded_data.append(padded_encoded_array)

        log_line()

    log_debug(f"Encoded vars sample: \n{encoded_data[:sample]}")

    y_encoded = np.stack(encoded_data, axis=1)
    # Stack the encoded arrays along the second axis to form the final encoded array
    log_info(f"y_encoded shape: {y_encoded.shape}")

    return y_encoded
