"""
Helper file to load data using pandas
"""

from colorama import Fore, Style
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import os
import replace_values
import encode_data
from typing import Literal, Sequence, Union


def load_data(file_path: str, header: Union[int, Sequence[int], Literal['infer'], None] = None) -> pd.DataFrame:
    df = None

    # If Excel file
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, header=header)

    # If CSV file
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path, header=header)

    # If data file
    elif file_path.endswith(".data"):
        df = pd.read_csv(file_path, header=header, delim_whitespace=True)

    return df


def check_file_exists(file_path: str) -> bool:
    file_exists = os.path.exists(file_path)
    if not file_exists:
        raise FileNotFoundError(f"File {file_path} not found")


def load_and_process_data(data_file: str, replace_null: bool = False, header: Union[int, Sequence[int], Literal['infer'], None] = None) -> pd.DataFrame:
    check_file_exists(data_file)

    # Load Data
    print(f"{Fore.LIGHTBLUE_EX}Loading Data{Style.RESET_ALL}")
    data = load_data(data_file, header)
    print(f"Data loaded from {data_file}")

    if replace_null:
        # Replace null values with mean instead
        print(f"{Fore.LIGHTBLUE_EX}Replacing Null Values with Mean{Style.RESET_ALL}")
        data = replace_values.replace_null_values_with_mean(data)
    else:
        # Remove null values
        print(f"{Fore.LIGHTBLUE_EX}Removing Null Values{Style.RESET_ALL}")
        data = replace_values.remove_null_values(data)

    # Encode non-numeric data
    print(f"{Fore.LIGHTBLUE_EX}Encoding Non-Numeric Data{Style.RESET_ALL}")
    data_encoded = encode_data.encode_non_numeric_data(data)

    return data_encoded


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
    X: DataFrame, y: DataFrame, ratio: float = 0.3
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
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


def process_classification_data(
    data_folder,
) -> Union[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, list[str]]:
    # Load the data
    file_path = os.path.join(data_folder, "w3classif.csv")

    data = load_data(file_path)

    data = tag_data(data, ["X1", "X2", "Y"])

    return data


def split_feature_response(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Splits the data into features and response variables.

    Parameters:
    - data (DataFrame): The data to split.

    Returns:
    - features (DataFrame): The features of the data.
    - response (DataFrame): The response variable of the data.
    """

    features = data.iloc[:, :-1]
    response = data.iloc[:, -1]

    return features, response
