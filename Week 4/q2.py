"""
Defined the function for question 2
"""

import os
from colorama import Fore, Style

from load_data import load_and_process_data
from linear import LinearRegressionModel


def q2(data_folder: str):
    # Part A - Import the data

    file_path = os.path.join(data_folder, "pokemonregr.csv")

    data = load_and_process_data(file_path, replace_null=True)

    print(data.head())

    print(data.info())

    # Get the features and target
    # Features are columns 0 to 5
    features = data.iloc[:, 0:5]
    # Target is column 6
    target = data.iloc[:, 6]

    # Part B - Fit Linear Regression Model
    linear_model = LinearRegressionModel(
        features, target, features, target, features.columns
    )

    linear_model.fit()

    # Part C - Get the coefficients
    print(f"{Fore.LIGHTCYAN_EX}Coefficients:{Fore.WHITE}{linear_model.model.coef_}")
    print(f"{Fore.LIGHTCYAN_EX}Intercept:{Fore.WHITE}{linear_model.model.intercept_}")
    print(
        f"{Fore.LIGHTCYAN_EX}R^2:{Fore.WHITE}{linear_model.model.score(features, target)}"
    )
