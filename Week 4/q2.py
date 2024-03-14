"""
Defined the function for question 2
"""

import os
from colorama import Fore, Style

from load_data import load_and_process_data
from linear import LinearRegressionModel
from sklearn.preprocessing import StandardScaler


def print_coefficients(coefs, intercept, linear_model, features, target):
    print(f"{Fore.LIGHTCYAN_EX}Coefficients:{Fore.WHITE}{coefs}")
    print(f"{Fore.LIGHTCYAN_EX}Intercept:{Fore.WHITE}{intercept}")
    print(
        f"{Fore.LIGHTCYAN_EX}R^2:{Fore.WHITE}{linear_model.model.score(features, target)}"
    )


def variable_importance(coefs, feature_names):
    coefs_dict = {(i, c) for i, c in enumerate(coefs)}
    coefs_dict = sorted(coefs_dict, key=lambda x: abs(x[1]), reverse=True)
    print(f"{Fore.LIGHTCYAN_EX}Variable Importance:{Fore.WHITE}")
    for i, c in coefs_dict:
        print(f"    {Fore.LIGHTMAGENTA_EX}{feature_names[i]:>10}:{Fore.WHITE} {c}")


def fit_model_and_print_results(features, target, feature_names):
    # Fit the model
    linear_model = LinearRegressionModel(
        features, target, features, target, feature_names
    )

    linear_model.fit()

    # Get the coefficients
    coefs = linear_model.model.coef_
    intercept = linear_model.model.intercept_
    print_coefficients(coefs, intercept, linear_model, features, target)

    # Variable Importance
    # Using absolute value of coefficients as importance
    variable_importance(coefs, feature_names)


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
    fit_model_and_print_results(features, target, features.columns)

    # Part E - Normalise the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

    # Fit the model again
    print(f"{Fore.RED}Fitting model with normalised data{Fore.WHITE}")
    fit_model_and_print_results(features_scaled, target_scaled, features.columns)
