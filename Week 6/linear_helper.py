

from pandas import DataFrame
import linear
from typing import Union
from colorama import Fore, Style


def print_coefficients(coefs, intercept, model, features, target):
    print(f"{Fore.LIGHTCYAN_EX}Coefficients:{Style.RESET_ALL}{coefs}")
    print(f"{Fore.LIGHTCYAN_EX}Intercept:{Style.RESET_ALL}{intercept}")
    print(f"{Fore.LIGHTCYAN_EX}R^2:{Style.RESET_ALL}{model.model.score(features, target)}")


def variable_importance(coefs, feature_names):
    coefs_dict = {(i, c) for i, c in enumerate(coefs)}
    coefs_dict = sorted(coefs_dict, key=lambda x: abs(x[1]), reverse=True)
    print(f"{Fore.LIGHTCYAN_EX}Variable Importance:{Style.RESET_ALL}")
    for i, c in coefs_dict:
        print(
            f"    {Fore.LIGHTMAGENTA_EX}{feature_names[i]:>10}:{Style.RESET_ALL} {c}")


def linear_fit(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame,
               y_test: DataFrame, feature_names: list[str],
               penalty="l1", l1_ratio: float = Union[float, None]) \
        -> DataFrame:
    """
    Classification task using linear regression with regularization
    """

    # Apply the linear classifier
    linear_model = linear.LinearRegressionModel(
        X_train, y_train, X_test, y_test, feature_names, penalty=penalty, l1_ratio=l1_ratio
    )

    linear_model.fit()

    # Get the coefficients
    coefs = linear_model.model.coef_
    intercept = linear_model.model.intercept_
    print_coefficients(coefs, intercept, linear_model, X_train, y_train)

    # Variable Importance
    # Using absolute value of coefficients as importance
    variable_importance(coefs, feature_names)
