"""
Implements question 3 from Week 6 Prac
"""

from load_data import load_and_process_data, split_feature_response, test_train_split
import os
from colorama import Fore, Style
from linear_helper import linear_fit


def q3(data_folder: str):
    """
    In Prac W4 we applied linear regression to a pokemon dataset, where the loss function was sum 
    of squares (or mean squared) error. Revisit this task but add (a) L2; (b) L1 regularisation to
    the loss function, with some suitable value for the regularization hyperparameter (see Section 
    5.3 of the textbook). Compare the coefficient values from your different trained models. 
    """

    pokemon_data = "pokemonregr.csv"
    pokemon_data_path = os.path.join(data_folder, pokemon_data)

    # Load the data
    pokemon_data = load_and_process_data(
        pokemon_data_path, replace_null=True, header="infer")

    X_data, y_data = split_feature_response(pokemon_data)
    feature_names = X_data.columns
    print(f"{Fore.LIGHTMAGENTA_EX}Feature names: {Style.RESET_ALL}{feature_names}")

    X_train, y_train, X_test, y_test = test_train_split(X_data, y_data)

    penalties = ["l1", "l2", "elasticnet"]

    for penalty in penalties:
        print(f"{'-'*100}\n{Fore.LIGHTRED_EX}Penalty: {Style.RESET_ALL}{penalty}")

        if penalty == "elasticnet":
            linear_fit(X_train, y_train, X_test, y_test,
                       feature_names, penalty=penalty, l1_ratio=0.5)
        else:
            linear_fit(X_train, y_train, X_test, y_test,
                       feature_names, penalty=penalty)
