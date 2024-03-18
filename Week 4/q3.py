"""
Defines question 3 for week 4 prac
"""

import os
from load_data import load_and_process_data
from logistic import LogisticRegressionModel
from q2 import print_coefficients


def q3(data_folder):
    # Part A - Fit a logistic model to the data

    file_path = os.path.join(data_folder, "w3classif.csv")

    data = load_and_process_data(file_path, replace_null=True)

    # Get the features and target
    # Features are columns 0 to 1
    features = data.iloc[:, 0:1]
    # Target is column 2
    target = data.iloc[:, 2]

    feature_names = features.columns

    logistic_model = LogisticRegressionModel(
        features, target, features, target, feature_names
    )

    logistic_model.fit()

    # Parameters
    print("Parameters")
    coefs = logistic_model.model.coef_
    intercept = logistic_model.model.intercept_

    print_coefficients(coefs, intercept, logistic_model, features, target)

    # Part B - Test point (1.1, 1.1), what does your model predict as
    # p(y' = 1 | x = (1.1, 1.1))

    test_point = [[1.1, 1.1]]

    prob = logistic_model.model.predict_proba(test_point)

    print(f"Probability of y' = 1 given x = (1.1, 1.1): {prob[0][1]}")

    # Part C - Plot the data with the discriminant function
