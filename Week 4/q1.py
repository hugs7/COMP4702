"""
Defines function for question 1
"""

import matplotlib.pyplot as plt
from linear import LinearRegressionModel
from polynomial import PolynomialRegressionModel
import numpy as np
import generate


def q1():
    # Part A

    f = lambda x: x**3 + 1

    domain = np.linspace(-1, 1, 1000)

    # Plot the function
    plt.plot(domain, f(domain))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.show()

    # Part B

    # Generate training points
    num_training_points = 30

    train_set = {
        generate.generate_training_point(f) for _ in range(num_training_points)
    }

    print("Training set", train_set)

    # Overlay training set points onto function
    plt.scatter(*zip(*train_set), color="red")

    plt.show()

    # Part C - Linear Regression

    # Extract x and y values from the training set
    x_train, y_train = zip(*train_set)
    # Convert to numpy arrays
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    # Perform linear regression on the training set
    linear_model = LinearRegressionModel(x_train, y_train, x_train, y_train, ["x"])

    linear_model.fit()

    # Perform testing on the same training set to get SSE
    predictions = linear_model.predict(x_train)
    mse = linear_model.mean_squared_error(y_train)

    print("Mean Squared Error:", mse)

    # Part D - Polynomial Regression

    errors = {}

    for degree in range(2, 8):
        polynomial_model = PolynomialRegressionModel(
            x_train, y_train, x_train, y_train, ["x"], degree=degree
        )

        polynomial_model.fit()

        # Perform testing on the same training set to get SSE
        predictions = polynomial_model.predict(x_train)
        mse = polynomial_model.mean_squared_error(y_train)
        errors[degree] = mse

    # Plot the error for each degree
    plt.plot(errors.keys(), errors.values())
    plt.xlabel("Degree")
    plt.ylabel("Mean Squared Error")
    plt.show()
