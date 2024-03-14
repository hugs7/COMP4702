"""
Week 4 Prac Main file
"""

import matplotlib.pyplot as plt
import numpy as np
from linear import LinearRegressionModel


def generate_training_point(f: callable, noise: float = 0.2) -> float:
    """
    Generates a single training point with noise from the function f
    """

    # Generate a random x value
    x = np.random.uniform(-1, 1)

    # Generate a random noise value using a gaussian distribution
    noise = np.random.normal(0, noise)

    # Generate the y value
    y = f(x) + noise

    return x, y


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

    train_set = {generate_training_point(f) for _ in range(num_training_points)}

    print("Training set", train_set)

    # Overlay training set points onto function
    plt.scatter(*zip(*train_set), color="red")

    plt.show()

    # Part C

    # Extract x and y values from the training set
    x_train, y_train = zip(*train_set)
    # Convert to numpy arrays
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    # Perform linear regression on the training set
    linear_model = LinearRegressionModel(x_train, y_train)

    linear_model.fit()

    # Perform testing on the same training set to get SSE
    predictions = linear_model.predict(x_train)
    mse = linear_model.mean_squared_error(y_train)

    print("Mean Squared Error:", mse)

    # Part D


def q2():
    pass


def main():
    print("Week 4 Prac Main file")

    # Question 1
    q1()


if __name__ == "__main__":
    main()
