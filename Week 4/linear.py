"""
Module for helping with performing linear regression on data
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class LinearRegressionModel:
    """
    Class for performing linear regression on data
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model = LinearRegression()

        self.predictions = None
        self.x_train = x_train
        self.y_train = y_train

    def fit(self) -> None:
        """
        Fits the model to the training data
        """
        self.model.fit(self.x_train, self.y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the y values for the given x values
        """

        self.predictions = self.model.predict(x)

        return self.predictions

    def mean_squared_error(self, y: np.ndarray) -> float:
        """
        Returns the mean squared error for the given x and y values
        """
        if self.predictions is None:
            raise ValueError("No predictions have been made yet")

        return mean_squared_error(y, self.predictions)
