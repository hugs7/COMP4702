"""
Defines parent class for regressor
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from model import Model
from sklearn.base import RegressorMixin


class Regressor(Model):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
        model: RegressorMixin,
    ) -> None:
        super().__init__(X_train, y_train, X_test, y_test, feature_names)

        self.model = model

    def plot_regression_line(self, test_preds: np.ndarray) -> None:
        """
        Plots the regression line for a regressor.

        Parameters:
        - test_preds (ndarray): The predicted target values for the test data.

        Returns:
        - None

        This function plots the regression line for a k-nearest neighbors regressor by plotting the test data points
        and the predicted values on the same graph.

        Note:
        - The input data should have exactly one feature for proper visualization.
        - The regressor should have a `predict` method that takes a feature matrix as input and returns the predicted target values.
        """

        print("Plotting regression line...")
        plt.scatter(self.X_test, test_preds, color="blue")
        # plt.plot(self.X_test, self.model.predict(self.X_test), color="red")
        plt.title("k-NN Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
