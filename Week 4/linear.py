"""
Module for helping with performing linear regression on data
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from regressor import Regressor


class LinearRegressionModel(Regressor):
    """
    Class for performing linear regression on data
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
    ):
        self.model = LinearRegression()

        # Call the super class constructor
        super().__init__(X_train, y_train, X_test, y_test, feature_names, self.model)

    def fit(self) -> None:
        """
        Fits the model to the training data
        """
        self.model.fit(self.X_train, self.y_train)
