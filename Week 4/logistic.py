"""
Defines logistic model
"""

import numpy as np

from sklearn.linear_model import LogisticRegression

from regressor import Regressor


class LogisticRegressionModel(Regressor):
    """
    Class for performing logistic regression
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
    ):
        self.model = LogisticRegression()

        # Call the super class constructor
        super().__init__(X_train, y_train, X_test, y_test, feature_names, self.model)

    def fit(self) -> None:
        """
        Fits the model to the training data
        """
        self.model.fit(self.X_train, self.y_train)
