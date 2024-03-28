"""
Module for helping with performing linear regression on data with regularization
"""

from typing import Union
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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
        penalty: Union[None, str] = None,
        l1_ratio: Union[float, None] = None,
        alpha: float = 1.0  # Regularization strength (alpha parameter)
    ):
        if penalty == 'l1':
            self.model = Lasso(alpha=alpha)
        elif penalty == 'l2':
            self.model = Ridge(alpha=alpha)
        elif penalty == 'elasticnet':
            if l1_ratio is None:
                raise ValueError(
                    "l1_ratio must be provided when penalty is 'elasticnet'"
                )
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        else:
            # No penalty
            self.model = LinearRegression()

        # Call the super class constructor
        super().__init__(X_train, y_train, X_test, y_test, feature_names, self.model)

    def fit(self) -> None:
        """
        Fits the model to the training data
        """
        self.model.fit(self.X_train, self.y_train)
