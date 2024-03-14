"""
Polynomial class
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from regressor import Regressor
import matplotlib.pyplot as plt
import numpy as np


class PolynomialRegressionModel(Regressor):
    def __init__(self, X_train, y_train, X_test, y_test, feature_names, degree: int):
        self.model = LinearRegression()

        self.degree = degree

        # Create polynomial features
        self.poly = PolynomialFeatures(degree=self.degree)

        super().__init__(X_train, y_train, X_test, y_test, feature_names, self.model)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data using polynomial features
        """

        return self.poly.transform(X)

    def set_degree(self, degree: int) -> None:
        """
        Sets the degree of the polynomial features
        """

        self.degree = degree

        self.poly = PolynomialFeatures(degree=self.degree)

    def fit(self):
        # Fitting the linear regression model
        self.X_train_poly = self.poly.fit_transform(self.X_train)
        self.model.fit(self.X_train_poly, self.y_train)

    def predict(self, x):
        x_poly = self.poly.transform(x)

        return super().predict(x_poly)

    def mean_squared_error(self, y):
        return super().mean_squared_error(y)

    def plot(self, x, y):
        plt.scatter(x, y, color="red")
        plt.plot(x, self.predict(x), color="blue")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()
