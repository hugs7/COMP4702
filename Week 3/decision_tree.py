"""
Defines the DecisionTree class, which is a decision tree classifier.
"""

from typing import Any, Tuple
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pandas import DataFrame

from model import Model


class DecisionTree(Model):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
        k: int = 3,
    ):
        super().__init__(X_train, X_test, y_train, y_test, feature_names, k)

        self.model = DecisionTreeClassifier()

    def train(self, X: DataFrame, y: DataFrame):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
