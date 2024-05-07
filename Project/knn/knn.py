"""
KNN Classification model class
Hugo Burton
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from model.classifier import Classifier
from logger import *


class KNNClassify(Classifier):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        X_labels: list[str],
        y_labels: list[list[str]],
        k: int = 3,
    ):
        self.k = k
        # Define Model
        self.model = KNeighborsClassifier(n_neighbors=self.k)

        super().__init__(self.model, X_train, y_train, X_test, y_test, X_labels, y_labels)

    def get_k(self) -> int:
        return self.k

    def classify(
        self,
    ) -> tuple[np.ndarray, float, float]:
        """
        Performs k-nearest neighbors classification on the given training and test data.

        Returns:
            tuple[KNeighborsClassifier, np.ndarray, float, float]: A tuple containing the knn classifier, test predictions, train accuracy, and test accuracy.
        """

        log_info(
            f"X_train dim: {self.X_train.shape}, y_train dim: {self.y_train.shape}"
        )

        X_train_df = pd.DataFrame(self.X_train, columns=self.X_labels)
        X_test_df = pd.DataFrame(self.X_test, columns=self.X_labels)

        self.model.fit(X_train_df, self.y_train)

        # Get results
        train_accuracy = self.model.score(self.X_train, self.y_train)

        test_predictions = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)

        return test_predictions, train_accuracy, test_accuracy
