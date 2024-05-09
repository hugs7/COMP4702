"""
KNN Classification model class
Hugo Burton
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from model.classifier import Classifier
from logger import *


class KNNClassify(Classifier):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
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

    def classify(self) -> tuple[np.ndarray, float, float]:
        """
        Performs k-nearest neighbors classification on the given training and test data
        for the specified output variable.

        Returns:
            tuple[KNeighborsClassifier, np.ndarray, float, float]: A tuple containing the knn classifier, test predictions, train accuracy, and test accuracy.
        """

        log_info(f"X_train dim: {self.X_train.shape}, y_train dim: {self.y_train.shape}")

        log_title("Training KNN model...")

        self.model.fit(self.X_train, self.y_train)

        log_info("KNN model trained")

        log_title("Getting results...")
        # Get results
        train_accuracy = self.model.score(self.X_train, self.y_train)

        test_predictions = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)

        log_info("Results obtained")

        log_trace(f"Test predictions:\n{test_predictions}")
        log_info(f"Train accuracy: {train_accuracy}")
        log_info(f"Test accuracy: {test_accuracy}")

        return test_predictions, train_accuracy, test_accuracy
