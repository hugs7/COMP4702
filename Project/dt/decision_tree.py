"""
Decision tree classification class
Hugo Burton
"""

from typing import Any, Tuple
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from model.classifier import Classifier
from logger import *


class DTClassifier(Classifier):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_labels: list[str],
        y_labels: list[list[str]],
        max_tree_depth: int = None,
    ):
        self.model = DecisionTreeClassifier(max_depth=max_tree_depth)

        super().__init__(self.model, X_train, y_train, X_test, y_test, X_labels, y_labels)

    def classify(self) -> Tuple[np.ndarray, float, float]:
        """
        Trains the decision tree model using the provided training data.

        Returns:
            - A tuple containing the test predictions, train accuracy, and test accuracy.
        """

        log_debug(
            f"X_train dim: {self.X_train.shape}, y_train dim: {self.y_train.shape}")

        log_title("Training decision tree model...")

        # Fit model
        self.model.fit(self.X_train, self.y_train)

        log_info("Decision tree model trained")

        # Get results
        log_title("Getting results...")

        # Results from training
        train_accuracy = self.model.score(self.X_train, self.y_train)

        # Test predictions
        test_predictions = self.model.predict(self.X_test)
        test_accuracy = self.model.score(self.X_test, self.y_test)

        log_info("Results obtained")

        log_debug(f"Test predictions:\n{test_predictions}")
        log_info(f"Train accuracy: {train_accuracy}")
        log_info(f"Test accuracy: {test_accuracy}")

        return test_predictions, train_accuracy, test_accuracy
