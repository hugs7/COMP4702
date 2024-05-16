"""
Decision tree classification class
Hugo Burton
"""

from typing import Any, Tuple
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from model.classifier import Classifier


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

        X = pd.DataFrame(self.X_train, columns=self.X_labels)
        y = self.y_train

        # Fit model
        self.model.fit(X, y)

        # Results from training
        train_accuracy = self.model.score(X, y)

        # Test predictions
        test_predictions = self.model.predict(self.X_test)
        test_accuracy = self.model.score(self.X_test, self.y_test)

        return test_predictions, train_accuracy, test_accuracy
