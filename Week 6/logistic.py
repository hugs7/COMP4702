"""
Defines logistic model
"""

import numpy as np

from sklearn.metrics import accuracy_score
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

    def predict(self) -> np.ndarray:
        """
        Predicts the test data
        """
        test_predictions = self.model.predict(self.X_test)

        # Get results
        train_accuracy = self.model.score(self.X_train, self.y_train)

        test_accuracy = accuracy_score(self.y_test, test_predictions)

        return test_predictions, train_accuracy, test_accuracy

    def predict_proba(self, classes: list[str], threshold: float = 0.5) -> np.ndarray:
        """
        Predicts the test data
        """
        test_probabilities = self.model.predict_proba(self.X_test)

        # Apply threshold to determine predicted class
        test_predictions_num = (
            test_probabilities[:, 1] > threshold).astype(int)
        test_predictions = np.array(classes)[test_predictions_num]

        # Get results
        train_accuracy = self.model.score(self.X_train, self.y_train)

        test_accuracy = accuracy_score(self.y_test, test_predictions)

        return test_predictions, train_accuracy, test_accuracy
