import pandas as pd
from pandas import DataFrame
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from classifier import Classifier


class RFClassifier(Classifier):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
        n_trees: int = 100,
        max_tree_depth: int = 5,
    ) -> None:
        self.model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_tree_depth)

        super().__init__(X_train, y_train, X_test, y_test, feature_names, self.model)

    def classify(self) -> tuple[Any, float, float]:
        """
        Trains the random forest model using the provided training data.

        Args:

        Returns:
            - A tuple containing:
                - Test predications
                - Train accuracy
                - Test accuracy.
        """

        X = pd.DataFrame(self.X_train, columns=self.feature_names)
        y = self.y_train

        # Fit model
        self.model.fit(X, y)

        # Results from training
        train_accuracy = self.model.score(X, y)

        # Test predictions
        X = pd.DataFrame(self.X_test, columns=self.feature_names)
        y = self.y_test

        test_predictions = self.model.predict(X)
        test_accuracy = self.model.score(X, y)

        return test_predictions, train_accuracy, test_accuracy
