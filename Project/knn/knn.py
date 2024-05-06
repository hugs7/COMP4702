"""
KNN Classification model class
Hugo Burton
"""

from colorama import Fore, Style
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from model.classifier import Classifier


class KNNClassify(Classifier):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
        k: int = 3,
    ):
        self.k = k
        # Define Model
        self.model = KNeighborsClassifier(n_neighbors=self.k)

        super().__init__(X_train, y_train, X_test, y_test, feature_names, self.model)

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

        print(
            f"{Fore.RED}X_train dim: {self.X_train.shape}, y_train dim: {self.y_train.shape}{Style.RESET_ALL}"
        )

        X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        self.model.fit(X_train_df, self.y_train)

        # Get results
        train_accuracy = self.model.score(self.X_train, self.y_train)

        test_predictions = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)

        return test_predictions, train_accuracy, test_accuracy
