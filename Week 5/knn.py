"""
Applies knn classifier to data

"""

from colorama import Fore
import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

from classifier import Classifier
from regressor import Regressor


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
        self.model = KNeighborsClassifier()

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

        self.model = KNeighborsClassifier(n_neighbors=self.k)

        print(
            f"{Fore.RED}X_train dim: {self.X_train.shape}, y_train dim: {self.y_train.shape}{Fore.WHITE}"
        )

        X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        self.model.fit(X_train_df, self.y_train)

        # Get results
        train_accuracy = self.model.score(self.X_train, self.y_train)

        test_predictions = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)

        return test_predictions, train_accuracy, test_accuracy


class KNNRegress(Regressor):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
        k: int = 3,
    ):
        # Define Model
        self.model = KNeighborsRegressor()
        self.k = k

        super().__init__(X_train, y_train, X_test, y_test, feature_names, self.model)

    def get_k(self) -> int:
        return self.k

    def regress(
        self,
    ) -> tuple[np.ndarray, float, float]:
        """
        Performs k-nearest neighbors regression on the given training and test data.

        Returns:
            tuple[KNeighborsRegressor, np.ndarray, float, float]: A tuple containing the knn regressor, test predictions, train accuracy, and test accuracy.
        """

        self.model = KNeighborsRegressor(n_neighbors=self.get_k())

        print(
            f"{Fore.RED}X_train dim: {self.X_train.shape}, y_train dim: {self.y_train.shape}{Fore.WHITE}"
        )

        X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        self.model.fit(X_train_df, self.y_train)

        # Get results
        train_accuracy = self.model.score(self.X_train, self.y_train)

        test_predictions = self.model.predict(self.X_test)
        test_accuracy = self.model.score(self.X_test, self.y_test)

        return test_predictions, train_accuracy, test_accuracy
