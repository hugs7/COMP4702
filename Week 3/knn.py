"""
Applies knn classifier to data

"""

from colorama import Fore
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame

from model import Model


class KNNClassify(Model):
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

        # Define Model
        self.model = KNeighborsClassifier()

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

    def plot_decision_regions(
        self,
        X_test: DataFrame,
        test_preds: np.ndarray,
        resolution=0.02,
    ) -> None:
        """
        Plots the decision regions for a classifier.

        Parameters:
        - X_test (DataFrame): The input data used for testing the classifier.
        - test_preds (ndarray): The predicted labels for the test data.
        - resolution (float): The step size of the mesh grid used for plotting the decision regions. Default is 0.02.

        Returns:
        - None

        This function plots the decision regions for a classifier by creating a mesh grid based on the input data and
        classifying each point in the grid. The decision regions are then visualized using a contour plot.

        Note:
        - The input data should have exactly two features for proper visualization.
        - The classifier should have a `predict` method that takes a feature matrix as input and returns the predicted labels.
        """

        X1_test = X_test.iloc[:, 0]
        X2_test = X_test.iloc[:, 1]

        print(X1_test.shape)
        print(X2_test.shape)

        # Print the range of your input features
        print("X1 Range:", X1_test.min(), "-", X1_test.max())
        print("X2 Range:", X2_test.min(), "-", X2_test.max())

        # Generate a meshgrid of points to cover the feature space
        x_min, x_max = X1_test.min() - 0.5, X1_test.max() + 0.5
        y_min, y_max = X2_test.min() - 0.5, X2_test.max() + 0.5

        print(x_min, x_max, y_min, y_max)

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
        )

        print(xx.shape, "|", yy.shape)

        feature_names = ["X1", "X2"]
        Z = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_names)

        Z_preds = self.model.predict(Z)

        print("Before reshape Z_pred:", Z_preds.shape)

        Z_preds = Z_preds.reshape(xx.shape)

        print(Z_preds.shape)

        # Plot the decision boundary
        cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
        plt.pcolormesh(xx, yy, Z_preds, cmap=cmap_light, shading="auto")

        # Overlay the test points
        cmap_bold = ListedColormap(["#FF0000", "#00FF00"])
        plt.scatter(X1_test, X2_test, c=test_preds, cmap=cmap_bold)

        # Setup plot
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("k-NN decision regions (k = %d)" % self.k)
        plt.show()


class KNNRegress(Model):
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

        # Define Model
        self.model = KNeighborsRegressor()

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

    def plot_regression_line(self, test_preds: np.ndarray) -> None:
        """
        Plots the regression line for a k-nearest neighbors regressor.

        Parameters:
        - test_preds (ndarray): The predicted target values for the test data.

        Returns:
        - None

        This function plots the regression line for a k-nearest neighbors regressor by plotting the test data points
        and the predicted values on the same graph.

        Note:
        - The input data should have exactly one feature for proper visualization.
        - The regressor should have a `predict` method that takes a feature matrix as input and returns the predicted target values.
        """

        plt.scatter(self.X_test, test_preds, color="blue")
        # plt.plot(self.X_test, self.model.predict(self.X_test), color="red")
        plt.title("k-NN Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
