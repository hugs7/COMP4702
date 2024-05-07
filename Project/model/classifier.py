"""
Classification Model Class
Hugo Burton
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from model.base_model import Model


class Classifier(Model):
    def __init__(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_labels: list[str],
        y_labels: list[list[str]],
    ) -> None:
        super().__init__(X_train, y_train, X_test, y_test, X_labels, y_labels)

        self.model = model

    def plot_decision_regions(
        self,
        test_preds: np.ndarray,
        resolution=0.02,
        plot_title="Decision Regions",
    ) -> None:
        # TODO Fix this method to work with many features (more than 2)
        """
        Plots the decision regions for a classifier.

        Parameters:
        - X_test (ndarray): The input data used for testing the classifier.
        - test_preds (ndarray): The predicted labels for the test data.
        - resolution (float): The step size of the mesh grid used for plotting the decision regions. Default is 0.02.

        This function plots the decision regions for a classifier by creating a mesh grid based on the input data and
        classifying each point in the grid. The decision regions are then visualized using a contour plot.

        Note:
        - The input data should have exactly two features for proper visualization.
        - The classifier should have a `predict` method that takes a feature matrix as input and returns the predicted labels.
        """

        X1_test = self.X_test.iloc[:, 0]
        X2_test = self.X_test.iloc[:, 1]

        print(X1_test.shape)
        print(X2_test.shape)

        # Print the range of your input features
        print("X1 Range:", X1_test.min(), "-", X1_test.max())
        print("X2 Range:", X2_test.min(), "-", X2_test.max())

        # Generate a meshgrid of points to cover the feature space
        x_min, x_max = X1_test.min() - 0.5, X1_test.max() + 0.5
        y_min, y_max = X2_test.min() - 0.5, X2_test.max() + 0.5

        print(x_min, x_max, y_min, y_max)

        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

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
        plt.title(plot_title)
        plt.show()
