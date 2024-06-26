from model import Model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.base import BaseEstimator


class Classifier(Model):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
        model: BaseEstimator,
    ) -> None:
        super().__init__(X_train, y_train, X_test, y_test, feature_names)

        self.model = model

    def plot_decision_regions(
        self,
        resolution=0.02,
        plot_title="Decision Regions",
    ) -> None:
        """
        Plots the decision regions for a classifier and overlays the test data.
        Wrongly classified points are indicated by being positioned outside their 
        respective decision region.

        Parameters:
        - X_test (DataFrame): The input data used for testing the classifier.
        - test_preds (ndarray): The predicted labels for the test data.
        - resolution (float): The step size of the mesh grid used for plotting the decision regions. Default is 0.02.

        Returns:
        - None
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
        cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
        plt.scatter(X1_test, X2_test, c=self.y_test
                    , cmap=cmap_bold)

        # Setup plot
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(plot_title)
        plt.show()
