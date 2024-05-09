"""
Classification Model Class
Hugo Burton
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.base import BaseEstimator

from model.base_model import Model
from logger import *


BACKGROUND_COLOURS = ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFD700", "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]
FOREGROUND_COLOURS = ["#FF0000", "#00FF00", "#0000FF", "#FFD700", "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]


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
        variable_feature_indices: tuple,
        resolution=0.02,
        plot_title="Decision Regions",
        buffer: float = 0.5,
        show_plot: bool = True,
    ) -> plt.Axes:
        """
        Plots the decision regions for a classifier.

        Parameters:
        - test_preds (ndarray): The predicted labels for the test data.
        - feature_indices (tuple): The indices of the two features to be used for plotting decision regions.
        - resolution (float): The step size of the mesh grid used for plotting the decision regions. Default is 0.02.
        - plot_title (str): The title of the plot. Default is "Decision Regions".
        - buffer (float): The buffer to add to the minimum and maximum values of the features. Default is 0.5.
        - show_plot (bool): Whether to display the plot. Default is True. Function will always return the plot object.

        Returns:
        - plot (plt.Axes): The plot object.

        This function plots the decision regions for a classifier by creating a mesh grid based on the input data and
        classifying each point in the grid. The decision regions are then visualized using a contour plot.
        """

        # Check feature_indices
        if len(variable_feature_indices) != 2:
            raise ValueError("Feature indices must be a tuple of length 2")

        # Check the indices are within the bounds of the feature array
        for index in variable_feature_indices:
            if index < 0 or index >= self.X_test.shape[1]:
                raise ValueError(f"Feature index {index} is out of bounds")

        # Extract feature columns based on the provided indices
        variable_features = self.X_test[:, variable_feature_indices]

        log_debug(f"Features:\n{variable_features}")

        # Calculate mean values of non-variable features
        mean_values = np.mean(self.X_test[:, ~np.isin(np.arange(self.X_test.shape[1]), variable_feature_indices)], axis=0)

        log_debug(f"Mean values:\n{mean_values}")

        # Generate meshgrid of points to cover the feature space while holding other features constant at their mean values
        mins = variable_features.min(axis=0) - buffer
        maxs = variable_features.max(axis=0) + buffer

        xx, yy = np.meshgrid(
            np.arange(mins[0], maxs[0], resolution),
            np.arange(mins[1], maxs[1], resolution),
        )

        log_trace(f"Meshgrid XX:\n{xx}")
        log_trace(f"Meshgrid YY:\n{yy}")

        log_debug(f"Meshgrid shape XX: {xx.shape}")
        log_debug(f"Meshgrid shape yy: {yy.shape}")

        log_debug(f"Meshgrid range X: {xx.min()} - {xx.max()}")
        log_debug(f"Meshgrid range yy: {yy.min()} - {yy.max()}")

        # Compute mean values for non-variable features
        constant_means = np.tile(mean_values, (xx.ravel().shape[0], 1))

        log_trace(f"Constant means:\n{constant_means}")
        log_debug(f"Constant means shape: {constant_means.shape}")

        # Combine variable features with constant mean values
        meshgrid = np.hstack((constant_means, np.c_[xx.ravel(), yy.ravel()]))

        log_trace(f"Meshgrid:\n{meshgrid}")
        log_debug(f"Meshgrid shape: {meshgrid.shape}")

        # Predict the labels for meshgrid points
        Z_preds = self.model.predict(meshgrid)

        # Reshape the predictions to match the meshgrid dimensions
        Z_preds = Z_preds.reshape(xx.shape)

        # Plot the decision boundary
        num_test_classes = len(np.unique(test_preds))
        cmap_bg = ListedColormap(BACKGROUND_COLOURS[:num_test_classes])
        dr_plot: plt.Axes = plt.gca()
        dr_plot.pcolormesh(xx, yy, Z_preds, cmap=cmap_bg, shading="auto")

        # Overlay the test points
        cmap_points = ListedColormap(FOREGROUND_COLOURS[:num_test_classes])
        dr_plot.scatter(variable_features[:, 0], variable_features[:, 1], c=test_preds, cmap=cmap_points)

        # Setup plot
        dr_plot.set_xlim(xx.min(), xx.max())
        dr_plot.set_ylim(yy.min(), yy.max())
        dr_plot.set_xlabel("Feature 1")
        dr_plot.set_ylabel("Feature 2")
        dr_plot.set_title(plot_title)

        if show_plot:
            dr_plot.show()

        return dr_plot
