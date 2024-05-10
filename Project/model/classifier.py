"""
Classification Model Class
Hugo Burton
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.base import BaseEstimator
from typing import List
import utils

from model.base_model import Model
from logger import *


BACKGROUND_COLOURS = ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFD700",
                      "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]
FOREGROUND_COLOURS = ["#FF0000", "#00FF00", "#0000FF", "#FFD700",
                      "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]


def reconstruct_meshgrid(
    xx: np.ndarray, yy: np.ndarray, tiled_means: np.ndarray, variable_indices: List[int], tiled_means_indices: List[int]
) -> np.ndarray:
    """
    Reconstructs a meshgrid of points by inserting the means of non-variable features at the correct indices.

    Parameters:
    - xx (ndarray): The meshgrid of points for the first variable feature.
    - yy (ndarray): The meshgrid of points for the second variable feature.
    - tiled_means (ndarray): The means of non-variable features.
    - variable_indices (List[int]): The indices of the variable features in the meshgrid.
    - tiled_means_indices (List[int]): The indices of the non-variable features in the means array.

    Returns:
    - meshgrid (ndarray): The reconstructed meshgrid of points.
    """

    # Check length of variable_indices is 2
    num_variable_features = len(variable_indices)
    if num_variable_features != 2:
        raise ValueError("Variable indices must be a list of length 2")

    # Check length of xx and yy match
    if xx.shape[0] != yy.shape[0]:
        raise ValueError("Meshgrid dimensions do not match")

    # Check dimensionality of tiled_means matches the length of tiled_means_indices
    num_constant_features = len(tiled_means_indices)
    if tiled_means.shape[1] != num_constant_features:
        raise ValueError(
            f"Dimensionality of means {tiled_means.shape[1]} does not match the number of non-variable features {num_constant_features}.\ntiled_means_indices: {tiled_means_indices}"
        )

    # Create an empty array to store the reconstructed meshgrid
    total_meshgrid_features = num_variable_features + num_constant_features
    meshgrid_length = xx.ravel().shape[0]
    log_debug(f"Meshgrid length: {meshgrid_length}")

    meshgrid = np.empty((meshgrid_length, total_meshgrid_features))

    log_debug(f"Meshgrid shape: {meshgrid.shape}")

    # Insert the variable features into the meshgrid
    meshgrid[:, variable_indices] = np.c_[xx.ravel(), yy.ravel()]

    # log_trace(f"Meshgrid variable features:\n{meshgrid}")

    # Insert the means of non-variable features into the meshgrid
    # meshgrid[:, tiled_means_indices] = tiled_means

    return meshgrid


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
        all_col_labels: list[str],
        resolution=0.02,
        plot_title="Decision Regions",
        buffer: float = 0.5,
        show_plot: bool = True,
    ) -> plt.Axes:
        """
        Plots the decision regions for a classifier.

        Parameters:
        - test_preds (ndarray): The predicted labels for the test data.
        - variable_feature_indices (tuple): The indices of the two features to be used for plotting decision regions.
        - all_col_labels (list[str]): The labels of all the features.
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

        log_info(f"Plot resolution: {resolution}")

        # Calculate mean values of non-variable features
        constant_feature_indices = list(
            set(range(self.X_test.shape[1])) - set(variable_feature_indices))
        # constant_feature_indices = ~np.isin(np.arange(self.X_test.shape[1]), variable_feature_indices)

        log_debug(f"Variable feature indices: {variable_feature_indices}")
        log_debug(f"Constant feature indices: {constant_feature_indices}")

        variable_feature_labels = [all_col_labels[vfi]
                                   for vfi in variable_feature_indices]
        constant_feature_labels = [all_col_labels[cfi]
                                   for cfi in constant_feature_indices]

        log_debug(f"Variable feature labels: {variable_feature_labels}")

        # Extract feature columns based on the provided indices
        X_variable_features = self.X_test[:, variable_feature_indices]
        log_debug(f"X variable features:\n{X_variable_features}")

        mean_values = np.mean(self.X_test[:, constant_feature_indices], axis=0)
        log_debug(f"Mean values:")
        max_label_length = max([len(label) for label in all_col_labels])
        for i, mean_val in enumerate(mean_values):
            label = all_col_labels[constant_feature_indices[i]]
            log_debug(
                f"    {label:<{max_label_length+2}}: {mean_val}")

        # Generate meshgrid of points to cover the feature space while holding other features constant at their mean values
        mins = X_variable_features.min(axis=0) - buffer
        maxs = X_variable_features.max(axis=0) + buffer

        log_debug(f"Feature mins: {mins}")
        log_debug(f"Feature maxs: {maxs}")

        x = np.arange(mins[0], maxs[0], resolution)
        y = np.arange(mins[1], maxs[1], resolution)

        log_trace(f"X:\n{x}")
        log_trace(f"Y:\n{y}")

        log_debug(f"X shape: {x.shape}")
        log_debug(f"Y shape: {y.shape}")

        xx, yy = np.meshgrid(x, y)

        log_trace(f"Meshgrid XX:\n{xx}")
        log_trace(f"Meshgrid YY:\n{yy}")

        log_debug(f"Meshgrid shape XX: {xx.shape}")
        log_debug(f"Meshgrid shape yy: {yy.shape}")

        log_debug(f"Meshgrid range XX: {xx.min()} - {xx.max()}")
        log_debug(f"Meshgrid range yy: {yy.min()} - {yy.max()}")

        # Compute mean values for non-variable features
        tiled_means = np.tile(mean_values, (xx.ravel().shape[0], 1))

        log_trace(f"Constant means:")
        log_info(utils.np_to_pd(tiled_means, constant_feature_labels))
        log_debug(f"Constant means shape: {tiled_means.shape}")

        # Reconstruct a "fake" / contrived test set retaining original feature variable order
        meshgrid = reconstruct_meshgrid(
            xx, yy, tiled_means, variable_feature_indices, constant_feature_indices)

        log_info(utils.np_to_pd(meshgrid, all_col_labels))

        log_debug(f"Meshgrid shape: {meshgrid.shape}")
        if meshgrid.shape[1] != self.X_test.shape[1]:
            log_warning(
                f"Meshgrid shape {meshgrid.shape} does not match the number of features in the test data {self.X_test.shape}")

        log_line(level="DEBUG")
        log_info("Making predictions on meshgrid...")
        # Predict the labels for meshgrid points
        Z_preds = self.model.predict(meshgrid)
        log_debug(f"Predictions:\n{Z_preds}")
        log_line(level="DEBUG")
        # Reshape the predictions to match the meshgrid dimensions
        Z_preds = Z_preds.reshape(xx.shape)
        log_debug(f"Predictions (reshaped):\n{Z_preds}")
        log_line(level="DEBUG")

        # Plot the decision boundary
        num_test_classes = len(np.unique(test_preds))
        cmap_bg = ListedColormap(BACKGROUND_COLOURS[:num_test_classes])
        dr_plot: plt.Axes = plt.gca()
        dr_plot.pcolormesh(xx, yy, Z_preds, cmap=cmap_bg, shading="auto")

        # Overlay the test points
        cmap_points = ListedColormap(FOREGROUND_COLOURS[:num_test_classes])
        dr_plot.scatter(
            X_variable_features[:, 0], X_variable_features[:, 1], c=test_preds, cmap=cmap_points)

        # Setup plot
        dr_plot.set_xlim(xx.min(), xx.max())
        dr_plot.set_ylim(yy.min(), yy.max())
        dr_plot.set_xlabel("Feature 1")
        dr_plot.set_ylabel("Feature 2")
        dr_plot.set_title(plot_title)

        if show_plot:
            dr_plot.show()

        return dr_plot
