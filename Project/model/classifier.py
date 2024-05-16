"""
Classification Model Class
Hugo Burton
"""

import itertools
import math
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.base import BaseEstimator
from typing import List, Tuple
import utils

from model.base_model import Model
from logger import *


BACKGROUND_COLOURS = ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFD700",
                      "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]
FOREGROUND_COLOURS = ["#FF0000", "#00FF00", "#0000FF", "#FFD700",
                      "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]


def construct_X_flattened_mesh(
    xx_flat: np.ndarray,
    yy_flat: np.ndarray,
    mean_values: np.ndarray,
    variable_indices: List[int],
    constant_indices: List[int],
    all_col_labels: List[str],
) -> np.ndarray:
    """
    Constructs a flattened meshgrid of points of which the variable columns contain the range of the plot and the
    constant columns are the mean of the X_test set. This is used to predict the class of each point to obtain the
    Z_preds on the background of the decision boundary plot.

    Parameters:
    - xx_flat (ndarray): The flattened meshgrid of points for the first variable feature.
    - yy_flat (ndarray): The flattened meshgrid of points for the second variable feature.
    - mean_values (ndarray): The means of non-variable features.
    - variable_indices (List[int]): The indices of the variable features in the meshgrid.
    - constant_indices (List[int]): The indices of the non-variable features.
    - all_col_labels (List[str]): The labels of all the features.

    Returns:
    - flattened_meshgrid (ndarray): The flattened meshgrid of points with the variable features and constant features.
    """

    log_debug("Reconstructing meshgrid...")

    log_trace("Checking input parameters...")

    # Check length of variable_indices is 2
    num_variable_features = len(variable_indices)
    if num_variable_features != 2:
        raise ValueError("Variable indices must be a list of length 2")

    # Check length of xx and yy match
    if xx_flat.shape[0] != yy_flat.shape[0]:
        raise ValueError("Meshgrid dimensions do not match")

    # Check dimensionality of tiled_means matches the length of tiled_means_indices
    num_constant_features = len(constant_indices)
    if mean_values.shape[0] != num_constant_features:
        raise ValueError(
            f"Dimensionality of means {mean_values.shape[0]} does not match the number of non-variable features "
            + f"{num_constant_features}.\ntiled_means_indices: {constant_indices}"
        )

    # Create an empty array to store the reconstructed meshgrid
    total_meshgrid_features = num_variable_features + num_constant_features
    log_debug(f"Total meshgrid features: {total_meshgrid_features}")

    meshgrid_length = xx_flat.shape[0]
    # = yy_flat.shape[0] which was checked above
    log_debug(f"Meshgrid length: {meshgrid_length}")

    # Compute mean values for non-variable features
    constant_col_labels = [all_col_labels[cfi] for cfi in constant_indices]

    log_trace(f"Mean values:\n{mean_values}")
    tiled_means = np.tile(mean_values, (meshgrid_length, 1))
    log_debug(f"Tiled means shape: {tiled_means.shape}")
    log_trace(
        f"Tiled means:\n{utils.np_to_pd(tiled_means, constant_col_labels)}")

    flattened_meshgrid = np.empty((meshgrid_length, total_meshgrid_features))
    log_debug(f"Empty flattened meshgrid shape: {flattened_meshgrid.shape}")
    log_trace(
        f"Empty flattened meshgrid:\n{utils.np_to_pd(flattened_meshgrid, all_col_labels)}")

    # Insert the variable features into the meshgrid
    # Split variable indices into x and y indices
    x_index, y_index = variable_indices
    flattened_meshgrid[:, x_index] = xx_flat
    flattened_meshgrid[:, y_index] = yy_flat

    variable_col_labels = [all_col_labels[vfi] for vfi in variable_indices]
    log_debug(f"Variable column labels: {variable_col_labels}")
    log_trace(
        f"Flattened meshgrid with only variable columns inserted:\n{utils.np_to_pd(flattened_meshgrid, all_col_labels)}")

    # Insert the means of non-variable features into the meshgrid
    flattened_meshgrid[:, constant_indices] = tiled_means

    log_debug(f"Constant column labels: {constant_col_labels}")
    log_trace(
        f"Flattened meshgrid with variable and constant columns inserted:\n{utils.np_to_pd(flattened_meshgrid, all_col_labels)}")

    return flattened_meshgrid


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

    def legend_labels(self, classes: List[str]) -> List[str]:
        """
        Returns the class labels for the output variable.

        Parameters:
        - classes (List[str]): The unique class labels for the output variable.

        Returns:
        - List[str]: The class labels for the output variable.
        """

        return [f"{class_label}" for class_label in classes]

    def plot_multivar_decision_regions(
        self,
        output_variable_name: str,
        test_preds: np.ndarray,
        ordered_predictor_indicies: np.ndarray,
        y_var_unique_classes: List[str],
        delta: int = 5,
    ) -> None:
        """
        Wrapper for plotting decision regions for more than 2 features. Places the top Delta
        features within a grid of subplots and plots the decision regions for each pair of features.

        Parameters:
        - output_variable_name (str): The name of the output variable.
        - test_preds (ndarray): The predicted labels for the test data.
        - ordered_predictor_indicies (ndarray): The indices of the predictors ordered by importance for this output variable. Should only be 1D.
        - y_var_unique_classes (List[str]): The unique class names for the output variable.
        - delta (int): The number of top features to plot. Default is 5. Note this is not the number of plots.

        Returns:
        - None
        """

        log_info(
            f"Plotting decision boundaries for output variable {output_variable_name}...")

        # Clamp delta to the number of features
        if delta > len(self.X_labels):
            log_warning(
                f"Delta value {delta} exceeds the number of features {len(self.X_labels)}. Clamping to {len(self.X_labels)}")
            delta = min(delta, len(self.X_labels))

        # Plot decision regions for the top delta features
        top_predictor_indices = ordered_predictor_indicies[:delta]
        log_info(f"Top {delta} feature indices: {top_predictor_indices}")
        top_delta_feature_cols = [self.X_labels[idx]
                                  for idx in top_predictor_indices]

        # Calculate the total number of plots
        feature_combinations = list(
            itertools.combinations(top_predictor_indices, 2))
        log_trace(f"Feature combinations: {feature_combinations}")
        num_feature_pairs = len(feature_combinations)
        log_debug(f"Total number of plots: {num_feature_pairs}")

        # Determine the number of rows and columns for the square grid
        num_plots_per_row = math.ceil(math.sqrt(num_feature_pairs))
        num_plots_per_col = math.ceil(num_feature_pairs / num_plots_per_row)

        # Create a square grid of subplots
        fig, axs = plt.subplots(
            num_plots_per_row, num_plots_per_col, figsize=(10, 6))

        # Flatten the axs array to iterate over it easily
        # Flatten only if there is more than one row
        if num_plots_per_row > 1:
            axs = axs.flatten()

        # Iterate over each pair of input variables
        plot_index = 0

        for i, feature_pair in enumerate(feature_combinations):
            log_info(
                f"Plotting decision boundary for feature pair {feature_pair}. Progress: {i} / {num_feature_pairs}")

            # Get the current axes
            # If only 1 plot, axs is not an array
            if num_plots_per_row > 1:
                plt.sca(axs[plot_index])
            else:
                plt.sca(axs)

            # Generate and plot decision regions for the current pair of input variables
            subplot, scatter = self.plot_decision_regions(
                output_variable_name, test_preds, feature_pair, self.X_labels, y_var_unique_classes, show_plot=False, resolution=1, show_legend=False)
            # Set title for each subplot
            feature_label_x = self.X_labels[feature_pair[0]]
            feature_label_y = self.X_labels[feature_pair[1]]
            subplot.set_title(
                f"DB for features {feature_label_x} and {feature_label_y}")
            subplot.set_xlabel(feature_label_x)
            subplot.set_ylabel(feature_label_y)

            # Add subplot to the list of plots
            if num_plots_per_row > 1:
                axs[plot_index] = subplot
            else:
                axs = subplot

            # Increment plot index
            plot_index += 1

        log_debug("All decision boundary plots generated")

        # Hide empty subplots
        num_axes = len(axs) if isinstance(axs, np.ndarray) else 1
        for j in range(num_feature_pairs, num_axes):
            axs[j].axis("off")

        log_line(level="DEBUG")
        log_debug("X Test points:")
        X_test_important_features = self.X_test[:, top_predictor_indices]
        log_debug(utils.np_to_pd(
            X_test_important_features, top_delta_feature_cols))
        log_line(level="DEBUG")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Add a global legend
        # Set the position of the legend to the top right of the plot
        legend_labels = self.legend_labels(y_var_unique_classes)
        scatter_legend = Legend(
            fig, scatter.legend_elements()[0], legend_labels, title=output_variable_name, loc="upper right")
        fig.add_artist(scatter_legend)

        # Show the decision boundary plots for the current KNN classifier
        plt.show()

    def plot_decision_regions(
        self,
        output_variable_name: str,
        test_preds: np.ndarray,
        variable_feature_indices: tuple,
        all_col_labels: list[str],
        classes: list[str],
        resolution=0.02,
        plot_title="Decision Regions",
        buffer: float = 0.5,
        show_plot: bool = True,
        show_legend: bool = True,
    ) -> Tuple[plt.Axes, PathCollection]:
        """
        Plots the decision regions for a classifier.

        Parameters:
        - output_variable_name (str): The name of the output variable.
        - test_preds (ndarray): The predicted labels for the test data.
        - variable_feature_indices (tuple): The indices of the two features to be used for plotting decision regions.
        - all_col_labels (list[str]): The labels of all the features.
        - classes (list[str]): The unique class labels.
        - resolution (float): The step size of the mesh grid used for plotting the decision regions. Default is 0.02.
        - plot_title (str): The title of the plot. Default is "Decision Regions".
        - buffer (float): The buffer to add to the minimum and maximum values of the features. Default is 0.5.
        - show_plot (bool): Whether to display the plot. Default is True. Function will always return the plot object.
        - show_legend (bool): Whether to display the legend. Default is True.

        Returns:
        - Tuple[plt.Axes, PathCollection]: The plot and the scatter plot object.

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

        log_debug(f"Plot resolution: {resolution}")

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
            log_debug(f"    {label:<{max_label_length+2}}: {mean_val}")

        log_debug("Generating X meshgrid...")
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

        # Create meshgrid and flatten for predictions
        xx, yy = np.meshgrid(x, y)
        xx_flat = xx.ravel()
        yy_flat = yy.ravel()

        log_trace(f"Meshgrid flat XX:\n{xx_flat}")
        log_trace(f"Meshgrid flat YY:\n{yy_flat}")

        log_debug(f"Meshgrid flat shape XX: {xx_flat.shape}")
        log_debug(f"Meshgrid flat shape yy: {yy_flat.shape}")

        log_debug(f"Meshgrid flat range XX: {xx_flat.min()} - {xx_flat.max()}")
        log_debug(f"Meshgrid flat range yy: {yy_flat.min()} - {yy_flat.max()}")

        # Construct a "fake" / contrived test set retaining original feature variable order
        flattened_X_meshgrid = construct_X_flattened_mesh(
            xx_flat, yy_flat, mean_values, variable_feature_indices, constant_feature_indices, all_col_labels
        )

        log_trace(utils.np_to_pd(flattened_X_meshgrid, all_col_labels))

        log_debug(f"Meshgrid shape: {flattened_X_meshgrid.shape}")
        if flattened_X_meshgrid.shape[1] != self.X_test.shape[1]:
            log_warning(
                f"Meshgrid shape {flattened_X_meshgrid.shape} does not match the number of features in the test data {self.X_test.shape}"
            )

        log_line(level="DEBUG")
        log_debug("Making predictions on meshgrid...")
        # Predict the labels for meshgrid points
        Z_preds = self.model.predict(flattened_X_meshgrid)
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
        scatter = dr_plot.scatter(
            X_variable_features[:, 0], X_variable_features[:, 1], c=test_preds, cmap=cmap_points)

        # Setup plot
        dr_plot.set_xlim(xx.min(), xx.max())
        dr_plot.set_ylim(yy.min(), yy.max())
        dr_plot.set_xlabel("X")
        dr_plot.set_ylabel("Y")
        dr_plot.set_title(plot_title)

        # Add legend if required
        if show_legend:
            log_info("Rendering legend...")

            log_debug("classes: ", classes)
            legend_labels = self.legend_labels(classes)

            scatter_legend = Legend(
                dr_plot, scatter.legend_elements()[0], legend_labels, title=output_variable_name, loc="upper right")

            dr_plot.add_artist(scatter_legend)

        if show_plot:
            dr_plot.show()

        return dr_plot, scatter
