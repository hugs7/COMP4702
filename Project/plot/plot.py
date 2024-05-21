import os
import math
from typing import Dict, List, Tuple
import seaborn as sb
import numpy as np
import torch
import itertools
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import file_helper
from logger import *
import utils

import plot.mesh as mesh

BACKGROUND_COLOURS = ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFD700",
                      "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]
FOREGROUND_COLOURS = ["#FF0000", "#00FF00", "#0000FF", "#FFD700",
                      "#00CED1", "#FFA07A", "#98FB98", "#AFEEEE", "#D8BFD8", "#FFFFE0"]

PLOT_WIDTH = 25
PLOT_HEIGHT = 13

# 540p resolution for each subplot :)
PLOT_WIDTH_PIXELS = 340
PLOT_HEIGHT_PIXELS = 160


def lineplot(x_label: str, y_label: str, plot_title: str = "", save_path: str = None, *args: Tuple[np.ndarray, np.ndarray, str]):
    """
    Plots a line plot of the given data.

    Parameters:
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - plot_title (str): The title of the plot. Default is "".
    - save_path (str): The path to save the plot to. Default is None.
    - args (Tuple[np.ndarray, np.ndarray, str]): The data to plot. Each tuple contains
        - x (np.ndarray): The x values.
        - y (np.ndarray): The y values.
        - label (str): The label for the line.
    """

    sb.set_context("talk")
    sb.set_style("dark")
    plt.figure(figsize=(PLOT_WIDTH / 3, PLOT_HEIGHT / 2.2))

    for x, y, label in args:
        sb.lineplot(x=x, y=y, label=label)

    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if save_path is not None:
        file_helper.remove_file_if_exist(save_path)
        plt.savefig(save_path)

    plt.show()


def plot_gelu():
    geLu = torch.nn.GELU()

    input = torch.arange(-6, 6, step=0.1)
    output = geLu(input)

    lineplot("X", "GeLU(X)", "GeLU", None, (input, output, "GeLU"))

    return input


def plot_sigmoid(input):
    # Sigmoid activation function

    sigmoid = torch.nn.Sigmoid()

    # Input remains the same
    output = sigmoid(input)

    lineplot("X", "Sigmoid(X)", "Sigmoid", None, (input, output, "Sigmoid"))


def plot_knn_accuracies(accuracies_by_k: Dict[int, float], dataset_name: str, var_y_name: str, plots_folder_path: str) -> None:
    """
    Plots the accuracies for different values of k as a line plot.

    Parameters:
    - accuracies_by_k (Dict[int, float]): The accuracies for different values of k.
    - dataset_name (str): The name of the dataset.
    - var_y_name (str): The name of the output variable.
    - plots_folder_path (str): The path to save the plots to.

    Returns:
    - None
    """

    log_info("Plotting k-NN accuracies...")

    plot_path = ""
    if plots_folder_path is not None:
        plot_path = os.path.join(
            plots_folder_path, f"{dataset_name}_knn_accuracies_{var_y_name}.png")

    k_vals = list(accuracies_by_k.keys())
    accuracies = list(accuracies_by_k.values())

    log_debug(f"K values: {k_vals}")
    log_debug(f"Accuracies: {accuracies}")

    data_tuple = (np.array(k_vals), np.array(accuracies), "Accuracy")

    lineplot("k", "Accuracy",
             f"kNN Cross Validation Accuracies for var {var_y_name} on data {dataset_name}", plot_path, data_tuple)


def plot_multivar_decision_regions(
    output_variable_name: str,
    test_preds: np.ndarray,
    ordered_predictor_indicies: np.ndarray,
    y_var_unique_classes: List[str],
    X_points: np.ndarray,
    X_labels: List[str],
    predict_callback: callable,
    delta: int = 5,
    model_name: str = None,
    dataset_name: str = None,
    plots_folder_path: str = None,
    use_tensors: bool = False,
) -> None:
    """
    Wrapper for plotting decision regions for more than 2 features. Places the top Delta
    features within a grid of subplots and plots the decision regions for each pair of features.

    Parameters:
    - output_variable_name (str): The name of the output variable.
    - test_preds (ndarray): The predicted labels for the test data.
    - ordered_predictor_indicies (ndarray): The indices of the predictors ordered by importance for this output variable. Should only be 1D.
    - y_var_unique_classes (List[str]): The unique class names for the output variable.
    - X_points (ndarray): The test data features.
    - X_labels (List[str]): The labels of the features.
    - predict_callback (callable): The function to use to predict the labels for the meshgrid points.
    - delta (int): The number of top features to plot. Default is 5. Note this is not the number of plots.
    - model_name (str): The name of the model. Default is None.
    - dataset_name (str): The name of the dataset. Default is None.
    - plots_folder_path (str): The path to save the plots to. Default is None.
    - use_tensors (bool): Whether to use tensors for the meshgrid. Default is False.

    Returns:
    - None
    """

    log_info(
        f"Plotting decision boundaries for output variable {output_variable_name}...")

    save_plot = False
    plot_path = ""
    if model_name is not None and dataset_name is not None and plots_folder_path is not None:
        save_plot = True
        plot_path = os.path.join(
            plots_folder_path, f"{model_name}_{dataset_name}_decision_boundaries_{output_variable_name}.png")

    # Clamp delta to the number of features
    if delta > len(X_labels):
        log_warning(
            f"Delta value {delta} exceeds the number of features {len(X_labels)}. Clamping to {len(X_labels)}")
        delta = min(delta, len(X_labels))

    # Check dimensions of ordered_predictor_indicies
    if len(ordered_predictor_indicies.shape) != 1:
        raise ValueError(
            "Ordered predictor indicies must be a 1D array of indices")

    # Plot decision regions for the top delta features
    top_predictor_indices = ordered_predictor_indicies[:delta]
    log_info(f"Top {delta} feature indices: {top_predictor_indices}")
    log_debug("X Labels: ", X_labels)
    top_delta_feature_cols = [X_labels[idx] for idx in top_predictor_indices]

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
        num_plots_per_row, num_plots_per_col, figsize=(PLOT_WIDTH, PLOT_HEIGHT))

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

        # Get the labels for the current pair of input variables
        feature_label_x = X_labels[feature_pair[0]]
        feature_label_y = X_labels[feature_pair[1]]

        # Set title for each subplot
        plot_title = f"Boundary plot for {output_variable_name}: {feature_label_x} vs {feature_label_y}"

        # Generate and plot decision regions for the current pair of input variables
        subplot, scatter = plot_decision_regions(
            output_variable_name,
            test_preds,
            feature_pair,
            X_labels,
            y_var_unique_classes,
            X_points,
            predict_callback,
            plot_title=plot_title,
            show_plot=False,
            show_legend=False,
            x_label=feature_label_x,
            y_label=feature_label_y,
            use_tensors=use_tensors,
        )

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

    log_line(level="TRACE")
    log_trace("X Test points:")
    X_test_important_features = X_points[:, top_predictor_indices]
    log_trace(utils.np_to_pd(X_test_important_features,
              top_delta_feature_cols, use_tensors=use_tensors))
    log_line(level="TRACE")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Add a global legend
    # Set the position of the legend to the top right of the plot
    scatter_legend = Legend(fig, scatter.legend_elements(
    )[0], y_var_unique_classes, title=output_variable_name, loc="upper right")
    fig.add_artist(scatter_legend)

    if save_plot:
        file_helper.remove_file_if_exist(plot_path)
        log_info(f"Saving plot to {plot_path}")
        plt.savefig(plot_path)

    # Show the decision boundary plots for the current classifier
    plt.show()


def plot_decision_regions(
    output_variable_name: str,
    test_preds: np.ndarray,
    variable_feature_indices: tuple,
    all_col_labels: list[str],
    classes: list[str],
    X_points: np.ndarray | torch.Tensor,
    predict_callback: callable,
    plot_title="Decision Regions",
    buffer: float = 0.5,
    show_plot: bool = True,
    show_legend: bool = True,
    x_label: str = "X",
    y_label: str = "Y",
    use_tensors: bool = False,
) -> Tuple[plt.Axes, PathCollection]:
    """
    Plots the decision regions for a classifier.

    Parameters:
    - output_variable_name (str): The name of the output variable.
    - test_preds (ndarray | Tensor): The predicted labels for the test data.
    - variable_feature_indices (tuple): The indices of the two features to be used for plotting decision regions.
    - all_col_labels (list[str]): The labels of all the features.
    - classes (list[str]): The unique class labels.
    - X_points (ndarray): The test data features.
    - predict_callback (callable): The function to use to predict the labels for the meshgrid points.
    - plot_title (str): The title of the plot. Default is "Decision Regions".
    - buffer (float): The buffer to add to the minimum and maximum values of the features. Default is 0.5.
    - show_plot (bool): Whether to display the plot. Default is True. Function will always return the plot object.
    - show_legend (bool): Whether to display the legend. Default is True.
    - x_label (str): The label for the x-axis. Default is "X".
    - y_label (str): The label for the y-axis. Default is "Y".
    - use_tensors (bool): Whether to use tensors for the meshgrid. Default is False.

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
        if index < 0 or index >= X_points.shape[1]:
            raise ValueError(f"Feature index {index} is out of bounds")

    # Calculate mean values of non-variable features
    constant_feature_indices = list(
        set(range(X_points.shape[1])) - set(variable_feature_indices))

    log_debug(f"Variable feature indices: {variable_feature_indices}")
    log_debug(f"Constant feature indices: {constant_feature_indices}")

    variable_feature_labels = [all_col_labels[vfi]
                               for vfi in variable_feature_indices]
    constant_feature_labels = [all_col_labels[cfi]
                               for cfi in constant_feature_indices]

    log_debug(f"Variable feature labels: {variable_feature_labels}")

    # Extract feature columns based on the provided indices
    X_variable_features = X_points[:, variable_feature_indices]
    log_trace(f"X variable features:\n{X_variable_features}")
    log_debug(f"X points shape: {X_points.shape}")
    log_debug(f"X points dtype: {X_points.dtype}")

    if use_tensors:
        mean_values = torch.mean(X_points[:, constant_feature_indices], dim=0)
    else:
        mean_values = np.mean(X_points[:, constant_feature_indices], axis=0)

    log_debug(f"Mean values:")
    max_label_length = max([len(label) for label in all_col_labels])
    for i, mean_val in enumerate(mean_values):
        label = all_col_labels[constant_feature_indices[i]]
        log_debug(f"    {label:<{max_label_length+2}}: {mean_val}")

    log_debug("Generating X meshgrid...")
    # Generate meshgrid of points to cover the feature space while holding other features constant at their mean values
    if use_tensors:
        mins, _ = X_variable_features.min(axis=0)
        maxs, _ = X_variable_features.max(axis=0)
    else:
        mins = X_variable_features.min(axis=0)
        maxs = X_variable_features.max(axis=0)

    mins -= buffer
    maxs += buffer

    log_debug(f"Feature mins: {mins}")
    log_debug(f"Feature maxs: {maxs}")

    # Compute resolution from the range of the features
    resolution_x = (maxs[0] - mins[0]) / PLOT_WIDTH_PIXELS
    resolution_y = (maxs[1] - mins[1]) / PLOT_HEIGHT_PIXELS

    log_debug(f"Resolution X: {resolution_x}")
    log_debug(f"Resolution Y: {resolution_y}")

    if use_tensors:
        x = torch.arange(mins[0], maxs[0], resolution_x)
        y = torch.arange(mins[1], maxs[1], resolution_y)
    else:
        x = np.arange(mins[0], maxs[0], resolution_x)
        y = np.arange(mins[1], maxs[1], resolution_y)

    log_trace(f"X:\n{x}")
    log_trace(f"Y:\n{y}")

    log_debug(f"X shape: {x.shape}")
    log_debug(f"Y shape: {y.shape}")

    # Create meshgrid and flatten for predictions
    if use_tensors:
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
    else:
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
    flattened_X_meshgrid = mesh.construct_X_flattened_mesh(
        xx_flat, yy_flat, mean_values, variable_feature_indices, constant_feature_indices, all_col_labels, use_tensors=use_tensors
    )

    log_trace(utils.np_to_pd(flattened_X_meshgrid,
              all_col_labels, use_tensors=use_tensors))

    log_debug(f"Meshgrid shape: {flattened_X_meshgrid.shape}")
    if flattened_X_meshgrid.shape[1] != X_points.shape[1]:
        log_warning(
            f"Meshgrid shape {flattened_X_meshgrid.shape} does not match the number of features in the test data {X_points.shape}")

    log_line(level="DEBUG")
    log_debug("Making predictions on meshgrid...")
    # Predict the labels for meshgrid points
    Z_preds = predict_callback(flattened_X_meshgrid)
    log_trace(f"Predictions:\n{Z_preds}")
    log_line(level="DEBUG")
    # Reshape the predictions to match the meshgrid dimensions
    Z_preds = Z_preds.reshape(xx.shape)
    log_trace(f"Predictions (reshaped):\n{Z_preds}")
    log_line(level="DEBUG")

    # Plot the decision boundary
    num_test_classes = len(np.unique(test_preds))
    cmap_bg = ListedColormap(BACKGROUND_COLOURS[:num_test_classes])
    dr_plot: plt.Axes = plt.gca()
    dr_plot.pcolormesh(xx, yy, Z_preds, cmap=cmap_bg, shading="auto")

    if use_tensors:
        # Move X_variable_features to CPU if it is a tensor
        X_variable_features = utils.tensor_to_cpu(
            X_variable_features, detach=True)

    # Overlay the test points
    cmap_points = ListedColormap(FOREGROUND_COLOURS[:num_test_classes])

    log_debug(f"Test preds: {test_preds}")
    log_debug(f"X variable features shape: {X_variable_features.shape}")
    scatter = dr_plot.scatter(
        X_variable_features[:, 0], X_variable_features[:, 1], c=test_preds, cmap=cmap_points)

    # Setup plot
    xx_min, xx_max = xx.min(), xx.max()
    yy_min, yy_max = yy.min(), yy.max()

    if xx_min == xx_max:
        xx_min -= 1
        xx_max += 1

    if yy_min == yy_max:
        yy_min -= 1
        yy_max += 1

    log_debug(f"X min: {xx_min}, X max: {xx_max}")
    log_debug(f"Y min: {yy_min}, Y max: {yy_max}")

    dr_plot.set_xlim(xx_min, xx_max)
    dr_plot.set_ylim(yy_min, yy_max)
    dr_plot.set_xlabel(x_label)
    dr_plot.set_ylabel(y_label)
    dr_plot.set_title(plot_title)

    # Add legend if required
    if show_legend:
        log_info("Rendering legend...")

        log_debug("classes: ", classes)

        scatter_legend = Legend(dr_plot, scatter.legend_elements(
        )[0], classes, title=output_variable_name, loc="upper right")

        dr_plot.add_artist(scatter_legend)

    if show_plot:
        dr_plot.show()

    return dr_plot, scatter
