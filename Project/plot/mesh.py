"""
Helper script to construct a mesh grid used in plotting
"""

from typing import List
import numpy as np

from logger import *
import utils


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
