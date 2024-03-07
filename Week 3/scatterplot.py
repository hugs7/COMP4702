"""
Helper to plot dataset on a scatterplot
"""

from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scatterplot_2d(data) -> None:
    """
    Create a scatter plot with colored data points for 2D data.

    Parameters:
    - data (pandas.DataFrame): The input data containing the columns for x, y, and color values.

    Returns:
    None
    """

    fig, ax = plt.subplots()
    colours = ListedColormap(["r", "b"])
    scatter = ax.scatter(
        data.iloc[:, 0],
        data.iloc[:, 1],
    )
    ax.legend(*scatter.legend_elements())
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    plt.show()


def scatterplot_with_colour(
    data, x_column: str, y_column: str, color_column: str
) -> None:
    """
    Create a scatter plot with colored data points.

    Parameters:
    - data (pandas.DataFrame): The input data containing the columns for x, y, and color values.
    - x_column (str): The name of the column in the data frame to be used as the x-axis values.
    - y_column (str): The name of the column in the data frame to be used as the y-axis values.
    - color_column (str): The name of the column in the data frame to be used as the color values.

    Returns:
    None
    """

    fig, ax = plt.subplots()
    colours = ListedColormap(["r", "b"])
    scatter = ax.scatter(
        data[x_column],
        data[y_column],
        c=data[color_column],
        cmap=colours,
        label=data[color_column],
    )
    ax.legend(*scatter.legend_elements())
    ax.set_ylabel(y_column)
    ax.set_xlabel(x_column)
    plt.show()
