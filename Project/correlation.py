"""
Plotting the correlation matrix of the data
"""

import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame


def plot_correlation_matrix(data: DataFrame, title: str) -> None:
    """
    Plots the correlation matrix of the data.

    Args:
    - data (DataFrame): The data to plot the correlation matrix of.
    - title (str): The title of the plot.
    """

    # Compute correlation matrix
    corr = data.corr(numeric_only=True)

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title(title)
    plt.show()
