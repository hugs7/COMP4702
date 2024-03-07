"""
Helper to plot dataset on a scatterplot
"""

from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scatterplot(data):
    fig = plt.figure()
    colours = ListedColormap(["r", "b"])
    scatter = plt.scatter(
        data.loc[:, "X1"],
        data.loc[:, "X2"],
        c=data.loc[:, "Y"],
        cmap=colours,
        label=data.loc[:, "Y"],
    )
    plt.legend(*scatter.legend_elements())
    plt.ylabel("X2")
    plt.xlabel("X1")
    plt.show()
