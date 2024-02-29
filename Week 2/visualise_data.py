"""
Helper file for visualising the data using seaborn
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def plot_data_pairs(dataframe: pd.DataFrame):
    sb.pairplot(dataframe)
    plt.show()
