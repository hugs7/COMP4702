"""
Applies knn classifier to data

"""

from colorama import Fore
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame


def shuffle_data(data: DataFrame) -> DataFrame:
    """
    Shuffles the data randomly.

    Parameters:
    - data: A pandas DataFrame containing the data to be shuffled.

    Returns:
    - A new pandas DataFrame with the data shuffled randomly.
    """

    return data.sample(frac=1).reset_index(drop=True)


def test_train_split(
    X: DataFrame, y: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits the data into training and testing data.

    Parameters:
    - X (DataFrame): The input features.
    - y (DataFrame): The target variable.

    Returns:
    - X_train (DataFrame): The training data features.
    - X_test (DataFrame): The testing data features.
    - y_train (DataFrame): The training data target variable.
    - y_test (DataFrame): The testing data target variable.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def knn(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
    k: int = 3,
) -> tuple[KNeighborsClassifier, np.ndarray, float, float]:
    """
    Performs k-nearest neighbors classification on the given training and test data.

    Parameters:
        X_train (DataFrame): The feature matrix of the training data.
        y_train (DataFrame): The target values of the training data.
        X_test (DataFrame): The feature matrix of the test data.
        y_test (DataFrame): The target values of the test data.
        k (int, optional): The number of neighbors to consider. Defaults to 3.

    Returns:
        tuple[KNeighborsClassifier, np.ndarray, float, float]: A tuple containing the knn classifier, test predictions, train accuracy, and test accuracy.
    """

    model = KNeighborsClassifier(n_neighbors=k)

    print(
        f"{Fore.RED}X_train dim: {X_train.shape}, y_train dim: {y_train.shape}{Fore.WHITE}"
    )

    feature_names = ["X1", "X2"]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    model.fit(X_train_df, y_train)

    # Get results
    train_accuracy = model.score(X_train, y_train)

    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return model, test_predictions, train_accuracy, test_accuracy


def plot_decision_regions(
    X_test: DataFrame,
    test_preds: np.ndarray,
    classifier: KNeighborsClassifier,
    resolution=0.02,
) -> None:
    """
    Plots the decision regions for a classifier.

    Parameters:
    - X_test (DataFrame): The input data used for testing the classifier.
    - test_preds (ndarray): The predicted labels for the test data.
    - classifier: The trained knn classifier object.
    - resolution (float): The step size of the mesh grid used for plotting the decision regions. Default is 0.02.

    Returns:
    - None

    This function plots the decision regions for a classifier by creating a mesh grid based on the input data and
    classifying each point in the grid. The decision regions are then visualized using a contour plot.

    Note:
    - The input data should have exactly two features for proper visualization.
    - The classifier should have a `predict` method that takes a feature matrix as input and returns the predicted labels.

    Example usage:
    X_test = ...
    test_preds = ...
    classifier = ...
    plot_decision_regions(X_test, test_preds, classifier)
    """

    k: int = classifier.n_neighbors

    X1_test = X_test.iloc[:, 0]
    X2_test = X_test.iloc[:, 1]

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

    Z_preds = classifier.predict(Z)
    print("Before reshape Z_pred:", Z_preds.shape)
    Z_preds = Z_preds.reshape(xx.shape)
    print(Z_preds.shape)

    # Plot the decision boundary
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    plt.pcolormesh(xx, yy, Z_preds, cmap=cmap_light, shading="auto")

    # Overlay the test points
    cmap_bold = ListedColormap(["#FF0000", "#00FF00"])
    plt.scatter(X1_test, X2_test, c=test_preds, cmap=cmap_bold)

    # Setup plot
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("k-NN decision regions (k = %d)" % k)
    plt.show()
