"""
Helper function to compute the importance of features in the dataset
"""

from typing import Union
from sklearn.neighbors import NearestNeighbors
import numpy as np

from logger import *


def compute_feature_importance(X: np.ndarray, sample_idx: int, k: int, num_random_points: int = 100) -> np.ndarray:
    """
    Compute feature importance for a given sample using KNN distances.

    Parameters:
        X (numpy array): Input data matrix (features).
        sample_idx (int): Index of the sample for which to compute feature importance.
        k_neighbors (int): Number of nearest neighbors to consider.
        num_random_points (int): Number of random points to select for comparison. Default is 100.

    Returns:
        feature_importance (numpy array): Feature importance scores.
    """

    def point_get_nearest_neighbors(neigh: NearestNeighbors, sample_point: np.ndarray) -> np.ndarray:
        """
        Get the indices of the k-nearest neighbors of a sample in the dataset.

        Parameters:
            NearestNeighbors (NearestNeighbors): A fitted nearestNeighbors model object.
            sample_point (numpy array): The sample for which to find the nearest neighbors.

        Returns:
            neighb_indices (numpy array): Indices of the k-nearest neighbors of the sample.
        """

        _, neighb_indices = neigh.kneighbors(sample_point.reshape(1, -1))
        neighb_indices = neighb_indices[0]

        # Remove the sample itself from the neighbors
        neighb_indices = neighb_indices[neighb_indices != sample_idx]

        return neighb_indices

    def point_to_point_dist(x1: np.ndarray, x2: np.ndarray, feature_index: Union[int, None]) -> float:
        """
        Compute the Euclidean distance between two points.

        Parameters:
            x1 (numpy array): First point.
            x2 (numpy array): Second point.
            feature_index (Union[int, None]): Index of the feature to compute distances for. If none, compute the distance between all features.

        Returns:
            distance (float): Euclidean distance between the two points.
        """

        if feature_index:
            x1_feature = x1[feature_index]
            x2_feature = x2[feature_index]

            return np.abs(x1_feature - x2_feature)
        else:
            return np.linalg.norm(x1 - x2)

    def point_to_points_distances(X: np.ndarray, point: np.ndarray, points: np.ndarray, feature_index: int) -> np.ndarray:
        """
        Compute the distances between a point and its neighbors.

        Parameters:
            X (numpy array): Input data matrix (features).
            point (numpy array): The point for which to compute distances.
            points (numpy array): The points to which to compute distances.
            feature_index (int): The index of the feature to compute distances for.

        Returns:
            distances (numpy array): Distances between the point and points.
        """
        distances = np.zeros(len(points))
        for i, comparison_point in enumerate(points):
            distances[i] = point_to_point_dist(point, comparison_point, feature_index)

        return distances

    def indices_to_points(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Get the points in the dataset corresponding to a given set of indices.

        Parameters:
            X (numpy array): Input data matrix (features).
            indices (numpy array): Indices of the points to retrieve.

        Returns:
            points (numpy array): The points in the dataset corresponding to the given indices.
        """
        return X[indices]

    # Compute distances from the sample to its k-nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)

    sample_point = X[sample_idx]

    neighb_indices = point_get_nearest_neighbors(neigh, sample_point)
    neighb_points = indices_to_points(X, neighb_indices)
    # Compute distances from random points to their k-nearest neighbors
    rand_indices = np.random.choice(X.shape[0], size=num_random_points, replace=False)
    rand_points = indices_to_points(X, rand_indices)

    num_features = X.shape[1]
    feature_importance_coefs = np.zeros(num_features)
    for i, feature_index in enumerate(range(num_features)):
        log_info(f"Computing feature importance for feature {feature_index}...")

        neighb_dists = point_to_points_distances(X, sample_point, neighb_points, feature_index)
        rand_dists = point_to_points_distances(X, sample_point, rand_points, feature_index)

        log_trace(f"Nearest neighbors distances: {neighb_dists}")
        log_trace(f"Random neighbors distances: {rand_dists}")

        neighb_dist_avg = np.mean(neighb_dists, axis=0)
        rand_dist_avg = np.mean(rand_dists, axis=0)

        log_debug(f"Nearest neighbors distances: {neighb_dist_avg}")
        log_debug(f"Random neighbors distances: {rand_dist_avg}")

        if neighb_dist_avg == 0 or rand_dist_avg == 0:
            ratio = 0
        else:
            ratio = neighb_dist_avg / rand_dist_avg

        log_debug(f"Feature importance: {ratio}")

        feature_importance_coefs[i] = ratio

    log_info(f"Feature importance coefficients: {feature_importance_coefs}")

    return feature_importance_coefs


def compute_average_feature_importance(X: np.ndarray, num_samples: int, k: int, num_random_points: int = 100) -> np.ndarray:
    """
    Compute average feature importance across multiple samples using KNN distances.

    Parameters:
        X (numpy array): Input data matrix (features).
        num_samples (int): Number of samples to analyze.
        k_neighbors (int): Number of nearest neighbors to consider.
        num_random_points (int): Number of random points to select for comparison. Default is 100.

    Returns:
        avg_feature_importance (numpy array): Average feature importance scores.
    """
    # Randomly select samples from the data
    sample_indices = np.random.choice(X.shape[0], size=num_samples, replace=False)
    log_debug(f"Selected sample indices: {sample_indices}")

    # Compute feature importance for each sample
    num_features = X.shape[1]
    feature_importances = np.zeros((num_samples, num_features))
    for i, sample_idx in enumerate(sample_indices):
        log_info(f"Computing feature importance for sample {sample_idx}. Progress: {i} / {num_samples}...")
        feature_importances[i] = compute_feature_importance(X, sample_idx, k, num_random_points)

    # Average feature importance across samples
    log_debug(f"Feature importances: {feature_importances}")
    avg_feature_importance = np.mean(feature_importances, axis=0)
    log_info(f"Average feature importance: {avg_feature_importance}")

    return avg_feature_importance


def main():
    log_title("Example usage of compute_feature_importance function")
    # Example usage:
    # Assuming you have your data X and a sample index sample_idx
    sample_idx = 0  # Index of the sample for which to compute feature importance
    X = np.random.rand(100, 5)  # Example data matrix (100 samples, 5 features)

    feature_importance = compute_feature_importance(X, sample_idx)
    print("Feature Importance:", feature_importance)

    log_title("Example usage of compute_average_feature_importance function")
    # Example usage:
    # Assuming you have your data X
    X = np.random.rand(100, 5)  # Example data matrix (100 samples, 5 features)

    # Compute average feature importance across multiple samples
    avg_feature_importance = compute_average_feature_importance(X, num_samples=10)
    print("Average Feature Importance:", avg_feature_importance)


if __name__ == "__main__":
    main()
