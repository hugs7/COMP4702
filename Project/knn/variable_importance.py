"""
Helper function to compute the importance of features in the dataset
"""

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
    # Compute distances from the sample to its k-nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    _, neighb_indices = neigh.kneighbors(X[sample_idx].reshape(1, -1))
    neighb_dist = np.mean(neigh.kneighbors()[0], axis=1)

    # Compute distances from random points to their k-nearest neighbors
    rand_indices = np.random.choice(X.shape[0], size=num_random_points, replace=False)
    rand_dist = np.zeros((num_random_points, X.shape[1]))
    for i, rand_idx in enumerate(rand_indices):
        log_trace(f"Random index: {rand_idx}")
        _, rand_indices_neighb = neigh.kneighbors(X[rand_idx].reshape(1, -1))
        rand_indices_neighb = rand_indices_neighb[0]

        rand_neigh_dists = []
        for rand_neighb_idx in rand_indices_neighb:
            dist = np.linalg.norm(X[rand_idx] - X[rand_neighb_idx])
            rand_neigh_dists.append(dist)
        log_trace(f"Random indices neighbors: {rand_indices_neighb} with distances: {rand_neigh_dists}")
        rand_dist[i] = np.mean(rand_neigh_dists)

    # Compute feature importance as the ratio of neighb_dist to rand_dist
    feature_importance = np.mean(neighb_dist, axis=0) / np.mean(rand_dist, axis=0)
    log_debug(f"Feature importance: {feature_importance}")

    return feature_importance


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
    feature_importance_sum = np.zeros(X.shape[1])
    for sample_idx in sample_indices:
        log_debug(f"Computing feature importance for sample {sample_idx}")
        feature_importance_sum += compute_feature_importance(X, sample_idx, k, num_random_points)

    # Average feature importance across samples
    avg_feature_importance = feature_importance_sum / num_samples

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
