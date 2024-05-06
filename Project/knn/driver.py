""""
Driver script for knn classification model
Hugo Burton
06/05/2024
"""

from pandas import DataFrame

import knn.knn


def run_knn_model(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame,
                  y_test: DataFrame, feature_names: list[str], k: int = 5) \
        -> tuple[float, float]:
    """
    Classification task using knn
    """

    # Apply the knn classifier
    knn_classifier = knn.KNNClassify(
        X_train, y_train, X_test, y_test, feature_names, k=k
    )

    test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

    return test_preds, train_accuracy, test_accuracy
