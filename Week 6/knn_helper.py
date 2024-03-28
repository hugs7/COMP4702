from colorama import Fore, Style

from pandas import DataFrame
import knn


def knn_classify(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame,
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


def k_fold_split(data, num_folds, fold):
    """
    Splits the data into training and testing data for a given fold.
    """

    # Split the data into training and testing data
    fold_size = len(data) // num_folds
    test_start = fold_size * fold
    test_end = test_start + fold_size

    X_test = data.iloc[test_start:test_end, :-1]
    y_test = data.iloc[test_start:test_end, -1]

    X_train = data.drop(data.index[test_start:test_end]).iloc[:, :-1]
    y_train = data.drop(data.index[test_start:test_end]).iloc[:, -1]

    return X_train, y_train, X_test, y_test
