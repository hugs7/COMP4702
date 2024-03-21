from colorama import Fore, Style

import knn


def print_classify_results(
    train_accuracy: float, test_accuracy: float, show_mcr: bool
) -> None:
    print(f"{Fore.LIGHTMAGENTA_EX}Training accuracy: {Style.RESET_ALL}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing accuracy: {Style.RESET_ALL}", test_accuracy)

    if show_mcr:
        print("")
        print(
            f"{Fore.LIGHTMAGENTA_EX}Training MCR: {Style.RESET_ALL}", 1 - train_accuracy
        )
        print(
            f"{Fore.LIGHTMAGENTA_EX}Testing MCR: {Style.RESET_ALL}", 1 - test_accuracy
        )


def knn_classify(
    X_train, y_train, X_test, y_test, feature_names, plot: bool = False
) -> tuple[float, float]:
    """
    Classification task using knn
    """

    # Apply the knn classifier
    knn_classifier = knn.KNNClassify(
        X_train, y_train, X_test, y_test, feature_names, k=3
    )

    test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

    print_classify_results(train_accuracy, test_accuracy, False)

    if plot:
        # Plot decision regions
        knn_plot_title = f"k-NN decision regions (k = {knn_classifier.get_k()})"
        knn_classifier.plot_decision_regions(
            test_preds, resolution=0.02, plot_title=knn_plot_title
        )

    return train_accuracy, test_accuracy


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
