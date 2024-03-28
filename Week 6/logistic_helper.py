
from pandas import DataFrame
import logistic


def logistic_fit(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame,
                 y_test: DataFrame, feature_names: list[str], classes: list[str], threshold: float = 0.5) \
        -> DataFrame:
    """
    Classification task using logistic regression
    """

    # Apply the logistic classifier
    logistic_classifier = logistic.LogisticRegressionModel(
        X_train, y_train, X_test, y_test, feature_names
    )

    logistic_classifier.fit()

    # Get results
    test_predictions, train_accuracy, test_accuracy = logistic_classifier.predict_proba(classes,
                                                                                        threshold=threshold)

    return test_predictions, train_accuracy, test_accuracy
