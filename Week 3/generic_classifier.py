from pandas import DataFrame


class Classifier:
    def __init__(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
        k: int = 3,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.feature_names = feature_names

        self.k = k

    def get_X_train(self) -> DataFrame:
        return self.X_train

    def get_X_test(self) -> DataFrame:
        return self.X_test

    def get_y_train(self) -> DataFrame:
        return self.y_train

    def get_y_test(self) -> DataFrame:
        return self.y_test

    def get_feature_names(self) -> list[str]:
        return self.feature_names

    def get_k(self) -> int:
        return self.k
