from pandas import DataFrame


class Model:
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        feature_names: list[str],
    ):
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.feature_names = feature_names

    def get_X_train(self) -> DataFrame:
        return self.X_train

    def get_y_train(self) -> DataFrame:
        return self.y_train

    def get_X_test(self) -> DataFrame:
        return self.X_test

    def get_y_test(self) -> DataFrame:
        return self.y_test

    def get_feature_names(self) -> list[str]:
        return self.feature_names
