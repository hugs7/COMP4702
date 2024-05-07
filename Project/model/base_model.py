import numpy as np


class Model:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_labels: list[str],
        y_labels: list[list[str]],
    ):
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.X_labels = X_labels
        self.y_labels = y_labels

    def get_X_train(self) -> np.ndarray:
        return self.X_train

    def get_y_train(self) -> np.ndarray:
        return self.y_train

    def get_X_test(self) -> np.ndarray:
        return self.X_test

    def get_y_test(self) -> np.ndarray:
        return self.y_test

    def get_X_labels(self) -> list[str]:
        return self.X_labels

    def get_y_labels(self) -> list[list[str]]:
        return self.y_labels
