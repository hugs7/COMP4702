"""
Classification Model Class
Hugo Burton
"""

from typing import List
import numpy as np
from sklearn.base import BaseEstimator

from logger import *

from model.base_model import Model


TRAINING = "training"
VALIDATION = "validation"
TESTING = "testing"

DATA_TYPES = {TRAINING, VALIDATION, TESTING}


def check_data_type(data_type: str) -> int:
    """
    Checks if the data type is valid

    Args:
    - data_type (str): The data type

    Returns:
    - int: Error code for invalid data type. 0 if valid.
    """

    if data_type not in DATA_TYPES:
        log_error(f"Invalid data type. Must be one of: {' '.join(DATA_TYPES)}")
        return 1

    return 0


class Classifier(Model):
    def __init__(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_labels: list[str],
        y_labels: list[list[str]],
    ) -> None:
        super().__init__(X_train, y_train, X_test, y_test, X_labels, y_labels)

        self.model = model
