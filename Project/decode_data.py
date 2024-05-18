"""
Helper script to decode data. Primarily decoding one-hot encoded data.
"""

from typing import Tuple
import numpy as np

from check_log_level import set_log_level
from logger import *
import utils


def decode_one_hot_encoded_data(y: np.ndarray) -> Tuple[np.ndarray]:
    """
    Decodes the one-hot encoded target variable.

    Parameters:
    - y (ndarray): The one-hot encoded data. Should take shape [j, num_vars, k] for some j, k > 0.

    Returns:
    - Tuple[np.ndarray]: The decoded data in the form (y_decoded_1, y_decoded_2, ... y_decoded_num_vars) where each y_decoded_i
                        is the decoded data for the i-th variable of shape [j]. The decoded data is not one-hot encoded. It is
                        the index of the class with the highest probability.
    """

    log_trace("Decoding one-hot encoded data...")

    # First we need to understand in axis 2 (k) this data may be padded with zeros to the maximum number of classes in any var.
    # This is done becasue np.ndarray requires all dimensions to be the same size.
    # We need to remove these padded zeros to decode the data.

    # Move y to the CPU and convert to numpy array
    y = utils.tensor_to_cpu(y, detach=True)

    # Find the indices of the non-zero elements along the last axis
    decoded_data = np.argmax(y, axis=-1)

    # Split the decoded data along the second axis to separate each variable
    return tuple(decoded_data[:, i] for i in range(decoded_data.shape[1]))


def test_decode_one_hot_encoded_data():
    # Example usage
    y = np.array([
        [[0, 1, 0], [1, 0, 0]],  # Sample 1
        [[1, 0, 0], [0, 0, 1]],  # Sample 2
        [[0, 1, 0], [0, 1, 0]]   # Sample 3
    ])

    decoded_vars = decode_one_hot_encoded_data(y)
    for i, decoded in enumerate(decoded_vars):
        print(f"Decoded variable {i+1}: {decoded}")


if __name__ == "__main__":
    set_log_level()
    test_decode_one_hot_encoded_data()
