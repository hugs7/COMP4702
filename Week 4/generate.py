"""
Defined function generate a training point
"""

import numpy as np


def generate_training_point(f: callable, noise: float = 0.2) -> float:
    """
    Generates a single training point with noise from the function f
    """

    # Generate a random x value
    x = np.random.uniform(-1, 1)

    # Generate a random noise value using a gaussian distribution
    noise = np.random.normal(0, noise)

    # Generate the y value
    y = f(x) + noise

    return x, y
