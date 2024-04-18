"""
Utils helper script
"""

from typing import List


def prod(x: List[int]) -> int:
    """
    Computes the product of the list of numbers

    Args:
        x: List of integers

    Returns:
        product: int
    """

    product = 1
    for num in x:
        product *= num

    return product
