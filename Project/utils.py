"""
This file contains utility functions that are used throughout the project.
"""

from typing import Dict


def key_from_value(dict: Dict, value: str) -> int:
    """
    Get the key from a dictionary by the value. IF multiple
    keys have the same value, the first key found will be returned.

    Args:
    - dict (Dict): The dictionary to search.
    - value (str): The value to search for.

    Returns:
    - int: The key of the value.
    """
    return [k for k, v in dict.items() if v == value][0]
