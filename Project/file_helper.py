"""
Helper functions for file operations
Hugo Burton
"""

import os
from logger import *


def make_folder_if_not_exists(folder_path: str) -> None:
    """
    Creates a folder if it does not exist.

    Args:
    - folder_path (str): The path of the folder to create.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        log_info(f"Created folder: {folder_path}")
    else:
        log_info(f"Folder already exists: {folder_path}")


def file_exists(file_path: str) -> bool:
    """
    Checks if a file exists.

    Args:
    - file_path (str): The path of the file to check.

    Returns:
    - bool: True if the file exists.
    """

    return os.path.exists(file_path)


def remove_file_if_exist(file_path: str, warn_if_not_exist: bool = False) -> None:
    """
    Removes a file.

    Args:
    - file_path (str): The path of the file to remove.
    - warn_if_not_exist (bool): Whether to log a warning if the file does not exist. Default is False.
    """

    # Check if the file exists
    if not file_exists(file_path):
        if warn_if_not_exist:
            log_warning(f"File {file_path} does not exist")
        return

    os.remove(file_path)
    log_info(f"Removed file: {file_path}")
