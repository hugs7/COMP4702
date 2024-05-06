"""
Helper for printing messages in color
"""

from colorama import Fore, Style
from typing import Any, Dict, Sequence

LOG_LEVELS = {0: None, 1: "ERROR", 2: "WARNING", 3: "DEBUG", 4: "INFO", 5: "TRACE"}

LOG_LEVEL = LOG_LEVELS[5]


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


def check_log_level(level: str) -> bool:
    """
    Check if the log level is at least the specified level.

    Args:
    - level (str): The level to check against as a string.

    Returns:
    - bool: True if the log level is at least the specified level.
    """
    # Find the key for the level from the values
    level = key_from_value(LOG_LEVELS, level)

    if LOG_LEVEL is None:
        raise ValueError("LOG_LEVEL not found")

    return LOG_LEVEL <= level


def log_title(*messages: str) -> None:
    """
    Prints a title message in green text.
    """
    log_colored(Fore.GREEN, *messages)


def log_error(*messages: str) -> None:
    """
    Prints an error message in red text.
    """
    if check_log_level("ERROR"):
        log_colored(Fore.RED, *messages)


def log_warning(*messages: str) -> None:
    """
    Prints a warning message in yellow text.
    """
    if check_log_level("WARNING"):
        log_colored(Fore.YELLOW, *messages)


def log_debug(*messages: str) -> None:
    """
    Prints a debug message in magenta text.
    """
    if check_log_level("DEBUG"):
        log_colored(Fore.MAGENTA, *messages)


def log_info(*messages: str) -> None:
    """
    Prints an info message in blue text.
    """
    if check_log_level("INFO"):
        log_colored(Fore.LIGHTBLUE_EX, *messages)


def log_trace(*messages: str) -> None:
    """
    Prints a trace message in cyan text.
    """
    if check_log_level("TRACE"):
        log_colored(Fore.LIGHTBLACK_EX, *messages)


def log_colored(color: Any, *messages: str) -> None:
    """
    Prints messages in the specified color.
    """
    if not messages:
        return

    print(f"{color}{' '.join(map(str, messages))}{Style.RESET_ALL}")
