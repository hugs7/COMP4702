"""
Helper for printing messages in color
"""

from colorama import Fore, Style
from typing import Any
from utils import key_from_value

LOG_LEVELS = {0: None, 1: "ERROR", 2: "WARNING", 3: "INFO", 4: "DEBUG", 5: "TRACE"}

LOG_LEVEL = "DEBUG"
LOG_LEVEL_INDEX = key_from_value(LOG_LEVELS, LOG_LEVEL)


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

    if LOG_LEVEL_INDEX is None:
        raise ValueError("LOG_LEVEL not found")

    return level <= LOG_LEVEL_INDEX


def log_title(*messages: str, **kwargs: Any) -> None:
    """
    Prints a title message in green text.
    """
    log_colored(Fore.LIGHTGREEN_EX, *messages, **kwargs)


def log(*messages: str, level: str, **kwargs: Any) -> None:
    """
    Prints a message in white text.
    """
    if check_log_level(level):
        log_colored(Fore.WHITE, *messages, **kwargs)


def log_error(*messages: str, **kwargs: Any) -> None:
    """
    Prints an error message in red text.
    """
    if check_log_level("ERROR"):
        log_colored(Fore.RED, *messages, **kwargs)


def log_warning(*messages: str, **kwargs: Any) -> None:
    """
    Prints a warning message in yellow text.
    """
    if check_log_level("WARNING"):
        log_colored(Fore.YELLOW, *messages, **kwargs)


def log_debug(*messages: str, **kwargs: Any) -> None:
    """
    Prints a debug message in magenta text.
    """
    if check_log_level("DEBUG"):
        log_colored(Fore.LIGHTMAGENTA_EX, *messages, **kwargs)


def log_info(*messages: str, **kwargs: Any) -> None:
    """
    Prints an info message in blue text.
    """
    if check_log_level("INFO"):
        log_colored(Fore.LIGHTCYAN_EX, *messages, **kwargs)


def log_trace(*messages: str, **kwargs: Any) -> None:
    """
    Prints a trace message in cyan text.
    """
    if check_log_level("TRACE"):
        log_colored(Fore.LIGHTBLACK_EX, *messages, **kwargs)


def log_colored(color: Any, *messages: str, **kwargs: Any) -> None:
    """
    Prints messages in the specified color.
    """
    if not messages:
        return

    end = kwargs.get("end", "\n")
    sep = kwargs.get("sep", " ")

    print(color, end="")
    print(*messages, sep=sep, end=end)
    print(Style.RESET_ALL, end="")


def log_line(num_hyphens: int = 50, level: str = "ERROR") -> None:
    """
    Prints a line of '-' characters.

    Args:
    - num_hyphens (int): The number of hyphens to print. Default is 50.
    - level (str): The log level of the line. Default is "ERROR" (always print)
    """

    if not check_log_level(level):
        return

    print("-" * num_hyphens)
