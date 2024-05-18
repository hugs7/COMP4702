"""
Helper for printing messages in color
"""

from colorama import Fore, Style
from typing import Any
from utils import key_from_value

LOG_LEVEL_COLOUR = {
    "ERROR": Fore.RED,
    "WARNING": Fore.YELLOW,
    "DEBUG": Fore.LIGHTMAGENTA_EX,
    "INFO": Fore.LIGHTCYAN_EX,
    "TRACE": Fore.LIGHTBLACK_EX
}


def check_log_level(level: str) -> bool:
    """
    Check if the log level is at least the specified level.

    Args:
    - level (str): The level to check against as a string.

    Returns:
    - bool: True if the log level is at least the specified level.
    """
    from check_log_level import LOG_LEVEL_INDEX, LOG_LEVELS

    # Find the key for the level from the values
    level = key_from_value(LOG_LEVELS, level)

    if LOG_LEVEL_INDEX is None:
        raise ValueError(
            "LOG_LEVEL_INDEX not found. Ensure you have called 'set_log_level()' before using the logger.")

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
        colour = LOG_LEVEL_COLOUR[level]
        log_colored(colour, *messages, **kwargs)


def log_error(*messages: str, **kwargs: Any) -> None:
    """
    Prints an error message in red text.
    """
    if check_log_level("ERROR"):
        colour = LOG_LEVEL_COLOUR["ERROR"]
        log_colored(colour, *messages, **kwargs)


def log_warning(*messages: str, **kwargs: Any) -> None:
    """
    Prints a warning message in yellow text.
    """
    if check_log_level("WARNING"):
        colour = LOG_LEVEL_COLOUR["WARNING"]
        log_colored(colour, *messages, **kwargs)


def log_debug(*messages: str, **kwargs: Any) -> None:
    """
    Prints a debug message in magenta text.
    """
    if check_log_level("DEBUG"):
        colour = LOG_LEVEL_COLOUR["DEBUG"]
        log_colored(colour, *messages, **kwargs)


def log_info(*messages: str, **kwargs: Any) -> None:
    """
    Prints an info message in blue text.
    """
    if check_log_level("INFO"):
        colour = LOG_LEVEL_COLOUR["INFO"]
        log_colored(colour, *messages, **kwargs)


def log_trace(*messages: str, **kwargs: Any) -> None:
    """
    Prints a trace message in cyan text.
    """
    if check_log_level("TRACE"):
        colour = LOG_LEVEL_COLOUR["TRACE"]
        log_colored(colour, *messages, **kwargs)


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
