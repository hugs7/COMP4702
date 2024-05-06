"""
Helper for printing messages in color
"""

from colorama import Fore, Style
from typing import Any, Sequence


def print_title(*messages: str) -> None:
    """
    Prints a title message in green text.
    """
    print_colored(Fore.GREEN, *messages)


def print_error(*messages: str) -> None:
    """
    Prints an error message in red text.
    """
    print_colored(Fore.RED, *messages)


def print_warning(*messages: str) -> None:
    """
    Prints a warning message in yellow text.
    """
    print_colored(Fore.YELLOW, *messages)


def print_debug(*messages: str) -> None:
    """
    Prints a debug message in magenta text.
    """
    print_colored(Fore.MAGENTA, *messages)


def print_info(*messages: str) -> None:
    """
    Prints an info message in blue text.
    """
    print_colored(Fore.LIGHTBLUE_EX, *messages)


def print_trace(*messages: str) -> None:
    """
    Prints a trace message in cyan text.
    """
    print_colored(Fore.LIGHTBLACK_EX, *messages)


def print_colored(color: Any, *messages: str) -> None:
    """
    Prints messages in the specified color.
    """
    if not messages:
        return

    print(f"{color}{' '.join(map(str, messages))}{Style.RESET_ALL}")
