"""
Helper for getting the log level from the .env file.
This function should be called first in the main driver file.
"""

import os
from colorama import Fore, Style

from utils import key_from_value

LOG_LEVELS = {0: None, 1: "ERROR", 2: "WARNING",
              3: "INFO", 4: "DEBUG", 5: "TRACE"}

LOG_LEVEL = None
LOG_LEVEL_INDEX = None
DEFAULT_LOG_LEVEL = "INFO"


def make_env_file(env_file_path: str, initial_content: str = "") -> None:
    """
    Creates the .env file with the default log level set to INFO.

    Args:
    - env_file_path (str): The path of the .env file to be created.
    - initial_content (str): The initial content of the .env file.
    """

    with open(env_file_path, 'w') as environment_file:
        environment_file.write(initial_content)


def read_env_file(env_file_path: str) -> str:
    """
    Reads the .env file. Creates the env file if it does not exist.

    Args:
    - env_file_path (str): The path of the .env file.

    Returns:
    - str: The contents of the .env file.
    """

    # Check if the .env file exists
    if not os.path.exists(env_file_path):
        make_env_file(env_file_path)

    # Read the .env file
    with open(env_file_path, "r") as environment_file:
        return environment_file.readlines()


def append_env_file(env_file_path: str, line: str) -> None:
    """
    Appends a line to the .env file.

    Args:
    - env_file_path (str): The path of the .env file.
    - line (str): The line to append to the .env file.
    """

    with open(env_file_path, 'a') as environment_file:
        environment_file.write(line)


def get_log_level(env_contents: str) -> tuple[str, int]:
    for line in env_contents:
        print("Log level found", line)
        if line.startswith('LOG_LEVEL'):
            log_level = line.strip().split('=')[1].strip()
            log_level_index = key_from_value(LOG_LEVELS, log_level)

            return log_level, log_level_index

    return None, None


def set_log_level():
    global LOG_LEVEL, LOG_LEVEL_INDEX
    # Read from .env file
    current_dir = os.path.dirname(__file__)
    env_file_path = os.path.join(current_dir, ".env")
    env_contents = read_env_file(env_file_path)

    LOG_LEVEL, LOG_LEVEL_INDEX = get_log_level(env_contents)

    if LOG_LEVEL is None:
        print(
            f"{Fore.LIGHTYELLOW_EX}LOG_LEVEL not defined in .env. Setting to INFO.{Style.RESET_ALL}")
        log_level_env = f"LOG_LEVEL={DEFAULT_LOG_LEVEL}\n"
        append_env_file(env_file_path, log_level_env)

        # Read from .env file again
        env_contents = read_env_file(env_file_path)
        LOG_LEVEL, LOG_LEVEL_INDEX = get_log_level(env_contents)
