import torch
import sys
import numpy as np
import seaborn as sb
from colorama import Fore, Style

from logger import *


def welcome():
    log_title("Welcome to the Machine Learning Project")

    log_info("Torch version: ", torch.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        log_info("CUDA is available")

        log_info("Device count: ", end="")
        log(torch.cuda.device_count(), level="INFO")

        log_info("Current device: ", end="")
        log(torch.cuda.current_device(), level="INFO")

        log_info("Device name: ", end="")
        log(torch.cuda.get_device_name(), level="INFO")
    else:
        log_warning("CUDA is not available")

    log_line()

    log_info("Python version: ", end="")
    log(sys.version, level="INFO")

    log_info("Numpy version: ", end="")
    log(np.__version__, level="INFO")

    log_info("Seaborn version: ", end="")
    log(sb.__version__, level="INFO")

    log_line()


def available_items(collective_name: str, items: list[str]) -> None:
    print(f"{Fore.LIGHTGREEN_EX}Available {collective_name}:{Style.RESET_ALL}")
    for item in items:
        print(f"  - {item}")
    print()
