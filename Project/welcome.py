import torch
import sys
import numpy as np
import seaborn as sb
from colorama import Fore, Style

from logger import *


def welcome():
    print("Week 7: PyTorch")

    print("Torch version: ", torch.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"{Fore.LIGHTGREEN_EX}CUDA is available{Style.RESET_ALL}")
        print(f"{Fore.LIGHTMAGENTA_EX}Device count: {Style.RESET_ALL}{torch.cuda.device_count()}")
        print(f"{Fore.LIGHTMAGENTA_EX}Current device: {Style.RESET_ALL}{torch.cuda.current_device()}")
        print(f"{Fore.LIGHTMAGENTA_EX}Device name: {Style.RESET_ALL}{torch.cuda.get_device_name()}")
    else:
        print(f"{Fore.LIGHTRED_EX}CUDA is not available{Style.RESET_ALL}")

    print(f"{Fore.LIGHTCYAN_EX}Python version: {Style.RESET_ALL}{sys.version}")
    print(f"{Fore.LIGHTCYAN_EX}Numpy version: {Style.RESET_ALL}{np.__version__}")
    print(f"{Fore.LIGHTCYAN_EX}Seaborn version: {Style.RESET_ALL}{sb.__version__}")

    log_line()


def available_items(collective_name: str, items: list[str]) -> None:
    print(f"{Fore.LIGHTGREEN_EX}Available {collective_name}:{Style.RESET_ALL}")
    for item in items:
        print(f"  - {item}")
    print()
