"""
Main file for week 7 prac
11/04/2024
"""

import torch
import sys


def main():
    print("Week 7: PyTorch")

    print("Torch version: ", torch.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available")
        print("Device count: ", torch.cuda.device_count())
        print("Current device: ", torch.cuda.current_device())
        print("Device name: ", torch.cuda.get_device_name())
    else:
        print("CUDA is not available")

    print("Python version: ", sys.version)


if __name__ == "__main__":
    main()
