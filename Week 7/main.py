"""
Main file for week 7 prac
11/04/2024
"""

import torch
import sys
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


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

    # --- Define a Model ---

    # Activation function

    sb.set_context("talk")
    sb.set_style("dark")

    geLu = torch.nn.GELU()

    input = torch.arange(-6, 6, step=0.1)
    output = geLu(input)

    sb.lineplot(x=input, y=output)

    plt.xlabel("X")
    plt.ylabel("GeLU(X)")
    plt.show()


if __name__ == "__main__":
    main()
