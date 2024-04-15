
import torch
import sys
import numpy as np
import seaborn as sb


def welcome():
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
    print("Numpy version: ", np.__version__)
    print("Seaborn version: ", sb.__version__)

    print("-" * 40)
