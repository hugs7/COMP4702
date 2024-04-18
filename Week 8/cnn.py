"""
Week 8 Practical
Driver File
18/04/2024
"""

from welcome import welcome

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os


def main():
    welcome()

    # Transform and Pre-processing the Dataset

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalise the pixel values
        ]
    )

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    data_folder = os.path.join(current_directory, "data")
    mnist_data_folder = os.path.join(data_folder, "MNIST")

    is_download = not os.path.exists(mnist_data_folder)

    # Load the MNIST dataset
    mnist_train = datasets.MNIST(
        data_folder, train=True, download=is_download, transform=transform
    )
    mnist_test = datasets.MNIST(
        data_folder, train=False, download=is_download, transform=transform
    )

    # Split the dataset into training and validation
    train_ratio = 0.8
    train_size = int(train_ratio * len(mnist_train))
    val_size = len(mnist_train) - train_size

    mnist_train, mnist_val = random_split(mnist_train, [train_size, val_size])

    # Define data loaders
    batch_size = 64

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    main()
