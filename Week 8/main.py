"""
Week 8 Practical
Driver File
18/04/2024
"""

from welcome import welcome
import cnn
import train

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

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---

    learning_rate = 1e-4
    num_epochs = 10

    # --- CNN Model ---

    model = cnn.CNN(
        num_conv_layers=5,
        num_in_channels=1,
        hidden_channels=[32, 64, 64, 64, 32],
        num_out_channels=10,
        kernel_sizes=[4, 4, 4, 4, 4],
        pooling_kernel_sizes=[2, 2, 2, 2, 2],
        stride=1,
        padding=2,
        num_fc_layers=2,
        num_classes=10,
    )

    # Test the model

    print(model)
    print("---")
    print(model.parameters)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    train.train(
        device, model, train_loader, val_loader, optimiser, criterion, num_epochs
    )


if __name__ == "__main__":
    main()
