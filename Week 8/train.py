"""
Training script for the CNN model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style


def train_iteration():
    pass


def print_title(title: str):
    """
    Print the title of the section
    """

    print(f"{Fore.LIGHTBLUE_EX}{title}{Style.RESET_ALL}")


def train(
    device: torch.device,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
):
    """
    Training the model
    """

    for epoch in range(num_epochs):

        # Extract a mini-batch
        for i, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}")
        ):
            print_title("Moving data to device")
            # Move the data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            print_title("Forward pass")
            outputs = model(images)

            # Calculate the loss
            print_title("Calculate the loss")
            loss = criterion(outputs, labels)

            # Backward pass
            print_title("Backward pass")
            optimizer.zero_grad()

            # Compute the gradient
            loss.backward()

            # Step the optimizer to update the weights
            optimizer.step()

        # Validation
        print_title("Validation")
        model.eval()

        # Softmax to recover probabilities from logits
        # if using cross entropy loss, this step is is built-in by pytorch

        val_accuracy = correct / total

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy*100:.2f}%"
        )
