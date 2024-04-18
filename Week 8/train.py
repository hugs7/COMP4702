"""
Training script for the CNN model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_iteration():
    pass


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
            # Move the data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()

        # Validation
        model.eval()

        val_accuracy = correct / total

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy*100:.2f}%"
        )
