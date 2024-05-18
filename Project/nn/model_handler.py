"""
Helper for reading and saving model files
Hugo Burton
"""

import os
import torch
from typing import List

from logger import *


def read_model():
    pass


def save_model(folder_path: str, model: torch.nn.Module, model_name: str, train_losses: List[float], validation_losses: List[float], checkpoint_num: int = None):
    """
    Handles saving the model to a file.
    """

    is_checkpoint = checkpoint_num is not None

    model_obj = {
        "model_state_dict": model.state_dict(),
        "train_loss": train_losses,
        "validation_loss": validation_losses,
        "is_cuda": torch.cuda.is_available(),
        "model_name": model_name,
        "is_checkpoint": is_checkpoint,
        "checkpoint_num": checkpoint_num
    }

    if is_checkpoint:
        save_file_name = f"{model_name}_checkpoint_{checkpoint_num}.pt"
    else:
        save_file_name = f"{model_name}_final_model.pt"

    model_save_path = os.path.join(folder_path, save_file_name)

    with open(model_save_path, "wb") as f:
        torch.save(model_obj, f)

    log_debug(f"Model saved to: {model_save_path}")
