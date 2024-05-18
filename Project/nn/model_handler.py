"""
Helper for reading and saving model files
Hugo Burton
"""

import os
import torch
from typing import List, Union

import file_helper

from logger import *

NN_MODEL_NAME = "nn"


def read_model(model_save_path: str) -> Union[torch.nn.Module, None]:
    """
    Handler for reading a model from a file.

    Args:
    - model_save_path (str): The path to the model file.

    Returns:
    - torch.nn.Module: The model object if it exists else None.
    """

    # Check if the model file exists
    if not file_helper.file_exists(model_save_path):
        log_error(f"Model file does not exist: {model_save_path}")
        return None

    with open(model_save_path, "rb") as f:
        model_obj = torch.load(f)

    if not model_obj:
        log_error(f"Model object is None")
        return None

    if not isinstance(model_obj, dict):
        log_error(f"Model object is not a dictionary")
        return None

    # Check if the model is a checkpoint
    if model_obj["is_checkpoint"]:
        log_debug(
            f"Reading checkpoint {model_obj['checkpoint_num']} from: {model_save_path}")
    else:
        log_debug(f"Reading final model from: {model_save_path}")

    return model_obj


def save_model(folder_path: str, model: torch.nn.Module, metrics: List, checkpoint_num: int = None):
    """
    Handles saving the model to a file.

    Args:
    - folder_path (str): The folder to save the model to.
    - model (torch.nn.Module): The model to save.
    - metrics (List): The metrics to save.
    - checkpoint_num (int): The checkpoint number. If None, then it is the final model.
    """

    is_checkpoint = checkpoint_num is not None

    model_obj = {
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "is_cuda": torch.cuda.is_available(),
        "model_name": NN_MODEL_NAME,
        "is_checkpoint": is_checkpoint,
        "checkpoint_num": checkpoint_num
    }

    if is_checkpoint:
        save_file_name = f"{NN_MODEL_NAME}_checkpoint_{checkpoint_num}.pt"
    else:
        save_file_name = f"{NN_MODEL_NAME}_final_model.pt"

    model_save_path = os.path.join(folder_path, save_file_name)

    with open(model_save_path, "wb") as f:
        torch.save(model_obj, f)

    log_title(f"Model saved to: {model_save_path}")
