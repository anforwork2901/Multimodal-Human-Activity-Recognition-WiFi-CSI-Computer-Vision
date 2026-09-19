"""Training callbacks: Checkpointing, Early Stopping, and Learning Rate monitoring."""

import copy
import os
from typing import Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """Monitors validation metric and triggers early termination if no improvement."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelCheckpoint:
    """Manages saving and caching best model weights."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.best_state_dict = None

    def save_best(self, model: nn.Module) -> None:
        self.best_state_dict = copy.deepcopy(model.state_dict())
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        torch.save(self.best_state_dict, self.filepath)

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        elif os.path.exists(self.filepath):
            model.load_state_dict(torch.load(self.filepath, map_location="cpu"))
