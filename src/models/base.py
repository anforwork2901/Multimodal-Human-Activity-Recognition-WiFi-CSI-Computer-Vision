"""Base class for all Activity Recognition Models (SOLID - LSP)."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.domain.interfaces import IActivityModel


class BaseActivityModel(nn.Module, IActivityModel, ABC):
    """
    Abstract Base Class for activity models.
    Guarantees standard forward signature:
        forward(images=..., csi=...) -> (logits, optional_features)
    """

    def __init__(self, num_classes: int = 8, task: str = "classification"):
        super().__init__()
        self.num_classes = num_classes
        self.task = task

    @abstractmethod
    def forward(
        self, images: Optional[torch.Tensor] = None, csi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns:
            logits: (Batch, NumClasses)
            features: Optional (Batch, FeatureDim) for metric learning / triplet loss
        """
        pass

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
