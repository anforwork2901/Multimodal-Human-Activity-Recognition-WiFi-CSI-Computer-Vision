"""Domain interfaces and abstract contracts (SOLID - DIP, ISP)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from src.domain.entities import EvaluationResult, TrainingHistory


class IActivityModel(ABC):
    """Abstract interface for all activity recognition models (Vision, WiFi, Fusion)."""

    @abstractmethod
    def forward(
        self, images: Optional[torch.Tensor] = None, csi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns:
            logits: (Batch, NumClasses)
            features: Optional feature embedding (Batch, FeatureDim) for metric learning / triplet loss
        """
        pass


class ICSIPreprocessor(ABC):
    """Abstract interface for CSI signal denoising and normalization."""

    @abstractmethod
    def preprocess(self, raw_csi: Any) -> List[float]:
        """Convert raw CSI representation to a cleaned, normalized float vector."""
        pass


class IDataModule(ABC):
    """Abstract interface for preparing and providing datasets and dataloaders."""

    @abstractmethod
    def get_train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_val_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_test_dataloader(self) -> DataLoader:
        pass


class ITrainer(ABC):
    """Abstract interface for model training."""

    @abstractmethod
    def train(
        self,
        model: IActivityModel,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[IActivityModel, TrainingHistory]:
        pass


class IEvaluator(ABC):
    """Abstract interface for evaluating models."""

    @abstractmethod
    def evaluate(
        self, model: IActivityModel, model_name: str, test_loader: DataLoader
    ) -> EvaluationResult:
        pass
