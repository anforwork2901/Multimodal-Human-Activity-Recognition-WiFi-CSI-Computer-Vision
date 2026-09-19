"""Domain entities and value objects."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ActivitySample:
    """Represents a single multimodal sample combining CV frame and CSI carrier information."""
    class_id: int
    activity_name: str
    image_path: str
    csi_data: Any
    cv_filename: Optional[str] = None


@dataclass
class TrainingHistory:
    """Historical training and validation progress across epochs."""
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Evaluation metrics on test dataset."""
    model_name: str
    accuracy: float
    loss: float
    predictions: List[int] = field(default_factory=list)
    targets: List[int] = field(default_factory=list)
    probabilities: List[List[float]] = field(default_factory=list)
    classification_report: Dict[str, Any] = field(default_factory=dict)
