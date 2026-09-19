import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import torch


@dataclass
class ProjectConfig:
    name: str = "Multimodal-CSI-CV-Activity-Recognition"
    version: str = "1.0.0"
    random_seed: int = 42
    device: str = "auto"


@dataclass
class PathConfig:
    dataset_base_path: str = "Dataset/Data_acti_CSI_CV/Activity_CV_CSI_Dataset"
    fallback_dataset_path: str = "data_activity"
    output_base_path: str = "Output"

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.output_base_path, "models")

    @property
    def results_save_path(self) -> str:
        return os.path.join(self.output_base_path, "results")

    @property
    def plots_save_path(self) -> str:
        return os.path.join(self.output_base_path, "plots")

    def ensure_directories(self) -> None:
        for path in [
            self.output_base_path,
            self.model_save_path,
            self.results_save_path,
            self.plots_save_path,
        ]:
            os.makedirs(path, exist_ok=True)


@dataclass
class DataConfig:
    image_size: Tuple[int, int] = (224, 224)
    csi_length: int = 102
    num_classes: int = 8
    samples_per_class_train_val: int = 1000
    test_samples_per_class: int = 200
    train_split: float = 0.8
    val_split: float = 0.2
    batch_size: int = 16
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 5
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    gradient_clip_max_norm: float = 1.0
    models_to_train: List[str] = field(
        default_factory=lambda: ["Vision-Only", "CSI-Only", "Fusion"]
    )


@dataclass
class ModelConfig:
    feature_dim: int = 64
    triplet_margin: float = 0.2
    triplet_loss_weight: float = 0.001


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    classes: Dict[str, int] = field(
        default_factory=lambda: {
            "KhongHanhDong": 0,
            "VayTay": 1,
            "KeoGhe": 2,
            "Dung": 3,
            "Nam": 4,
            "Nga": 5,
            "Ngoi": 6,
            "NhatDo": 7,
        }
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def get_device(self) -> torch.device:
        """Resolve target device dynamically with priority: CUDA > MPS > CPU."""
        if self.project.device != "auto":
            try:
                return torch.device(self.project.device)
            except Exception:
                pass

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file or return default configuration."""
    if config_path is None or not os.path.exists(config_path):
        # Try finding configs/default_config.yaml relative to current working directory
        possible_paths = [
            "configs/default_config.yaml",
            os.path.join(os.path.dirname(__file__), "../../configs/default_config.yaml"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                config_path = p
                break

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        project = ProjectConfig(**raw.get("project", {}))
        paths = PathConfig(**raw.get("paths", {}))
        
        data_raw = raw.get("data", {})
        if "image_size" in data_raw and isinstance(data_raw["image_size"], list):
            data_raw["image_size"] = tuple(data_raw["image_size"])
        data = DataConfig(**data_raw)

        training = TrainingConfig(**raw.get("training", {}))
        model = ModelConfig(**raw.get("model", {}))
        classes = raw.get("classes", AppConfig().classes)

        return AppConfig(
            project=project,
            paths=paths,
            data=data,
            classes=classes,
            training=training,
            model=model,
        )

    return AppConfig()
