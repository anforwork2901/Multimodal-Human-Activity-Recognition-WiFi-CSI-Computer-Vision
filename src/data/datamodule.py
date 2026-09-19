"""DataModule managing dataset instances and DataLoader pipelines."""

import os
from typing import Optional
from torch.utils.data import DataLoader

from src.config.settings import AppConfig
from src.data.dataset import ActivityDataset
from src.domain.interfaces import IDataModule
from src.preprocessing.csi_filters import CSIPreprocessor
from src.preprocessing.image_transforms import (
    get_train_transforms,
    get_val_test_transforms,
)


class ActivityDataModule(IDataModule):
    """Encapsulates all data setup, splitting, and DataLoader generation."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.csi_preprocessor = CSIPreprocessor(target_length=config.data.csi_length)
        self.train_transform = get_train_transforms(image_size=config.data.image_size)
        self.val_test_transform = get_val_test_transforms(image_size=config.data.image_size)

        self._resolve_dataset_path()

        self.train_dataset: Optional[ActivityDataset] = None
        self.val_dataset: Optional[ActivityDataset] = None
        self.test_dataset: Optional[ActivityDataset] = None
        self.setup()

    def _resolve_dataset_path(self) -> None:
        """Automatically find available dataset base path."""
        candidate_paths = [
            self.config.paths.dataset_base_path,
            os.path.join(
                os.path.dirname(__file__),
                "../../",
                self.config.paths.dataset_base_path,
            ),
            self.config.paths.fallback_dataset_path,
            os.path.join(
                os.path.dirname(__file__),
                "../../",
                self.config.paths.fallback_dataset_path,
            ),
        ]
        self.resolved_path = ""
        for p in candidate_paths:
            if os.path.exists(p):
                self.resolved_path = os.path.abspath(p)
                break

        if not self.resolved_path:
            # Fallback to configured path
            self.resolved_path = self.config.paths.dataset_base_path

    def setup(self) -> None:
        """Create datasets for train, val, test splits."""
        self.train_dataset = ActivityDataset(
            dataset_path=self.resolved_path,
            transform=self.train_transform,
            split="train",
            train_ratio=self.config.data.train_split,
            val_ratio=self.config.data.val_split,
            samples_per_class=self.config.data.samples_per_class_train_val,
            test_samples_per_class=self.config.data.test_samples_per_class,
            csi_preprocessor=self.csi_preprocessor,
            image_size=self.config.data.image_size,
        )

        self.val_dataset = ActivityDataset(
            dataset_path=self.resolved_path,
            transform=self.val_test_transform,
            split="val",
            train_ratio=self.config.data.train_split,
            val_ratio=self.config.data.val_split,
            samples_per_class=self.config.data.samples_per_class_train_val,
            test_samples_per_class=self.config.data.test_samples_per_class,
            csi_preprocessor=self.csi_preprocessor,
            image_size=self.config.data.image_size,
        )

        self.test_dataset = ActivityDataset(
            dataset_path=self.resolved_path,
            transform=self.val_test_transform,
            split="test",
            train_ratio=self.config.data.train_split,
            val_ratio=self.config.data.val_split,
            samples_per_class=self.config.data.samples_per_class_train_val,
            test_samples_per_class=self.config.data.test_samples_per_class,
            csi_preprocessor=self.csi_preprocessor,
            image_size=self.config.data.image_size,
        )

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

    def get_test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )
