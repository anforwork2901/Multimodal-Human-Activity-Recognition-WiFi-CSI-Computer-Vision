"""PyTorch Dataset implementation for multimodal CV + CSI activity recognition."""

import os
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.domain.constants import ACTIVITY_CLASSES, DEFAULT_CSI_LENGTH, DEFAULT_IMAGE_SIZE
from src.domain.entities import ActivitySample
from src.preprocessing.csi_filters import CSIPreprocessor


class ActivityDataset(Dataset):
    """
    Unified Activity Dataset supporting both:
    1. Sequential mapped dataset (label_mapping.csv + activity/<Class>/images/)
    2. Timestamp direct dataset (<Class>/csi/<class>_balanced_csi.csv + <Class>/images/)
    """

    def __init__(
        self,
        dataset_path: str,
        label_mapping_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        samples_per_class: int = 1000,
        test_samples_per_class: int = 200,
        csi_preprocessor: Optional[CSIPreprocessor] = None,
        image_size: tuple = DEFAULT_IMAGE_SIZE,
    ):
        self.dataset_path = dataset_path
        self.label_mapping_path = label_mapping_path
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.samples_per_class = samples_per_class
        self.test_samples_per_class = test_samples_per_class
        self.csi_preprocessor = csi_preprocessor or CSIPreprocessor()
        self.image_size = image_size

        self.samples: List[ActivitySample] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Detect layout and load samples."""
        # Detect if label_mapping.csv exists
        if self.label_mapping_path and os.path.exists(self.label_mapping_path):
            self._load_from_label_mapping()
        elif os.path.exists(os.path.join(self.dataset_path, "label_mapping.csv")):
            self.label_mapping_path = os.path.join(self.dataset_path, "label_mapping.csv")
            self._load_from_label_mapping()
        else:
            self._load_from_directory_structure()

    def _load_from_label_mapping(self) -> None:
        """Load using label_mapping.csv and CV-CSI timestamp alignment."""
        label_df = pd.read_csv(self.label_mapping_path)
        activity_dir = (
            os.path.join(self.dataset_path, "activity")
            if os.path.exists(os.path.join(self.dataset_path, "activity"))
            else self.dataset_path
        )

        # Build filename mapping: frame_0001.jpg -> CSI row
        csi_filename_mapping = {}
        for activity_name in ACTIVITY_CLASSES.keys():
            csi_file = os.path.join(
                activity_dir, activity_name, f"{activity_name.lower()}_csi.csv"
            )
            if not os.path.exists(csi_file):
                # Check alternative csi/ subfolder
                csi_file = os.path.join(
                    activity_dir, activity_name, "csi", f"{activity_name.lower()}_balanced_csi.csv"
                )

            if os.path.exists(csi_file):
                try:
                    csi_df = pd.read_csv(csi_file)
                    if "csi_timestamp" in csi_df.columns:
                        csi_df["csi_timestamp"] = pd.to_datetime(csi_df["csi_timestamp"])
                        csi_df = csi_df.sort_values("csi_timestamp").reset_index(drop=True)

                    for idx, row in csi_df.iterrows():
                        seq_name = f"frame_{idx + 1:04d}.jpg"
                        csi_filename_mapping[seq_name] = row.get(
                            "normalized_csi_data", row.get("csi_data", "[]")
                        )
                except Exception:
                    pass

        # Split per class
        for activity_name, class_id in ACTIVITY_CLASSES.items():
            activity_samples = label_df[label_df["activity_name"] == activity_name].copy()
            valid_samples = [
                row
                for _, row in activity_samples.iterrows()
                if row["image_filename"] in csi_filename_mapping
            ]

            if not valid_samples:
                continue

            total = len(valid_samples)
            train_size = int(total * 0.7)
            val_size = int(total * 0.15)

            if self.split == "train":
                selected = valid_samples[:train_size]
            elif self.split == "val":
                selected = valid_samples[train_size : train_size + val_size]
            else:  # test
                selected = valid_samples[train_size + val_size :]

            for row in selected:
                fn = row["image_filename"]
                img_path = os.path.join(activity_dir, activity_name, "images", fn)
                csi_data = csi_filename_mapping.get(fn, "[]")

                self.samples.append(
                    ActivitySample(
                        class_id=class_id,
                        activity_name=activity_name,
                        image_path=img_path,
                        csi_data=csi_data,
                        cv_filename=fn,
                    )
                )

    def _load_from_directory_structure(self) -> None:
        """Load directly from structured folder <activity>/csi/ and <activity>/images/."""
        for activity_name, class_id in ACTIVITY_CLASSES.items():
            act_folder = os.path.join(self.dataset_path, activity_name)
            img_folder = os.path.join(act_folder, "images")
            csi_file = os.path.join(
                act_folder, "csi", f"{activity_name.lower()}_balanced_csi.csv"
            )
            if not os.path.exists(csi_file):
                csi_file = os.path.join(act_folder, f"{activity_name.lower()}_csi.csv")

            if not os.path.exists(img_folder) or not os.path.exists(csi_file):
                continue

            csi_df = pd.read_csv(csi_file)
            if "csi_timestamp" in csi_df.columns:
                csi_df["csi_timestamp"] = pd.to_datetime(csi_df["csi_timestamp"])
                csi_df = csi_df.sort_values("csi_timestamp").reset_index(drop=True)

            available = len(csi_df)
            train_val_end = min(self.samples_per_class, available)
            train_end = int(train_val_end * self.train_ratio)

            if self.split == "train":
                selected_df = csi_df.iloc[:train_end]
            elif self.split == "val":
                selected_df = csi_df.iloc[train_end:train_val_end]
            else:  # test
                test_start = train_val_end
                test_end = min(test_start + self.test_samples_per_class, available)
                selected_df = csi_df.iloc[test_start:test_end]

            for _, row in selected_df.iterrows():
                image_fn = row["image_filename"]
                img_path = os.path.join(img_folder, image_fn)
                csi_data = row.get("normalized_csi_data", row.get("csi_data", "[]"))

                self.samples.append(
                    ActivitySample(
                        class_id=class_id,
                        activity_name=activity_name,
                        image_path=img_path,
                        csi_data=csi_data,
                        cv_filename=image_fn,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 1. Load image
        try:
            if not os.path.exists(sample.image_path):
                image = torch.zeros(3, self.image_size[0], self.image_size[1])
            else:
                img_pil = Image.open(sample.image_path).convert("RGB")
                if self.transform:
                    image = self.transform(img_pil)
                else:
                    image = torch.zeros(3, self.image_size[0], self.image_size[1])
        except Exception:
            image = torch.zeros(3, self.image_size[0], self.image_size[1])

        # 2. Process CSI
        csi_processed = self.csi_preprocessor.preprocess(sample.csi_data)
        csi_tensor = torch.tensor(csi_processed, dtype=torch.float32)

        return {
            "images": image,
            "csi": csi_tensor,
            "class_ids": sample.class_id,
        }
