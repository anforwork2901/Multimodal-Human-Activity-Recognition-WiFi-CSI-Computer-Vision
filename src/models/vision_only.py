"""Vision-Only Convolutional Neural Network Architecture."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.base import BaseActivityModel


class VisionOnlyCNN(BaseActivityModel):
    """
    Vision-Only CNN Model with downsampling blocks and adaptive pooling.
    Extracts visual features from CV video frames.
    """

    def __init__(self, num_classes: int = 8, task: str = "classification"):
        super().__init__(num_classes=num_classes, task=task)

        # Multi-stage convolution with downsampling
        self.conv_block = nn.Sequential(
            # Stage 1: 224 -> 112 -> 56
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Stage 2: 56 -> 28 -> 14
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Stage 3: 14 -> 7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Global average pooling: 7x7 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        feature_size = 64
        self.feature_size = feature_size

        # Classification / Regression Head
        out_features = num_classes if task == "classification" else 1
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, out_features),
        )

    def forward(
        self, images: Optional[torch.Tensor] = None, csi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if images is None:
            raise ValueError("VisionOnlyCNN requires 'images' input tensor.")

        # Extract features
        x = self.conv_block(images)  # [B, 64, 1, 1]
        features = x.view(x.size(0), -1)  # [B, 64]
        output = self.classifier(features)

        if self.task == "regression":
            output = output.squeeze(-1)

        return output, features
