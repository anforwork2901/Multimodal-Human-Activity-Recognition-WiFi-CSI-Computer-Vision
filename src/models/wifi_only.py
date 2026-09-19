"""WiFi CSI-Only Neural Network Architecture."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.base import BaseActivityModel


class WiFiOnlyNN(BaseActivityModel):
    """
    WiFi CSI-Only Neural Network with multi-stage Linear-BatchNorm-Dropout regularization.
    Extracts wireless RF disturbance features for activity classification.
    """

    def __init__(
        self, csi_length: int = 102, num_classes: int = 8, task: str = "classification"
    ):
        super().__init__(num_classes=num_classes, task=task)
        self.csi_length = csi_length

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(csi_length),
            nn.Linear(csi_length, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        out_features = num_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32, out_features),
        )

    def forward(
        self, images: Optional[torch.Tensor] = None, csi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if csi is None:
            raise ValueError("WiFiOnlyNN requires 'csi' input tensor.")

        if csi.dim() == 1:
            csi = csi.unsqueeze(0)

        # Dimension alignment
        if csi.size(1) != self.csi_length:
            if csi.size(1) < self.csi_length:
                padding = torch.zeros(
                    csi.size(0), self.csi_length - csi.size(1), device=csi.device
                )
                csi = torch.cat([csi, padding], dim=1)
            else:
                csi = csi[:, : self.csi_length]

        features = self.feature_extractor(csi)  # [B, 32]
        output = self.head(features)

        if self.task == "regression":
            output = output.squeeze(-1)

        return output, features
