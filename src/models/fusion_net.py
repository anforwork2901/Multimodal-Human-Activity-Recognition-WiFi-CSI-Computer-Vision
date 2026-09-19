"""Hybrid FPN Fusion Network combining Vision (CV-FPN) and Wireless (CSI)."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseActivityModel
from src.models.fpn import CVBackbone, FPN


class CSIBranch(nn.Module):
    """CSI signal processing branch aligned with WiFiOnlyNN structure."""

    def __init__(self, input_dim: int = 102, output_dim: int = 256):
        super().__init__()
        self.csi_layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
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
            nn.Linear(32, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        if csi.dim() == 1:
            csi = csi.unsqueeze(0)
        return self.csi_layers(csi)


class HybridFPNFusionNet(BaseActivityModel):
    """
    State-of-the-art Multimodal Fusion Network:
    - Computer Vision Branch: Multi-scale CVBackbone (C1..C4) + Feature Pyramid Network (FPN P1..P4) + Multi-level Aggregation -> 512D
    - WiFi CSI Branch: Multi-layer regularized RF representation -> 256D
    - Cross-Modal Fusion: Early-Late Concatenation -> 768D
    - Output Head: Multi-layer Classification Head -> Class Logits + Feature Embeddings (for Triplet Metric Learning)
    """

    def __init__(
        self, csi_length: int = 102, num_classes: int = 8, task: str = "classification"
    ):
        super().__init__(num_classes=num_classes, task=task)
        self.csi_length = csi_length

        # 1. Vision pathway with FPN
        self.cv_backbone = CVBackbone()
        self.fpn = FPN(in_channels=[16, 32, 64, 128], out_channels=256)

        # 2. CSI RF pathway
        self.csi_branch = CSIBranch(input_dim=csi_length, output_dim=256)

        # 3. Multi-scale feature aggregator
        self.feature_aggregator = nn.Sequential(
            nn.Conv2d(256 * 4, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # 4. Final fusion & output head
        fusion_dim = 512 + 256  # CV: 512, CSI: 256 -> 768
        if self.task == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
            )

    def forward(
        self, images: Optional[torch.Tensor] = None, csi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if images is None or csi is None:
            raise ValueError("HybridFPNFusionNet requires both 'images' and 'csi' inputs.")

        # 1. Vision Feature Pyramid Processing
        backbone_features = self.cv_backbone(images)  # [C1, C2, C3, C4]
        fpn_features = self.fpn(backbone_features)  # [P1, P2, P3, P4]

        # Resample all FPN levels to common spatial dimension
        target_size = fpn_features[0].shape[-2:]
        resized_fpn = []
        for feat in fpn_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            resized_fpn.append(feat)

        multi_scale_feat = torch.cat(resized_fpn, dim=1)  # [B, 256*4, H, W]
        cv_feat = self.feature_aggregator(multi_scale_feat)  # [B, 512, 1, 1]
        cv_feat = cv_feat.view(cv_feat.size(0), -1)  # [B, 512]

        # 2. CSI Signal Processing
        csi_feat = self.csi_branch(csi)  # [B, 256]

        # 3. Cross-Modal Fusion
        fused_feat = torch.cat([cv_feat, csi_feat], dim=1)  # [B, 768]

        # 4. Decision Head
        if self.task == "classification":
            output = self.classifier(fused_feat)
        else:
            output = self.regressor(fused_feat).squeeze(-1)

        return output, fused_feat
