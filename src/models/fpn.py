"""Feature Pyramid Network (FPN) and Computer Vision Backbone components."""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block with skip connection for stable gradient propagation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CVBackbone(nn.Module):
    """
    Multi-Scale Feature Extractor Backbone.
    Produces 4 feature stages: C1 (16ch), C2 (32ch), C3 (64ch), C4 (128ch).
    """

    def __init__(self):
        super().__init__()
        # C1: 224 -> 112 -> 56
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        # C2: 56 -> 28 -> 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)

        # C3: 14 -> 7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # C4: 7 -> 7
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool1(c1)  # [B, 16, 56, 56]

        c2 = self.conv2(c1)
        c2 = self.bn2(c2)
        c2 = self.relu(c2)
        c2 = self.maxpool2(c2)  # [B, 32, 14, 14]

        c3 = self.conv3(c2)
        c3 = self.bn3(c3)
        c3 = self.relu(c3)  # [B, 64, 7, 7]

        c4 = self.conv4(c3)
        c4 = self.bn4(c4)
        c4 = self.relu(c4)  # [B, 128, 7, 7]

        return [c1, c2, c3, c4]


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) with lateral connections and top-down pathways.
    Combines high-level semantic features with low-level spatial details.
    """

    def __init__(self, in_channels: List[int] = [16, 32, 64, 128], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        fpn_features = []
        # Start from highest level (P4)
        prev_feat = laterals[-1]
        fpn_features.append(self.fpn_convs[-1](prev_feat))

        # Top-down pathway with bilinear upsampling & element-wise addition
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                prev_feat,
                size=laterals[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = laterals[i] + upsampled
            fpn_feat = self.fpn_convs[i](fused)
            fpn_features.insert(0, fpn_feat)
            prev_feat = fused

        return fpn_features
