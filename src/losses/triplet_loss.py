"""Triplet Loss and metric learning utilities for multimodal representation alignment."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning (GaitFi methodology).
    Enforces anchor-positive distance < anchor-negative distance by margin.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance(p=2)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        pos_dist = self.distance(anchor, positive)
        neg_dist = self.distance(anchor, negative)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)


def create_triplets(
    features: torch.Tensor, labels: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract (anchor, positive, negative) feature triplets from a batch.
    """
    labels_np = labels.detach().cpu().numpy()
    unique_labels = np.unique(labels_np)

    anchors = []
    positives = []
    negatives = []

    for label in unique_labels:
        pos_indices = np.where(labels_np == label)[0]
        neg_indices = np.where(labels_np != label)[0]

        if len(pos_indices) >= 2 and len(neg_indices) >= 1:
            for i in range(len(pos_indices)):
                anchor_idx = pos_indices[i]
                for j in range(len(pos_indices)):
                    if i != j:
                        positive_idx = pos_indices[j]
                        neg_idx = np.random.choice(neg_indices)

                        anchors.append(features[anchor_idx])
                        positives.append(features[positive_idx])
                        negatives.append(features[neg_idx])

    if len(anchors) == 0:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
