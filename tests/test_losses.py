"""Unit tests for metric learning loss functions."""

import unittest
import torch
from src.losses.triplet_loss import TripletLoss, create_triplets


class TestLosses(unittest.TestCase):
    def test_triplet_loss_computation(self):
        loss_fn = TripletLoss(margin=0.2)

        # Anchor, Positive close; Negative far
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.1]])
        negative = torch.tensor([[0.0, 1.0]])

        loss = loss_fn(anchor, positive, negative)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_triplet_loss_margin(self):
        loss_fn = TripletLoss(margin=0.5)

        # Anchor and Positive identical; Negative very far away -> Loss should be 0.0
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[0.0, 0.0]])
        negative = torch.tensor([[100.0, 100.0]])

        loss = loss_fn(anchor, positive, negative)
        self.assertEqual(loss.item(), 0.0)

    def test_create_triplets_mining(self):
        # Create features with 2 samples of class 0, 1 sample of class 1
        features = torch.tensor([
            [1.0, 1.0],
            [1.1, 1.0],
            [9.0, 9.0],
        ])
        labels = torch.tensor([0, 0, 1])

        anchors, positives, negatives = create_triplets(features, labels)
        self.assertIsNotNone(anchors)
        self.assertIsNotNone(positives)
        self.assertIsNotNone(negatives)
        self.assertEqual(anchors.shape, positives.shape)
        self.assertEqual(positives.shape, negatives.shape)


if __name__ == "__main__":
    unittest.main()
