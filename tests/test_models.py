"""Unit tests for Deep Learning model architectures."""

import unittest
import torch

from src.models.base import BaseActivityModel
from src.models.fpn import CVBackbone, FPN, ResidualBlock
from src.models.fusion_net import CSIBranch, HybridFPNFusionNet
from src.models.vision_only import VisionOnlyCNN
from src.models.wifi_only import WiFiOnlyNN


class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_classes = 8
        self.csi_length = 102
        self.dummy_images = torch.randn(self.batch_size, 3, 224, 224)
        self.dummy_csi = torch.randn(self.batch_size, self.csi_length)

    def test_vision_only_cnn(self):
        model = VisionOnlyCNN(num_classes=self.num_classes)
        self.assertIsInstance(model, BaseActivityModel)
        self.assertGreater(model.count_parameters(), 0)

        logits, features = model(images=self.dummy_images)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(features.shape, (self.batch_size, 64))

        # Test regression task mode
        reg_model = VisionOnlyCNN(num_classes=1, task="regression")
        reg_out, _ = reg_model(images=self.dummy_images)
        self.assertEqual(reg_out.shape, (self.batch_size,))

    def test_wifi_only_nn(self):
        model = WiFiOnlyNN(csi_length=self.csi_length, num_classes=self.num_classes)
        self.assertIsInstance(model, BaseActivityModel)

        logits, features = model(csi=self.dummy_csi)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(features.shape, (self.batch_size, 32))

        # Test 1D tensor handling during inference (eval mode)
        model.eval()
        single_csi = torch.randn(self.csi_length)
        single_out, _ = model(csi=single_csi)
        self.assertEqual(single_out.shape, (1, self.num_classes))

    def test_fpn_components(self):
        backbone = CVBackbone()
        stages = backbone(self.dummy_images)
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[0].shape[1], 16)   # C1
        self.assertEqual(stages[1].shape[1], 32)   # C2
        self.assertEqual(stages[2].shape[1], 64)   # C3
        self.assertEqual(stages[3].shape[1], 128)  # C4

        fpn = FPN(in_channels=[16, 32, 64, 128], out_channels=256)
        pyramids = fpn(stages)
        self.assertEqual(len(pyramids), 4)
        for p in pyramids:
            self.assertEqual(p.shape[1], 256)

        csi_branch = CSIBranch(input_dim=self.csi_length, output_dim=256)
        csi_feat = csi_branch(self.dummy_csi)
        self.assertEqual(csi_feat.shape, (self.batch_size, 256))

    def test_hybrid_fpn_fusion_net(self):
        model = HybridFPNFusionNet(
            csi_length=self.csi_length, num_classes=self.num_classes
        )
        self.assertIsInstance(model, BaseActivityModel)
        self.assertGreater(model.count_parameters(), 0)

        logits, features = model(images=self.dummy_images, csi=self.dummy_csi)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        # 512 (CV-FPN) + 256 (CSI) = 768
        self.assertEqual(features.shape, (self.batch_size, 768))


if __name__ == "__main__":
    unittest.main()
