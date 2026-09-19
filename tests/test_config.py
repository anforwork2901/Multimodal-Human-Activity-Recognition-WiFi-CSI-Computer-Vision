"""Unit tests for configuration and settings."""

import os
import unittest
import torch
from src.config.settings import AppConfig, load_config


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        config = AppConfig()
        self.assertEqual(config.project.name, "Multimodal-CSI-CV-Activity-Recognition")
        self.assertEqual(config.data.image_size, (224, 224))
        self.assertEqual(config.data.csi_length, 102)
        self.assertEqual(config.data.num_classes, 8)
        self.assertEqual(len(config.classes), 8)

    def test_yaml_config_loading(self):
        config = load_config("configs/default_config.yaml")
        self.assertIsInstance(config, AppConfig)
        self.assertIn("Fusion", config.training.models_to_train)
        self.assertEqual(config.training.learning_rate, 0.001)

    def test_device_resolution(self):
        config = AppConfig()
        device = config.get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ["cpu", "cuda", "mps"])

    def test_path_helpers(self):
        config = AppConfig()
        self.assertTrue(config.paths.model_save_path.endswith("models"))
        self.assertTrue(config.paths.results_save_path.endswith("results"))
        self.assertTrue(config.paths.plots_save_path.endswith("plots"))


if __name__ == "__main__":
    unittest.main()
