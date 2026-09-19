"""Unit tests for CSI and Vision preprocessing modules."""

import unittest
import numpy as np
from PIL import Image
import torch

from src.preprocessing.csi_filters import (
    CSIPreprocessor,
    enhanced_hampel_filter,
    moving_average_filter,
    normalize_csi,
)
from src.preprocessing.image_transforms import (
    get_train_transforms,
    get_val_test_transforms,
)


class TestPreprocessing(unittest.TestCase):
    def test_moving_average_filter(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        filtered = moving_average_filter(data, window_size=3)
        self.assertEqual(len(filtered), len(data))
        # Center value (index 2: window [2, 3, 4]) should be 3.0
        self.assertAlmostEqual(filtered[2], 3.0)

    def test_enhanced_hampel_filter(self):
        # Array with a single extreme outlier at index 5
        data = [10.0, 10.1, 10.0, 9.9, 10.0, 1000.0, 10.0, 9.9, 10.1, 10.0]
        filtered = enhanced_hampel_filter(data, K=2, n_sigmas=3.0)
        self.assertEqual(len(filtered), len(data))
        # The outlier should be replaced by median near 10.0
        self.assertLess(filtered[5], 50.0)

    def test_normalize_csi(self):
        data = [0.0, 50.0, 100.0]
        norm = normalize_csi(data)
        self.assertAlmostEqual(norm[0], -1.0)
        self.assertAlmostEqual(norm[1], 0.0)
        self.assertAlmostEqual(norm[2], 1.0)

    def test_normalize_csi_constant(self):
        data = [5.0, 5.0, 5.0]
        norm = normalize_csi(data)
        self.assertEqual(norm, [0.0, 0.0, 0.0])

    def test_csi_preprocessor_pipeline(self):
        preprocessor = CSIPreprocessor(target_length=102)

        # 1. Valid string list
        raw_str = "[-10, 20, 30, -40, 50]"
        out = preprocessor.preprocess(raw_str)
        self.assertEqual(len(out), 102)
        self.assertIsInstance(out[0], float)

        # 2. None or empty string input
        out_empty = preprocessor.preprocess(None)
        self.assertEqual(len(out_empty), 102)
        self.assertEqual(out_empty, [0.0] * 102)

        # 3. Truncation when input exceeds target_length
        long_input = [1.0] * 200
        out_long = preprocessor.preprocess(long_input)
        self.assertEqual(len(out_long), 102)

    def test_image_transforms(self):
        # Create dummy RGB image
        img = Image.new("RGB", (300, 300), color=(128, 64, 32))

        train_tf = get_train_transforms(image_size=(224, 224))
        val_tf = get_val_test_transforms(image_size=(224, 224))

        tensor_train = train_tf(img)
        tensor_val = val_tf(img)

        self.assertEqual(tensor_train.shape, (3, 224, 224))
        self.assertEqual(tensor_val.shape, (3, 224, 224))
        self.assertEqual(tensor_train.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
