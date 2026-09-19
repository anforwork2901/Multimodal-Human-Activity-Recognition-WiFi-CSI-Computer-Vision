"""Unit tests for training callbacks."""

import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn

from src.training.callbacks import EarlyStopping, ModelCheckpoint


class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_early_stopping(self):
        es = EarlyStopping(patience=3, mode="max")

        self.assertTrue(es.step(80.0))
        self.assertFalse(es.early_stop)

        # Improvement
        self.assertTrue(es.step(85.0))
        self.assertEqual(es.counter, 0)

        # Non-improving steps
        self.assertFalse(es.step(84.0))
        self.assertEqual(es.counter, 1)

        self.assertFalse(es.step(83.0))
        self.assertEqual(es.counter, 2)

        self.assertFalse(es.step(82.0))
        self.assertEqual(es.counter, 3)
        self.assertTrue(es.early_stop)

    def test_model_checkpoint(self):
        ckpt_path = os.path.join(self.temp_dir, "test_model.pth")
        checkpoint = ModelCheckpoint(filepath=ckpt_path)

        model = nn.Linear(4, 2)
        # Modify weights
        with torch.no_grad():
            model.weight.fill_(5.0)

        checkpoint.save_best(model)
        self.assertTrue(os.path.exists(ckpt_path))

        # Change model weights
        with torch.no_grad():
            model.weight.fill_(0.0)
        self.assertEqual(model.weight[0, 0].item(), 0.0)

        # Restore
        checkpoint.restore_best(model)
        self.assertAlmostEqual(model.weight[0, 0].item(), 5.0)


if __name__ == "__main__":
    unittest.main()
