"""Unit tests for dataset loading and dataloaders."""

import unittest
from src.config.settings import load_config
from src.data.datamodule import ActivityDataModule
from src.data.dataset import ActivityDataset


class TestData(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/default_config.yaml")
        self.config.data.batch_size = 2
        self.config.data.num_workers = 0
        dm = ActivityDataModule(self.config)
        if len(dm.train_dataset) == 0:
            self.skipTest(
                "Dataset not found in local workspace (expected when cloning repo without large dataset)"
            )

    def test_datamodule_and_dataset(self):
        dm = ActivityDataModule(self.config)

        self.assertIsNotNone(dm.train_dataset)
        self.assertGreater(len(dm.train_dataset), 0)

        # Fetch one sample from dataset directly
        sample = dm.train_dataset[0]
        self.assertIn("images", sample)
        self.assertIn("csi", sample)
        self.assertIn("class_ids", sample)

        self.assertEqual(sample["images"].shape, (3, 224, 224))
        self.assertEqual(sample["csi"].shape, (102,))
        self.assertIsInstance(sample["class_ids"], int)

    def test_dataloaders(self):
        dm = ActivityDataModule(self.config)
        train_loader = dm.get_train_dataloader()
        val_loader = dm.get_val_dataloader()
        test_loader = dm.get_test_dataloader()

        batch = next(iter(train_loader))
        self.assertEqual(batch["images"].shape[0], 2)
        self.assertEqual(batch["csi"].shape[0], 2)
        self.assertEqual(len(batch["class_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
