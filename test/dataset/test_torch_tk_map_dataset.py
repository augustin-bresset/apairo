import unittest
import os

import numpy as np

from src.dataset.torch_wrapper import TorchTKDataset
from src.loader import NPYLoader

from test.paths import tartan2kitti_path


class TestTorchTKDataset(unittest.TestCase):
    """
    Test the wrapper of torch dataset applied to tartan dataset. 
    Be careful to not use images in the example that checks length because
    all of the images have not been downloaded.
    """

    def setUp(self):
        self.directory = tartan2kitti_path
        keys = ["controls", "image_left"]
        self.dataset = TorchTKDataset(directory=self.directory, keys=keys)

        self.true_timestamps = {
            key: np.loadtxt(os.path.join(self.directory, key, "timestamps.txt")) for key in keys
        }

    def test_len(self):
        """Check if the length of the dataset is correct"""
        self.dataset.keys = ["controls"]
        controls = NPYLoader(os.path.join(self.directory, "controls"))
        self.assertEqual(len(self.dataset), len(controls))

    def test_getitem(self):
        """Check the data loaded for the iteration over one key"""

        self.dataset.keys = ["controls"]
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            self.assertEqual(data.get("key"), "controls")

    def test_loaders(self):
        """Check if the loaders are correctly set."""
        self.dataset.keys = ["controls", "cmd"]
        self.assertEqual(len(self.dataset.loaders), 2)
        self.assertEqual(len(self.dataset.timestamps), 2)
        for key in self.dataset.keys:
            self.assertIn(key, self.dataset.loaders)
            self.assertIn(key, self.dataset.timestamps)
            self.assertEqual(len(self.dataset.loaders[key]), len(self.dataset.timestamps[key]))

    def test_getitem_time_order(self):
        """Check if the data is loaded in time order"""
        self.dataset.keys = ["controls", "cmd"]
        last_timestamp = 0
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            self.assertTrue(data["timestamp"] >= last_timestamp)
            last_timestamp = data["timestamp"]
