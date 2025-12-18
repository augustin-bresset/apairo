import unittest
import os
import numpy as np

from src.dataset.torch_wrapper import TorchTKIterDataset
from src.loader import NPYSLoader, IMGLoader, NPYLoader

from test.paths import tartan2kitti_path



class TestTorchTKIterDataset(unittest.TestCase):
    """ 
    Test the wrapper of iterable dataset torch applied to tartan dataset. 
    """
    def setUp(self):
        self.directory = tartan2kitti_path
        keys = ["controls", "image_left", "image_right"]
        self.dataset = TorchTKIterDataset(directory=self.directory, keys=keys)

        self.true_timestamps = {
            key: np.loadtxt(os.path.join(self.directory, key, "timestamps.txt")) for key in keys
        }

    def test_init_timeline(self):
        """Check if the timeline is correctly initialized"""
        for key in self.dataset.keys:
            self.assertEqual(self.dataset.timelines[self.dataset.k2idx[key]], self.true_timestamps[key][0])

    def test_iter_one_key(self):
        """Check the data loaded for the iteration over one key"""
        self.dataset.keys = ["controls"]
        image_loader = NPYLoader(os.path.join(self.directory, "controls"))
        for i, data in enumerate(self.dataset):
            self.assertTrue(np.allclose(data["data"], image_loader[i]))

    def test_iter_two_keys_same_timestamps(self):
        """Check the data loaded for the iteration over two keys with the same timestamps"""
        loaders = {
            "image_left": IMGLoader(os.path.join(self.directory, "image_left")), 
            "image_right": IMGLoader(os.path.join(self.directory, "image_right"))
        }
        self.dataset.keys = list(loaders.keys())
        
        previous_data = {key: {"data": None, "index": -1} for key in loaders.keys()}
        # We limit the number of iterations to 100 because image has only 100 timestamps
        for _, data in zip(range(100), self.dataset):
            self.assertIn(data.get("key"), loaders.keys())
            previous_data[data["key"]]["index"] +=1
            previous_data[data["key"]]["data"] = data["data"]
            self.assertTrue(np.allclose(data["data"], loaders[data["key"]][previous_data[data["key"]]["index"]]))
            
    def test_iter_two_keys_different_timestamps(self):
        """Check the data loaded for the iteration over two keys with different timestamps"""
        loaders = {
            "velodyne_0": NPYSLoader(os.path.join(self.directory, "velodyne_0")), 
            "controls": NPYLoader(os.path.join(self.directory, "controls"))
            }
        self.dataset.keys = list(loaders.keys())
        
        previous_data = {key: {"data": None, "index": -1} for key in loaders.keys()}
        # We limit the number of iterations to 100 because velodyne_0 has only 100 timestamps
        for _, data in zip(range(100), self.dataset):
            self.assertIn(data.get("key"), loaders.keys())
            previous_data[data["key"]]["index"] +=1
            previous_data[data["key"]]["data"] = data["data"]
            self.assertTrue(np.allclose(data["data"], loaders[data["key"]][previous_data[data["key"]]["index"]]))
            

    def test_length(self):
        """Check if the length of the dataset is correct"""
        self.dataset.keys = ["controls"]
        controls = NPYLoader(os.path.join(self.directory, "controls"))
        self.assertEqual(len(list(self.dataset)), len(controls))
    
