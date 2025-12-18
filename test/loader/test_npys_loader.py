import os
import unittest
import numpy as np
import torch

from src.loader import NPYSLoader

from test.paths import tmp_path
from test.utils import create_random_npy_files

class TestNPYSLoader(unittest.TestCase):
    def setUp(self):
        
        self.npys_folder = os.path.join(tmp_path, "npys_loader_test")
        create_random_npy_files(100, (5, 5), self.npys_folder)
        create_random_npy_files(100, (5, 5), self.npys_folder, "intensity")
        self.loader = NPYSLoader(self.npys_folder)

    def test_format(self):
        self.assertEqual(self.loader.npy_formats, {"", "intensity"})

    def test_len(self):
        self.assertEqual(len(self.loader), 100)

    def test_getitem(self):
        file0 = torch.from_numpy(np.load(os.path.join(self.npys_folder, "000000.npy")))
        file0_intensity = torch.from_numpy(np.load(os.path.join(self.npys_folder, "000000_intensity.npy")))

        self.assertTrue(np.allclose(self.loader[0], file0))
        
        self.loader.set_format("intensity")
        file0_intensity = torch.from_numpy(np.load(os.path.join(self.npys_folder, "000000_intensity.npy")))
        self.assertTrue(np.allclose(self.loader[0], file0_intensity))

    def tearDown(self):
        for f in os.listdir(self.npys_folder):
            os.remove(os.path.join(self.npys_folder, f))
        os.rmdir(self.npys_folder)

if __name__ == '__main__':
    unittest.main()