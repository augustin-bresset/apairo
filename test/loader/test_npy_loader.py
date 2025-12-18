import os
import unittest
import numpy as np

from src.loader import NPYLoader

from test.paths import tmp_path
from test.utils import create_npy_file

class TestNPYLoader(unittest.TestCase):
    def setUp(self):
        self.data = np.array([1, 2, 3, 4, 5])
        create_npy_file(self.data, filename="data.npy", directory="npy_loader_test")
        self.loader = NPYLoader(os.path.join(tmp_path, "npy_loader_test"))

    def test_len(self):
        self.assertEqual(self.loader.array.shape, self.data.shape)

    def test_getitem(self):
        for i in range(len(self.loader)):
            self.assertTrue(np.allclose(self.loader[i], self.data[i]))

    def test_shape(self):
        self.assertEqual(self.loader.shape, (1))

    def tearDown(self):
        os.remove(os.path.join(tmp_path, "npy_loader_test", "data.npy"))
        os.rmdir(os.path.join(tmp_path, "npy_loader_test"))

if __name__ == "__main__":
    unittest.main()