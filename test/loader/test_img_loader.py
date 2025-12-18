import os
import unittest
import numpy as np
from torchvision.io import read_image
import torch

from src.loader import IMGLoader

from test.paths import tmp_path
from test.utils import create_random_images

class TestIMGLoader(unittest.TestCase):
    def setUp(self):
        self.directory = os.path.join(tmp_path, "img_loader_test")
        create_random_images(100, (16, 16), self.directory)
        self.loader = IMGLoader(self.directory)

    def test_len(self):
        self.assertEqual(len(self.loader), 100)

    def test_getitem(self):
        image0 = read_image(os.path.join(self.directory, "000000.png"))
        self.assertTrue(np.allclose(self.loader[0], image0))

    def tearDown(self):
        for f in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, f))
        os.rmdir(self.directory)
        


if __name__ == '__main__':
    unittest.main()