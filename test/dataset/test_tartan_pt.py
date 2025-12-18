import unittest
import os
from src.dataset.tartan_pt import TartanPT

from test.paths import tmp_path
from test.utils import create_random_pt_file

class TestTartanPT(unittest.TestCase):

    def setUp(self) -> None:
        ...