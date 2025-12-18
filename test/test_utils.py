import os
import sys
import unittest
import torch

from src.utils.utils import dict_flatten, npy_analyser
from test.paths import tartan2kitti_path


class TestUtils(unittest.TestCase):
    def test_dict_flatten(self):
        d = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        expected = {
            "a": 1,
            "c": 2,
            "e": 3
        }
        self.assertEqual(dict_flatten(d), expected)

        expected = {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3
        }
        format_key = lambda k, sk: f"{k}.{sk}"
        self.assertEqual(dict_flatten(d, format_key), expected, msg="format_key argument is not working")


    def test_npy_analyser(self):
        folder = tartan2kitti_path
        folder_npy = os.path.join(folder, "livox")
        self.assertEqual(npy_analyser(folder_npy), {"", "intensity"})

        folder_npy = os.path.join(folder, "cmd")

        self.assertEqual(npy_analyser(folder_npy), {""})

        
if __name__ == "__main__":
    unittest.main()


"""
import os
import pytest
import torch

from src.utils.utils import dict_flatten, npy_analyser
from test.paths import tartan2kitti_path

@pytest.fixture
def nested_dict():
    return {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        }
    }

@pytest.fixture
def tartan2kitti_folder():
    return tartan2kitti_path

def test_dict_flatten(nested_dict):
    expected = {
        "a": 1,
        "c": 2,
        "e": 3
    }
    assert dict_flatten(nested_dict) == expected

    expected = {
        "a": 1,
        "b.c": 2,
        "b.d.e": 3
    }
    format_key = lambda k, sk: f"{k}.{sk}"
    assert dict_flatten(nested_dict, format_key) == expected, "format_key argument is not working"

def test_npy_analyser(tartan2kitti_folder):
    folder_npy = os.path.join(tartan2kitti_folder, "livox")
    assert npy_analyser(folder_npy) == {"", "intensity"}

    folder_npy = os.path.join(tartan2kitti_folder, "cmd")
    assert npy_analyser(folder_npy) == {""}
"""