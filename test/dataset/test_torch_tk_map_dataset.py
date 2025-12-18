
import pytest
import os
import numpy as np
from src.dataset.torch_wrapper import TorchTKDataset
from src.loader import NPYLoader
# from test.paths import tartan2kitti_path

@pytest.fixture
def dataset_setup():
    # Mocking or using real path
    # keys = ["controls", "image_left"]
    # return TorchTKDataset(directory=tartan2kitti_path, keys=keys)
    pass

def test_len():
    pass
    
def test_getitem():
    pass

def test_loaders():
    pass

def test_getitem_time_order():
    pass
