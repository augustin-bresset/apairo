
import pytest
import os
import numpy as np
import torch
from torchvision.io import read_image
from src.loader import IMGLoader
from test.utils import create_random_images

@pytest.fixture
def img_loader_data(tmp_path):
    directory = tmp_path / "img_loader_test"
    directory.mkdir()
    create_random_images(100, (16, 16), str(directory))
    return directory

def test_len(img_loader_data):
    loader = IMGLoader(str(img_loader_data))
    assert len(loader) == 100

def test_getitem(img_loader_data):
    loader = IMGLoader(str(img_loader_data))
    image0_path = img_loader_data / "000000.png"
    image0 = read_image(str(image0_path))
    assert np.allclose(loader[0], image0)