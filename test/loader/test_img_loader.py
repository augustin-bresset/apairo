import pytest
import numpy as np
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
    assert loader.shape == (3, 16, 16)


def test_iteration(img_loader_data):
    directory = img_loader_data
    loader = IMGLoader(str(directory))
    image0_path = directory / "000000.png"
    image0 = read_image(str(image0_path))
    assert np.allclose(loader[0], image0)
