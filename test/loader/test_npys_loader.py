
import pytest
import os
import torch
import numpy as np
from src.loader import NPYSLoader
from test.utils import create_random_npy_files

@pytest.fixture
def npys_loader_data(tmp_path):
    npys_folder = tmp_path / "npys_loader_test"
    
    # Needs to be created
    npys_folder.mkdir()
    
    create_random_npy_files(100, (5, 5), str(npys_folder))
    create_random_npy_files(100, (5, 5), str(npys_folder), "intensity")
    return npys_folder

def test_format(npys_loader_data):
    loader = NPYSLoader(str(npys_loader_data))
    assert loader.npy_formats == {"", "intensity"}

def test_len(npys_loader_data):
    loader = NPYSLoader(str(npys_loader_data))
    assert len(loader) == 100

def test_getitem(npys_loader_data):
    loader = NPYSLoader(str(npys_loader_data))
    
    file0_path = npys_loader_data / "000000.npy"
    file0 = torch.from_numpy(np.load(str(file0_path)))
    
    # Default format is ""
    assert np.allclose(loader[0], file0)
    
    loader.set_format("intensity")
    file0_intensity_path = npys_loader_data / "000000_intensity.npy"
    file0_intensity = torch.from_numpy(np.load(str(file0_intensity_path)))
    
    assert np.allclose(loader[0], file0_intensity)