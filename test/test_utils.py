
import pytest
import os
from src.utils.utils import dict_flatten, npy_analyser
# Assuming paths is available or we mock it
# from test.paths import tartan2kitti_path 
# Replaced with fixture if possible, or use logic

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

def test_npy_analyser(tmp_path):
    # Mocking folder structure instead of relying on external path
    folder = tmp_path / "tartan2kitti"
    folder.mkdir()
    
    livox = folder / "livox"
    livox.mkdir()
    (livox / "000000.npy").touch()
    (livox / "000000_intensity.npy").touch()
    
    cmd = folder / "cmd"
    cmd.mkdir()
    (cmd / "000000.npy").touch()
    
    assert npy_analyser(str(livox)) == {"", "intensity"}
    assert npy_analyser(str(cmd)) == {""}