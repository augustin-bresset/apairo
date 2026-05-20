
import pytest
from apairo.loader import PTLoader
from apairo.core.utils.exceptions import EmptyLoaderWarning
from test.utils import create_random_pt_file

# Fixture to setup and teardown is handled implicitly by pytest tmp_path


@pytest.fixture
def pt_loader(tmp_path):
    dir_name = "pt_loader_test"
    file_name = "test.pt"

    directory = tmp_path / dir_name
    directory.mkdir()
    file_path = directory / file_name

    keys = ["data_a", "data_b"]
    len_ = 10
    data_shape = [(7), (3, 3)]
    shapes = {key: data_shape[i] for i, key in enumerate(keys)}

    create_random_pt_file(keys, len_, data_shape, file_name, str(directory))

    loader = PTLoader(str(file_path))

    return loader, keys, len_, shapes, str(file_path)


def test_len(pt_loader):
    loader, keys, len_, shapes, _ = pt_loader
    assert len(loader) == len_


def test_getitem(pt_loader):
    loader, _, _, _, _ = pt_loader
    assert loader[0] is not None


def test_shape(pt_loader):
    loader, keys, len_, shapes, _ = pt_loader
    for key in loader.data.keys():
        if key == "dt":
            continue
        if isinstance(shapes[key], int):
            assert int(loader.shape[key][0]) == shapes[key]
        else:
            assert tuple(loader.shape[key]) == tuple(shapes[key])


def test_set_keys(pt_loader):
    loader, keys, _, _, file_path_str = pt_loader
    loader.set_keys(["data_a"])
    assert loader.data.keys() == {"data_a", "dt"}

    # Re-initialize to test warning/exception
    loader = PTLoader(file_path_str)
    # Pytest way to check for warnings/exceptions
    # The original test checked for EmptyLoaderWarning
    with pytest.raises(EmptyLoaderWarning):
        loader.set_keys(["data_c"])


def test_reset(pt_loader):
    loader, keys, _, _, _ = pt_loader
    loader.set_keys(["data_a"])
    loader.reset()
    assert loader.data.keys() == {"data_a", "data_b", "dt"}
