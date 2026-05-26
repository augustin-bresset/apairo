import numpy as np
import pytest
import torch
from pathlib import Path
from apairo.loader import DERIVED_LOADERS


def test_npy_loader_returns_tensor(tmp_path):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    npy_path = tmp_path / "test.npy"
    np.save(npy_path, data)

    result = DERIVED_LOADERS["npy"](npy_path)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, torch.from_numpy(data))


def test_pt_loader_returns_tensor(tmp_path):
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    pt_path = tmp_path / "test.pt"
    torch.save(data, pt_path)

    result = DERIVED_LOADERS["pt"](pt_path)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, data)


def test_bin_loader_returns_tensor(tmp_path):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    bin_path = tmp_path / "test.bin"
    data.tofile(bin_path)

    result = DERIVED_LOADERS["bin"](bin_path)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, torch.from_numpy(data))


def test_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        DERIVED_LOADERS["xyz"]
