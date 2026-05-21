import pytest
import numpy as np
import torch
from pathlib import Path

from apairo.dataset.goose import Goose3DDataset
from apairo.core.sample import Sample

N_POINTS = 80


def _make_bin(path: Path, n: int = N_POINTS):
    np.random.rand(n, 4).astype(np.float32).tofile(path)


def _make_label(path: Path, n: int = N_POINTS):
    np.random.randint(0, 64, n, dtype=np.int32).tofile(path)


@pytest.fixture
def goose_root(tmp_path):
    n_frames = 5
    for seq in ["seq_a", "seq_b"]:
        lidar_dir = tmp_path / seq / "lidar" / "scan"
        label_dir = tmp_path / seq / "labels" / "scan"
        lidar_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for i in range(n_frames):
            _make_bin(lidar_dir / f"{i:06d}.bin")
            _make_label(label_dir / f"{i:06d}.label")
    return tmp_path, n_frames * 2  # 2 sequences × n_frames


def test_len(goose_root):
    root, total = goose_root
    ds = Goose3DDataset(root, keys=["lidar", "labels"])
    assert len(ds) == total


def test_getitem_returns_sample(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar", "labels"])
    s = ds[0]
    assert isinstance(s, Sample)
    assert s.timestamp is None
    assert s.data["lidar"].shape == (N_POINTS, 4)
    assert s.data["labels"].shape == (N_POINTS,)
    assert s.data["lidar"].dtype == torch.float32
    assert s.data["labels"].dtype == torch.int64


def test_iter(goose_root):
    root, total = goose_root
    ds = Goose3DDataset(root, keys=["lidar", "labels"])
    assert len(list(ds)) == total


def test_next(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    it = iter(ds)
    s = next(it)
    assert isinstance(s, Sample)


def test_keys_subset(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    s = ds[0]
    assert "lidar" in s.data
    assert "labels" not in s.data


def test_invalid_key(goose_root):
    root, _ = goose_root
    with pytest.raises(KeyError):
        Goose3DDataset(root, keys=["nonexistent"])


def test_out_of_range(goose_root):
    root, total = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    with pytest.raises(IndexError):
        ds[total]


def test_is_synchronous(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    assert ds.timestamps is None
    assert ds.is_synchronous is True


def test_paired_files_count_mismatch(tmp_path):
    """Init must fail if lidar and labels counts differ."""
    lidar_dir = tmp_path / "seq" / "lidar" / "scan"
    label_dir = tmp_path / "seq" / "labels" / "scan"
    lidar_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    _make_bin(lidar_dir / "000000.bin")
    _make_bin(lidar_dir / "000001.bin")
    _make_label(label_dir / "000000.label")  # only one label file
    with pytest.raises(ValueError):
        Goose3DDataset(tmp_path, keys=["lidar", "labels"])
