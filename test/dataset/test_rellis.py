import pytest
import numpy as np
import torch
from pathlib import Path

from apairo.dataset.rellis import Rellis3DDataset
from apairo.core.sample import Sample

N_POINTS = 60


def _make_bin(path: Path, n: int = N_POINTS):
    np.random.rand(n, 4).astype(np.float32).tofile(path)


def _make_label(path: Path, n: int = N_POINTS):
    np.random.randint(0, 20, n, dtype=np.int32).tofile(path)


@pytest.fixture
def rellis_root(tmp_path):
    n_frames = 3
    for seq in ["00000", "00001"]:
        bin_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        lbl_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_label"
        bin_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n_frames):
            _make_bin(bin_dir / f"{i:06d}.bin")
            _make_label(lbl_dir / f"{i:06d}.label")
    return tmp_path, n_frames * 2  # 2 sequences × n_frames


def test_len(rellis_root):
    root, total = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar", "labels"])
    assert len(ds) == total


def test_getitem_returns_sample(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar", "labels"])
    s = ds[0]
    assert isinstance(s, Sample)
    assert s.timestamp is None
    assert s.data["lidar"].shape == (N_POINTS, 4)
    assert s.data["labels"].shape == (N_POINTS,)
    assert s.data["lidar"].dtype == torch.float32
    assert s.data["labels"].dtype == torch.int64


def test_iter(rellis_root):
    root, total = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar", "labels"])
    assert len(list(ds)) == total


def test_next(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    it = iter(ds)
    s = next(it)
    assert isinstance(s, Sample)


def test_keys_subset(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    s = ds[0]
    assert "lidar" in s.data
    assert "labels" not in s.data


def test_invalid_key(rellis_root):
    root, _ = rellis_root
    with pytest.raises(KeyError):
        Rellis3DDataset(root, keys=["nonexistent"])


def test_out_of_range(rellis_root):
    root, total = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    with pytest.raises(IndexError):
        ds[total]


def test_is_synchronous(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    assert ds.timestamps is None
    assert ds.is_synchronous is True


def test_mismatched_file_counts(tmp_path):
    """Init must fail if lidar and labels file counts differ."""
    bin_dir = tmp_path / "Rellis-3D" / "00000" / "os1_cloud_node_kitti_bin"
    lbl_dir = tmp_path / "Rellis-3D" / "00000" / "os1_cloud_node_kitti_label"
    bin_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    np.random.rand(60, 4).astype(np.float32).tofile(bin_dir / "000000.bin")
    np.random.rand(60, 4).astype(np.float32).tofile(bin_dir / "000001.bin")
    np.random.randint(0, 20, 60, dtype=np.int32).tofile(lbl_dir / "000000.label")
    with pytest.raises(ValueError):
        Rellis3DDataset(tmp_path, keys=["lidar", "labels"])
