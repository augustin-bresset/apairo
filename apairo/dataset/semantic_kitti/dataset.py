from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.sample import Sample
from apairo.loader import DERIVED_LOADERS

_AVAILABLE_KEYS = {"lidar", "labels"}


class SemanticKittiDataset(SynchronousDataset):
    available_keys = frozenset(_AVAILABLE_KEYS)
    r"""SemanticKITTI synchronous dataset (LiDAR + semantic labels).

    Args:
        root_dir: Dataset root containing a ``sequences/`` subdirectory.
        keys: Modalities to load — any subset of ``{"lidar", "labels"}``.
    """

    def __init__(self, root_dir: str | Path, keys: list[str] | None = None) -> None:
        if keys is None:
            keys = ["lidar", "labels"]
        root = Path(root_dir)
        self._root = root

        native_keys = [k for k in keys if k in _AVAILABLE_KEYS]
        derived_keys = [k for k in keys if k not in _AVAILABLE_KEYS]

        if derived_keys and not native_keys:
            raise KeyError(
                f"Derived keys {derived_keys} require at least one native key "
                f"({sorted(_AVAILABLE_KEYS)}) alongside them."
            )

        self._set_keys(list(keys))
        self._files: dict[str, list[Path]] = {}
        if "lidar" in self._keys:
            self._files["lidar"] = sorted(root.glob("sequences/*/velodyne/*.bin"))
        if "labels" in self._keys:
            self._files["labels"] = sorted(root.glob("sequences/*/labels/*.label"))
        lengths = {k: len(v) for k, v in self._files.items() if k in _AVAILABLE_KEYS}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Mismatched file counts per key: {lengths}")

        self._derived_loaders: dict[str, str] = {}
        if derived_keys:
            ref_files = self._files[native_keys[0]]
            seq_dirs = sorted({f.parent.parent for f in ref_files})
            for key in derived_keys:
                ext = self._get_derived_ext(seq_dirs, key)
                self._derived_loaders[key] = ext
                self._files[key] = self._discover_derived(key, ext)

    def __len__(self) -> int:
        return len(next(iter(self._files.values())))

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        data: dict = {}
        for key in self._keys:
            path = self._files[key][idx]
            if key == "lidar":
                arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
                data[key] = torch.from_numpy(arr)
            elif key == "labels":
                arr = np.fromfile(path, dtype=np.int32) & 0xFFFF
                data[key] = torch.from_numpy(arr.copy()).long()
            else:
                ext = self._derived_loaders[key]
                data[key] = DERIVED_LOADERS[ext](path)
        return Sample(data=data)

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample
