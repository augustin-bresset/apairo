from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np

from apairo.loader import PTLoader
from apairo.core import AbstractDataset
from apairo.core.sample import Sample


class TartanPT(AbstractDataset):
    r"""Dataset for Tartan Drive `.pt` format files.

    Args:
        file_path: Path to the `.pt` file.
        keys: Data modalities to expose.
    """
    synchronous: bool = True

    def __init__(self, file_path: str | Path, keys: Sequence[str]) -> None:
        self._file_path = Path(file_path)
        self._loader = PTLoader(str(self._file_path))
        self.timestamps: np.ndarray = self._loader.get_timestamps()
        self._set_keys(list(keys))
        self._loader.set_keys(keys)

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def keys(self) -> list[str]:
        return self._keys

    @keys.setter
    def keys(self, keys) -> None:
        self._loader = PTLoader(str(self._file_path))
        self.timestamps = self._loader.get_timestamps()
        self._loader.set_keys(keys)
        self._set_keys(list(keys))

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int | tuple) -> Sample:
        if isinstance(idx, int):
            if not 0 <= idx < len(self):
                raise IndexError(f"Index {idx} out of range")
            data = {key: self._loader[key, idx] for key in self._keys}
            return Sample(key=",".join(self._keys), data=data, timestamp=float(self.timestamps[idx]))
        if isinstance(idx, tuple):
            key, i = idx
            return Sample(key=key, data=self._loader[key, i], timestamp=float(self.timestamps[i]))
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_idx >= len(self):
            raise StopIteration
        sample = self[self._iter_idx]
        self._iter_idx += 1
        return sample
