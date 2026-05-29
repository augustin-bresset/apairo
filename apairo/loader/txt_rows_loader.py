from pathlib import Path

import numpy as np

from apairo.core import AbstractLoader
from apairo.core.utils.exceptions import FileExtensionError


class TXTRowsLoader(AbstractLoader):
    """Loader for a single text file where each row represents one frame.

    Mirrors :class:`NPYLoader` but for plain-text files.  Pass the exact
    path to the ``.txt`` file so there is no ambiguity when a directory
    contains multiple text files (e.g. ``poses.txt`` and ``calib.txt``).

    Typical use: per-sequence ``poses.txt`` in RELLIS-3D, where every row
    is a flattened 3×4 transformation matrix for one scan.

    Args:
        path: Exact path to the ``.txt`` file to load.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileExtensionError(f"File not found: {path}")
        self.array: np.ndarray = np.loadtxt(path)
        if self.array.ndim == 1:
            self.array = self.array.reshape(1, -1)

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.array[idx]

    @property
    def shape(self) -> tuple:
        return tuple(self.array.shape[1:])
