import os
import numpy as np
from typing import Dict, Collection

from apairo.utils import dict_flatten
from apairo.core import AbstractLoader
from apairo.core.utils.exceptions import FileExtensionError, EmptyLoaderWarning


class PTLoader(AbstractLoader):
    r"""A :class:`Loader` for `.pt` format (PyTorch serialised tensors).

    `.pt` is PyTorch's native format and requires torch to load.
    Values are converted to ``np.ndarray`` so the rest of the pipeline
    remains framework-agnostic.

    Raises:
        ImportError: If PyTorch is not installed.
    """

    data: Dict[str, np.ndarray]
    lenght: int

    def __init__(self, file_path):
        try:
            import torch as _torch
        except ImportError:
            raise ImportError(
                "Loading .pt files requires PyTorch. "
                "Install it with: pip install torch"
            )

        self.file_path = file_path
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File path {file_path} not found or is not a file")
        extension = file_path.split(".")[-1]
        if extension != "pt":
            raise FileExtensionError(
                f"This script extension {extension} is not a pytorch file (pt)"
            )

        _data = _torch.load(file_path, weights_only=True)
        _data = dict_flatten(_data, format_key=lambda k, sk: "{}.{}".format(k, sk))

        self.data = {k: v.numpy() for k, v in _data.items()}
        self.lenght = len(self.data["dt"])
        for key in self.data.keys():
            assert self.lenght == self.data[key].shape[0], "Data size is not uniform"
        self._timestamps: np.ndarray | None = None

    def set_keys(self, keys: Collection[str]):
        """Keep in the loader only the ``keys``."""
        keys = set(keys)
        if "dt" not in keys:
            keys.add("dt")
        current_keys = set(self.data.keys())
        for key in current_keys:
            if key not in keys:
                del self.data[key]
        if self.data.keys() == {"dt"}:
            raise EmptyLoaderWarning(
                "No data left in the loader, reload the file to retrieve the data"
            )
        self._timestamps = None

    def get_timestamps(self) -> np.ndarray:
        if self._timestamps is None:
            self._timestamps = np.cumsum(self.data["dt"])
        return self._timestamps

    def __len__(self):
        return self.lenght

    def __getitem__(self, idx) -> np.ndarray | Dict[str, np.ndarray]:
        if isinstance(idx, int):
            return {key: self.data[key][idx] for key in self.data.keys()}
        elif isinstance(idx, tuple):
            key, idx = idx
            return self.data[key][idx]
        else:
            raise TypeError(f"Index type {type(idx)} is not supported")

    @property
    def shape(self):
        return {
            key: self.data[key].shape[1:] for key in self.data.keys() if key != "dt"
        }

    def reset(self):
        self.__init__(self.file_path)
