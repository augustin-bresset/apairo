from __future__ import annotations
from abc import abstractmethod
from pathlib import Path

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample


class SynchronousDataset(AbstractDataset):
    """Base class for datasets where index i returns a complete synchronous frame.

    All modalities are captured at the same time — no timestamps, no interleaving.
    Compatible with PyTorch's standard DataLoader (random/sequential sampling).

    Subclasses must implement: __len__, __getitem__, __iter__, __next__.
    """

    timestamps = None

    @property
    def root_dir(self) -> Path:
        return self._root

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        ref = next(iter(self._files.values()))[idx]
        return ref.parent.parent / key / f"{ref.stem}.{ext}"

    def _discover_derived(self, key: str, ext: str) -> list[Path]:
        paths = [self.derived_path(i, key, ext) for i in range(len(self))]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Derived key '{key}': {len(missing)} file(s) missing. "
                f"Run run_preprocess(...) to generate them."
            )
        return paths

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample: ...

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __next__(self) -> Sample: ...
