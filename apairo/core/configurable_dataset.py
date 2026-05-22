from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from apairo.core.config import register_channel as _register_channel, read_config, write_config, CONFIG_FILENAME


class ConfigurableDataset:
    """Mixin for datasets that support preprocessed-channel extensibility via ``.apairo``.

    Any dataset class that wants to be extensible at runtime (i.e. allow users to
    register new preprocessed channels without touching source code) should inherit
    from this mixin alongside its normal base class.

    Concrete subclasses must implement :meth:`_bootstrap_config`, which describes
    how to auto-discover the dataset's raw channels when ``.apairo`` does not yet
    exist.

    Usage pattern for preprocessing scripts::

        MyDataset.register_channel(
            seq_dir, "my_channel", "npys",
            timestamps_from="lidar",
            sources=["lidar"],
        )

    Usage in dataset ``__init__``::

        config = self._load_or_create_config(sequence_dir)
    """

    @classmethod
    def register_channel(
        cls,
        sequence_dir: str | Path,
        key: str,
        loader: str,
        *,
        timestamps_from: Optional[str] = None,
        sources: Optional[list[str]] = None,
    ) -> None:
        """Register a preprocessed channel in ``sequence_dir/.apairo``.

        Args:
            sequence_dir: Dataset sequence directory.
            key: Channel name — must match its subdirectory name.
            loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, or ``"img"``.
            timestamps_from: Channel whose timestamps to share when this channel
                has no ``timestamps.txt`` of its own.
            sources: Provenance — channels this channel was derived from.
        """
        _register_channel(
            sequence_dir, key, loader,
            timestamps_from=timestamps_from,
            sources=sources,
        )

    @abstractmethod
    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        """Return an initial ``.apairo`` config for this dataset.

        Called when no ``.apairo`` exists yet.  Should auto-discover all raw
        channels present in ``sequence_dir`` and return a config dict of the form::

            {
                "version": 1,
                "channels": {
                    "channel_name": {"loader": "npys", "has_timestamps": True},
                    ...
                },
            }
        """
        ...

    def _load_or_create_config(self, sequence_dir: Path) -> dict:
        """Read ``.apairo`` if it exists, otherwise bootstrap and write it."""
        config_path = sequence_dir / CONFIG_FILENAME
        if not config_path.exists():
            config = self._bootstrap_config(sequence_dir)
            write_config(sequence_dir, config)
        else:
            config = read_config(sequence_dir)
        return config
