from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np

from apairo.loader import str_to_loader, loads_timestamps, load_profile
from apairo.utils.files import get_files
from apairo.utils.timestamps import get_end_of_time
from apairo.dataset.kitti import KittiDataset
from apairo.core.configurable_dataset import ConfigurableDataset

_PROFILE_PATH = Path(__file__).parent / "profile.yaml"


class TartanKittiDataset(KittiDataset, ConfigurableDataset):
    r"""TartanDrive v2 sequence dataset (KITTI layout).

    On first instantiation an ``.apairo`` file is created in the sequence
    directory listing all raw channels discovered from the bundled profile.
    Preprocessed channels produced by external scripts are added to that
    file via :meth:`register_channel`.

    Expected layout::

        <sequence>/
        ├── .apairo              ← created on first load; updated by register_channel
        ├── cmd/
        ├── gicp_poses/          ← declared with register_channel() after preprocessing
        ├── multisense_imu/
        ├── trav_label/          ← declared with register_channel() after preprocessing
        └── velodyne_0/

    Args:
        sequence_dir: Path to a single TartanDrive sequence directory.
        keys: Channels to load.  ``None`` → all channels listed in ``.apairo``.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
        keys: Optional[List[str]] = None,
    ) -> None:
        sequence_dir = Path(sequence_dir)
        config = self._load_or_create_config(sequence_dir)
        channels: dict = config.get("channels", {})

        missing_dirs = [k for k in channels if not (sequence_dir / k).is_dir()]
        if missing_dirs:
            raise FileNotFoundError(
                f".apairo lists channels whose directories are missing: {missing_dirs}"
            )

        if keys is None:
            keys = sorted(channels.keys())
        else:
            unknown = set(keys) - set(channels)
            if unknown:
                raise KeyError(
                    f"Keys {unknown} are not declared in .apairo. "
                    f"Register preprocessed channels with "
                    f"{type(self).__name__}.register_channel()."
                )

        # Set before super().__init__ so overridden _init_loaders can use them.
        self._effective_profile: dict[str, str] = {k: v["loader"] for k, v in channels.items()}
        self._timestamp_aliases: dict[str, str] = {
            k: v["timestamps_from"]
            for k, v in channels.items()
            if "timestamps_from" in v
        }

        super().__init__(directory=sequence_dir, keys=keys, dataset_profile=_PROFILE_PATH)

    # ---------------------------------------------------------------------- #

    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        raw_profile = load_profile(_PROFILE_PATH)
        available = get_files(str(sequence_dir))
        channels: dict = {}
        for key in sorted(available):
            if key not in raw_profile:
                continue
            has_ts = (sequence_dir / key / "timestamps.txt").exists()
            channels[key] = {"loader": raw_profile[key], "has_timestamps": has_ts}
        return {"version": 1, "channels": channels}

    # ---------------------------------------------------------------------- #

    @property
    def keys(self) -> List[str]:
        return self._keys

    @keys.setter
    def keys(self, keys: List[str]) -> None:
        unknown = set(keys) - set(self._effective_profile)
        if unknown:
            raise KeyError(
                f"Keys {unknown} are not declared in .apairo. "
                f"Register preprocessed channels with "
                f"{type(self).__name__}.register_channel()."
            )
        self._set_keys(list(keys))
        self._init()

    # ---------------------------------------------------------------------- #

    def _init_loaders(self) -> None:
        aliased = {k: self._timestamp_aliases[k] for k in self._keys if k in self._timestamp_aliases}
        standard = [k for k in self._keys if k not in aliased]

        self.loaders = {
            key: str_to_loader[self._effective_profile[key]](self._files[key])
            for key in self._keys
        }

        extra_sources = set(aliased.values()) - set(self._keys)
        for src in extra_sources:
            if src not in self._files:
                raise ValueError(
                    f"Channel '{src}' is required as a timestamp source for "
                    f"{[k for k, v in aliased.items() if v == src]} "
                    f"but its directory was not found in the sequence."
                )

        ts_keys = list({*standard, *aliased.values()})
        base_ts = loads_timestamps(ts_keys, self._files)

        self.timestamps: dict[str, np.ndarray] = {
            k: base_ts[self._timestamp_aliases[k]] if k in aliased else base_ts[k]
            for k in self._keys
        }
        self.end_of_time: float = get_end_of_time(self.timestamps) + 1.0
