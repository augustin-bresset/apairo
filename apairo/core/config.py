from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml

CONFIG_FILENAME = ".apairo"


def read_config(sequence_dir: Path) -> dict:
    with open(sequence_dir / CONFIG_FILENAME) as f:
        return yaml.safe_load(f)


def write_config(sequence_dir: Path, config: dict) -> None:
    with open(sequence_dir / CONFIG_FILENAME, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)


def register_channel(
    sequence_dir: str | Path,
    key: str,
    loader: str,
    *,
    timestamps_from: Optional[str] = None,
    sources: Optional[list[str]] = None,
) -> None:
    """Register a preprocessed channel in a sequence ``.apairo`` config.

    This is the low-level standalone function.  Most users will prefer the
    classmethod :meth:`ConfigurableDataset.register_channel` so that the call
    site names the dataset type explicitly.

    Args:
        sequence_dir: Dataset sequence directory.
        key: Channel name — must match its subdirectory name.
        loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, or ``"img"``.
        timestamps_from: Channel whose timestamps to share when this channel
            has no ``timestamps.txt`` of its own.
        sources: Provenance — raw channels this channel was derived from.
    """
    sequence_dir = Path(sequence_dir)
    config_path = sequence_dir / CONFIG_FILENAME
    config = read_config(sequence_dir) if config_path.exists() else {"version": 1, "channels": {}}

    has_ts = (sequence_dir / key / "timestamps.txt").exists()
    entry: dict = {"has_timestamps": has_ts, "kind": "preprocess", "loader": loader}
    if timestamps_from is not None:
        entry["timestamps_from"] = timestamps_from
    if sources:
        entry["sources"] = list(sources)

    config["channels"][key] = entry
    write_config(sequence_dir, config)
