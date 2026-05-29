from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    """A single dataset sample -- one timeline event or one complete frame.

    For temporal datasets: data has one key, timestamp is set.
    For synchronous datasets: data has all modality keys, timestamp is None.
    """

    data: dict[str, Any]
    timestamp: float | None = None
