from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

import numpy as np
import torch

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.sample import Sample
from apairo.loader import DERIVED_LOADERS

_PROFILES_DIR = Path(__file__).parent.parent / "dataset" / "profiles"

_EXT_TO_LOADER: dict[str, str] = {
    ".bin":   "bin",
    ".label": "bin",
    ".npy":   "npy",
    ".png":   "img",
    ".jpg":   "img",
    ".pt":    "pt",
}


@dataclass
class ModalitySpec:
    ext: str
    dtype: str
    reshape: Optional[list] = None
    mask: Optional[int] = None
    torch_dtype: Optional[str] = None
    loader: Optional[str] = None
    subpath: list[str] = field(default_factory=list)
    optional: bool = False

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "ModalitySpec":
        ext = d["ext"]
        if not ext.startswith("."):
            ext = f".{ext}"
        return cls(
            ext=ext,
            dtype=d["dtype"],
            reshape=d.get("reshape"),
            mask=d.get("mask"),
            torch_dtype=d.get("torch_dtype"),
            loader=d.get("loader"),
            subpath=d.get("subpath", []),
            optional=d.get("optional", False),
        )

    def effective_subpath(self, key: str) -> list[str]:
        return self.subpath if self.subpath else [key]


@dataclass
class LayerSpec:
    type: str
    value: object = None


def _parse_layers(raw: list) -> list[LayerSpec]:
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(LayerSpec(type=item))
        elif isinstance(item, dict):
            k, v = next(iter(item.items()))
            result.append(LayerSpec(type=k, value=v))
    return result


class ProfiledDataset(SynchronousDataset, ConfigurableDataset):
    """Dataset driven by a YAML structural profile.

    Subclasses set _profile = "filename.yaml" (relative to profiles dir)
    or an absolute path.
    """
    _profile: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        profile_attr = cls.__dict__.get("_profile")
        if profile_attr:
            p = Path(profile_attr)
            profile_path = p if p.is_absolute() else _PROFILES_DIR / p
            if profile_path.exists():
                with open(profile_path) as f:
                    raw = yaml.safe_load(f)
                cls.available_keys = frozenset(raw.get("modalities", {}).keys())
