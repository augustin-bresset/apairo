"""Compute per-point traversability from a labelled GOOSE3D sequence.

Given:
  - a GOOSE dataset root (lidar + labels)
  - a YAML listing which semantic labels are traversable

Produces a boolean mask (N,) per frame: True = traversable point.

Usage:
    python examples/goose_traversability.py \
        --root /data/goose/2023-06-21_flight \
        --config examples/goose_traversable_labels.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml

from apairo import Goose3DDataset


def load_traversable_labels(config_path: str | Path) -> set[int]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return set(cfg["traversable_map"])


def make_traversability_mask(
    labels: torch.Tensor, traversable: set[int]
) -> torch.Tensor:
    """Return a boolean mask: True for each point whose label is traversable."""
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for lbl in traversable:
        mask |= labels == lbl
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="GOOSE dataset root directory")
    parser.add_argument("--config", default="examples/goose_traversable_labels.yaml")
    args = parser.parse_args()

    traversable = load_traversable_labels(args.config)
    print(f"Traversable labels: {sorted(traversable)}")

    ds = Goose3DDataset(args.root, keys=["lidar", "labels"])
    print(f"Frames: {len(ds)}")

    for idx, sample in enumerate(ds):
        pts: torch.Tensor = sample.data["lidar"]  # (N, 4)  x y z intensity
        labels: torch.Tensor = sample.data["labels"]  # (N,)

        mask = make_traversability_mask(labels, traversable)

        trav_pts = pts[mask]
        non_trav_pts = pts[~mask]

        print(
            f"[{idx:04d}]  total={len(pts):6d}  "
            f"traversable={mask.sum().item():6d}  "
            f"non-traversable={( ~mask).sum().item():6d}  "
            f"ratio={mask.float().mean().item():.2%}"
        )

        # trav_pts / non_trav_pts can be passed to a model, saved, visualised, …
        _ = trav_pts, non_trav_pts


if __name__ == "__main__":
    main()
