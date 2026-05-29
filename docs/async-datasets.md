# Asynchronous Datasets

Asynchronous datasets model the reality of multi-sensor recording rigs: each sensor fires at its own rate, producing a separate stream of timestamped files. apairo merges these streams into a single timestamp-ordered timeline.

---

## Core concept

`ds[i]` returns **one event** -- one scan, one image, one IMU reading -- at its position in the merged timeline. Only the sensor that produced event `i` is populated in `sample.data`.

```python
sample = ds[0]
print(sample.timestamp)            # float -- seconds since epoch
print(list(sample.data.keys()))    # ["velodyne_0"]  -- exactly one key
print(sample.data["velodyne_0"].shape)
```

This is different from synchronous datasets where all requested modalities are always present. In async, iterate with `if "velodyne_0" in sample.data` to branch on sensor type.

---

## TartanKittiDataset

`TartanKittiDataset` handles TartanDrive v2 recordings. It auto-detects whether the path is a single sequence or a root directory containing several.

### Single sequence

```python
from apairo import TartanKittiDataset

ds = TartanKittiDataset("/data/tartan/2024-01-01_forest", keys=["velodyne_0", "cmd"])
print(len(ds))        # total events across all loaded channels
print(ds.keys)        # ["cmd", "velodyne_0"]
```

### Root directory (multiple sequences)

```python
ds = TartanKittiDataset("/data/tartan", keys=["velodyne_0"])
print(len(ds))        # sum across all sequences
print(len(ds.sequences))    # number of sequences
```

### Lazy initialisation

Omit `keys` to inspect before loading anything:

```python
ds = TartanKittiDataset("/data/tartan/2024-01-01_forest")   # no loaders built yet
print(ds.available)          # frozenset of channels in .apairo
ds.keys = ["velodyne_0"]     # build loaders on demand
ds.keys = "all"              # or load every available channel
```

### describe()

`describe()` gives a human-readable breakdown of what is available without loading any data:

```python
TartanKittiDataset.describe("/data/tartan/2024-01-01_forest")
```

```
TartanKittiDataset -- 2024-01-01_forest
──────────────────────────────────────────────────
Raw channels
  present  : cmd, image_left, velodyne_0
  missing  : depth_left, imu

Preprocessed channels
  trav_label           npys   <- timestamps from velodyne_0  sources: ['velodyne_0']
```

---

## Auto-discovery and .apairo

On the first load of a new sequence, `TartanKittiDataset` scans the directory for known channel subdirectories and writes a `.apairo` sidecar:

```yaml
version: 1
channels:
  cmd:
    has_timestamps: true
    loader: npy
  velodyne_0:
    has_timestamps: true
    loader: npys
```

Subsequent loads read from `.apairo` and skip the scan. You can inspect or edit this file directly -- it is the authoritative record of what is available.

---

## register_channel

To manually register a channel (without running a preprocessor), use `register_channel`:

```python
from apairo import TartanKittiDataset

TartanKittiDataset.register_channel(
    "/data/tartan/2024-01-01_forest",
    key="my_channel",
    loader="npys",
    timestamps_from="velodyne_0",   # share velodyne timestamps
    sources=["velodyne_0"],         # provenance metadata
)
```

After registration, the channel is available as a loadable key:

```python
ds = TartanKittiDataset("/data/tartan/2024-01-01_forest", keys=["velodyne_0", "my_channel"])
```

`register_channel` is called automatically at the end of every `run_preprocess` call -- you only need it for manually placed files.

---

## Using with PyTorch DataLoader

`DataLoader` uses `isinstance(dataset, IterableDataset)` to decide whether to use map-style (random access) or iterable-style (sequential) loading. For async datasets, declare `IterableDataset` so `DataLoader` does not attempt random access:

```python
from torch.utils.data import IterableDataset, DataLoader

class MyTartanDataset(TartanKittiDataset, IterableDataset):
    pass

ds = MyTartanDataset("/data/tartan/2024-01-01_forest", keys=["velodyne_0"])
loader = DataLoader(ds, batch_size=1, num_workers=0)

for batch in loader:
    if "velodyne_0" in batch:
        pts = batch["velodyne_0"]   # (1, N, 4)
```

!!! note "Why num_workers=0?"
    Iterable-style datasets with sequential state (timestamp-ordered iteration) don't parallelise trivially. For training, apply the temporal sampling you need before batching, or use a `LowFreqUniformSampler`.

---

## KittiDataset

`KittiDataset` is the base class for any KITTI-layout dataset -- one subdirectory per channel, each with a `timestamps.txt` and data files in a format declared by a loader profile YAML.

```python
from apairo import KittiDataset

ds = KittiDataset(
    directory="/data/my_recording",
    keys=["lidar", "imu"],
    dataset_profile="/path/to/my_profile.yaml",
)
```

**Profile YAML format:**

```yaml
# my_profile.yaml -- maps channel name -> loader type
lidar: npys
imu: npy
camera: img
```

**Extending KittiDataset:**

To create a reusable class for your dataset, subclass `KittiDataset` and `ConfigurableDataset`, and hardcode the profile path:

```python
from pathlib import Path
from apairo.dataset.kitti import KittiDataset
from apairo.core.configurable_dataset import ConfigurableDataset

_PROFILE = Path(__file__).parent / "my_profile.yaml"


class MyDataset(KittiDataset, ConfigurableDataset):
    def __init__(self, directory, keys=None):
        super().__init__(directory=directory, keys=keys or [], dataset_profile=_PROFILE)

    def _bootstrap_config(self, sequence_dir):
        # Return initial .apairo content when file doesn't exist yet
        return {
            "version": 1,
            "channels": {
                "lidar": {"loader": "npys", "has_timestamps": True},
            },
        }
```

---

## Expected directory layout

```
<sequence_dir>/
  .apairo               <- created automatically on first load
  velodyne_0/
    000000.bin
    000001.bin
    ...
    timestamps.txt
  image_left/
    000000.png
    ...
    timestamps.txt
  cmd/
    cmd.npy             <- single stacked file (NPYLoader)
    timestamps.txt
```
