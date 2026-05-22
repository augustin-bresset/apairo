# apairo

Unified robotics dataset loader for time-series and annotated sensor data.

Handles the two fundamental layouts you find in robotics datasets:
- **Synchronous** — every index `i` returns a complete, co-captured frame (semantic segmentation datasets, `.pt` files)
- **Asynchronous** — multiple sensors firing at different rates, interleaved into a single timeline (KITTI-style multi-modal recordings)

---

## Installation

```bash
git clone https://github.com/augustin-bresset/Apairo
cd Apairo
pip install -e .
```

Requires Python ≥ 3.11, PyTorch, NumPy.

---

## Quickstart

```python
import apairo

# Synchronous: SemanticKITTI point cloud + labels
ds = apairo.SemanticKittiDataset("/data/kitti", keys=["lidar", "labels"])
sample = ds[0]
# sample.data["lidar"]  → torch.Tensor (N, 4) float32  [x, y, z, intensity]
# sample.data["labels"] → torch.Tensor (N,)   int64    [raw semantic class]
# sample.timestamp      → None

for sample in ds:
    lidar  = sample.data["lidar"]   # (N, 4)
    labels = sample.data["labels"]  # (N,)
```

**Preprocess**
You can u
```python

>import apairo
>ds_goose = apairo.Goose("/data/goose")
>ds_goose.features
['lidar','label']
>
>SET_TRAVERSABLE = {
>    23, # Asphalt
>    24, # Gravel
>    31, # Soil
>    50, # Low Grass
>    51, # High Grass
>}
>def map_label(label: int) -> int:
>    """ Label remapping for traversability
>    """
>    return int(label in SET_TRAVERSABLE)
>
>apairp


```


---

## Synchronous datasets

Index `i` always returns a complete frame. `timestamps` is `None`, compatible with PyTorch `DataLoader`.

### SemanticKITTI

```python
ds = apairo.SemanticKittiDataset(
    root_dir="/data/semantic_kitti",   # must contain sequences/
    keys=["lidar", "labels"],          # or just ["lidar"]
)
```

Directory layout expected:
```
sequences/
  00/
    velodyne/  000000.bin  000001.bin  ...
    labels/    000000.label  000001.label  ...
  01/
    ...
```

### Rellis-3D

```python
ds = apairo.Rellis3DDataset(
    root_dir="/data/rellis",           # must contain Rellis-3D/
    keys=["lidar", "labels"],
)
```

### Goose3D

```python
ds = apairo.Goose3DDataset(
    root_dir="/data/goose",
    keys=["lidar", "labels"],
)
```

### Tartan Drive (`.pt` format)

```python
ds = apairo.TartanDataset(
    file_path="/data/tartan/seq_01.pt",
    keys=["lidar", "image_left"],
)
```

---

## Asynchronous datasets

Multiple modalities recorded at different rates. Index `i` returns one event from the interleaved timeline, with its timestamp.

### KittiDataset (multi-modal KITTI-style)

```python
ds = apairo.KittiDataset(
    directory="/data/tartan_kitti/seq_01",
    keys=["image_left", "height_map", "controls"],
)
sample = ds[0]
# sample.data      → {"image_left": tensor}   (one modality per event)
# sample.timestamp → float
```

---

## Combining datasets

### Concatenate multiple recordings

```python
sequences = [
    apairo.SemanticKittiDataset(f"/data/kitti/seq_{i:02d}", keys=["lidar", "labels"])
    for i in range(4)
]
combined = apairo.ConcatDataset(sequences)
len(combined)  # sum of all frames
```

### Train / val / test split at sequence level

Splitting at the sequence level avoids temporal leakage between splits.

```python
train_seqs, val_seqs, test_seqs = apairo.split_sequences(sequences, ratios=(0.8, 0.1, 0.1))
train = apairo.ConcatDataset(train_seqs)
val   = apairo.ConcatDataset(val_seqs)
```

---

## PyTorch DataLoader integration

Synchronous datasets and `ConcatDataset` over synchronous data work directly with PyTorch's `DataLoader`:

```python
from torch.utils.data import DataLoader

torch_ds = apairo.TorchConcatDataset(sequences)
loader = DataLoader(torch_ds, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    # batch is a list of Sample objects (default collate)
    pass
```

For asynchronous KITTI data:

```python
torch_ds = apairo.TorchKittiDataset(
    directory="/data/tartan_kitti/seq_01",
    keys=["image_left", "controls"],
)
loader = DataLoader(torch_ds, batch_size=1)
```

---

## Samplers (asynchronous data)

Samplers operate on the timestamps of an asynchronous dataset to produce batched index windows.

### LowFreqUniformSampler

Batches data using the slowest modality as reference clock — every batch has the same shape.

```python
sampler = apairo.LowFreqUniformSampler(
    timestamp=ds.timestamps,
    sample_size=3,
)

for index_batch in sampler:
    # index_batch: {"image_left": array([12, 13, 14]), "controls": array([60, 61, 62, ...])}
    pass
```

### LatestSyncSampler

Batches data when all streams have advanced — uses the latest available sample per stream.

```python
sampler = apairo.LatestSyncSampler(
    timestamp=ds.timestamps,
    sample_size=1,
)
```

---

## Custom datasets

Implement `SynchronousDataset` for co-captured frames:

```python
import apairo
from apairo import Sample
from pathlib import Path
import torch

class MyDataset(apairo.SynchronousDataset):
    def __init__(self, root: str, keys=None):
        if keys is None:
            keys = ["lidar"]
        self._set_keys(keys)
        self._files = sorted(Path(root).glob("*.bin"))

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        return Sample(data={"lidar": torch.zeros(100, 4)})

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self) -> Sample:
        if self._pos >= len(self):
            raise StopIteration
        s = self[self._pos]
        self._pos += 1
        return s
```

For asynchronous multi-modal data (multiple sensors at different rates), subclass `AbstractDataset` and implement a timestamp-based timeline — see `apairo/dataset/kitti.py` as reference.

---

## Public API

```python
import apairo

# Datasets
apairo.SemanticKittiDataset   # synchronous, LiDAR + labels
apairo.Rellis3DDataset        # synchronous, LiDAR + labels
apairo.Goose3DDataset         # synchronous, LiDAR + labels
apairo.TartanDataset          # synchronous, .pt format
apairo.KittiDataset           # asynchronous, multi-modal KITTI layout

# Composition
apairo.ConcatDataset          # concatenate datasets
apairo.TorchConcatDataset     # map-style PyTorch Dataset
apairo.TorchKittiDataset      # map-style PyTorch Dataset (async)
apairo.TorchKittiIterDataset  # iterable-style PyTorch Dataset (async)
apairo.split_sequences        # sequence-level train/val/test split

# Samplers (async)
apairo.LowFreqUniformSampler
apairo.LatestSyncSampler

# Base classes for extension
apairo.SynchronousDataset
apairo.Sample
```
