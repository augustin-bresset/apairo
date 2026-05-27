# Samplers

Samplers solve a specific problem in multi-rate sensor fusion: given a dataset where different channels fire at different frequencies (e.g. LiDAR at 10 Hz, camera at 30 Hz, IMU at 200 Hz), how do you build temporally consistent batches?

apairo provides two samplers that operate on the `timestamps` dict of an async dataset and produce index batches — they don't load data, they produce `{key: array_of_indices}` dicts that you then use to fetch samples.

---

## LowFreqUniformSampler

Uses the slowest channel as the reference rate and builds sliding windows of fixed size. Guarantees that every batch has the same number of indices per channel.

```python
from apairo import TartanKittiDataset, LowFreqUniformSampler

ds = TartanKittiDataset("/data/tartan/seq", keys=["velodyne_0", "image_left"])

sampler = LowFreqUniformSampler(
    timestamp=ds.timestamps,    # dict[str, np.ndarray]
    sample_size=3,              # window of 3 reference frames
    shuffle=False,
)

print(len(sampler))   # number of windows

for index_batch in sampler:
    # index_batch = {"velodyne_0": array([0,1,2]), "image_left": array([0,1,2,...,8])}
    lidar_frames = [ds.loaders["velodyne_0"][i] for i in index_batch["velodyne_0"]]
    camera_frames = [ds.loaders["image_left"][i] for i in index_batch["image_left"]]
```

**How it works:**

1. Identifies the channel with the fewest timestamps — the *reference* channel
2. For each window of `sample_size` reference frames, finds the corresponding frames in every other channel that fall in the same time interval
3. The window size per faster channel is computed as the minimum frames-per-reference-frame across the whole sequence, keeping batch shapes constant

**Use case:** training a multi-modal network where you need a fixed number of LiDAR scans paired with camera images, at a rate you can control.

---

## LatestSyncSampler

Yields one batch per reference frame, where each other channel contributes its *latest available* reading — regardless of whether it was updated since the last batch.

```python
from apairo import LatestSyncSampler

sampler = LatestSyncSampler(
    timestamp=ds.timestamps,
    sample_size=1,
)

for index_batch in sampler:
    # index_batch = {"velodyne_0": [42], "image_left": [127]}
    # image_left[127] is the latest image available at velodyne_0[42]'s time
    pass
```

**How it works:**

1. Advances one step at a time along the reference channel
2. For every other channel, returns the latest index whose timestamp is ≤ the current reference timestamp

**Use case:** online/streaming scenarios where you always want the most recent sensor reading rather than a fixed window — accepts that some readings may be repeated across batches.

!!! warning "Variable shapes"
    `LatestSyncSampler` can yield batches where a channel repeats its last index if its sensor hasn't fired since the previous step. This is intentional — design your training loop to handle it.

---

## Choosing between the two

| | `LowFreqUniformSampler` | `LatestSyncSampler` |
|---|---|---|
| Batch shape | Fixed — same size every step | Variable — may repeat indices |
| Temporal alignment | Windowed — groups by reference period | Point-in-time — latest reading |
| Use case | Offline training | Online / streaming |
| Shuffle support | Yes | No |

---

## Sampler interface

Both samplers implement the standard Python iterator protocol:

```python
len(sampler)          # number of batches
iter(sampler)         # reset and return self
next(sampler)         # dict[str, np.ndarray of indices]
sampler.iterate(start=10, end=50)   # iterate a range (LowFreqUniformSampler only)
```

The returned index dicts are designed to be passed directly to your loader:

```python
for batch_indices in sampler:
    batch = {
        key: torch.stack([ds.loaders[key][i] for i in batch_indices[key]])
        for key in ds.keys
    }
```
