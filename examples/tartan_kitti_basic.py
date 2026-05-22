"""Load a TartanDrive v2 sequence and iterate over its channels."""

from apairo.dataset import TartanKittiDataset

SEQ_DIR = "/data/tartan/2024-01-01_forest"

# On first run, .apairo is created automatically from discovered raw channels.
ds = TartanKittiDataset(SEQ_DIR)
print("Available channels:", ds.keys)
print("Timeline length   :", len(ds))

for sample in ds:
    key = list(sample.data.keys())[0]
    print(f"t={sample.timestamp:.3f}  {key}: {sample.data[key].shape}")
