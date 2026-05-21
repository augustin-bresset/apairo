import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from apairo import TorchKittiDataset
from apairo.sampler.low_freq_uniform_sampler import LowFreqUniformSampler


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        ...

    def backward(self):
        ...


tartan_kitti_path = "data/tartan_2_kitti"
for data_dir in os.listdir(tartan_kitti_path):
    dataset = TorchKittiDataset(
        directory=os.path.join(tartan_kitti_path, data_dir),
        keys=["controls", "gps_odom"],
    )
    sampler = LowFreqUniformSampler(timestamps=dataset.timestamps, sample_size=1)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    # DO STUFF
