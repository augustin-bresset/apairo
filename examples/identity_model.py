import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset
from apairo.dataset.torch_wrapper import TorchTKDataset
from apairo.sampler.low_freq_uniform_sampler import LowFreqUniformSampler

class Model(nn.Module):
    """ Model
    """
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self):
        ...
    def backward(self):
        ...
    
tartan_kitti_path = "data/tartan_2_kitti"
for data_dir in os.listdir(tartan_kitti_path):
    dataset = TorchTKDataset(directory=os.path.join(tartan_kitti_path, data_dir), keys=["controls", "gps_odom"])) 
    sampler = LowFreqUniformSampler(timestamps=dataset.timestamps, sample_size=1)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    # DO STUFF


