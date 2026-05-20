from torch.utils.data import DataLoader
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, SequentialSampler
import torchvision


from apairo.dataset import TorchTKDataset
from apairo.sampler import LowFreqUniformSampler
from apairo.utils.timestamps import get_frequency

"""
In this example we will create a pipeline that will take command and odometry value
from wish we will try to predict the next odometry.

We will load the data from Tartan Drive 2.0 in Kitti format.

We will normalized the cmd value by making the assumption that we know the limits 
of command.

We will sample the data.

And finally use it on a very simple model.

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using device {device}")

class SimpleModel(nn.Module):
    def __init__(self, input_shape : int, output_shape : int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, input_shape),
            nn.ReLU(),
            nn.Linear(input_shape, output_shape)
        )
        self.double()
    def forward(self, x):
        return self.linear_relu_stack(x)
        

input_keys = ["cmd", "super_odom"]
output_keys = ["super_odom"]
keys = list(set(input_keys + output_keys))

dataset = TorchTKDataset(
    directory="data/tartan_2_kitti/example1",
    keys = keys
)

def dimension(dataset, key):
    return math.prod(dataset.shape[key])
    
input_size = sum([dimension(dataset, key) for key in input_keys])
output_size = sum([dimension(dataset, key) for key in output_keys])

print(f"Input size: {input_size}")
print(f"Output size: {output_size}")

for key in keys:
    print(f"Frequency of {key}: {get_frequency(dataset.timestamps[key])}")

sampler = LowFreqUniformSampler(dataset.timestamps, sample_size=1)

print(f"Sampler reference: {sampler.reference}")
print(f"Sampler sample size: {sampler.sample_size}")
print(f"Sampler delta indexes : {sampler._delta_indexes}")
print(f"Sampler indexes: ", " ".join([str(sampler.sample_last_indexes[key][:3]) for key in keys]))


# batch_sampler = BatchSampler(SequentialSampler(sampler), batch_size=10, drop_last=True)
model = SimpleModel(input_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, dataset, sampler, optimizer):    
    dataloader = DataLoader(dataset, sampler=sampler)
    
    model.train()

    for epoch in range(1):
        for batch in dataloader:
            for key in batch:
                print(key, len(batch[key]["data"]), batch[key]["data"][0].shape)
            
            continue

            x = torch.concat([batch[key]["data"][-1] for key in input_keys]).to(device)
            predictions = model(x)
            y = torch.concat([batch[key]["data"][0] for key in output_keys]).to(device)
            loss = nn.MSELoss()(predictions, y)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
            optimizer.zero_grad()



if __name__ == "__main__":
    train(model, dataset, sampler, optimizer)
    print("Done")