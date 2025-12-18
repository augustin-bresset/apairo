from torch.utils.data import DataLoader
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, SequentialSampler
import torchvision
from matplotlib import pyplot as plt
from src.dataset import TorchTKDataset
from src.sampler import LowFreqUniformSampler

from .utils.image import plt_image, plt_image_scaled, sub_rect_image, sub_centered_low_image
import cv2


def sub_low_image(image: torch.Tensor) -> torch.Tensor:
    """Return a sub image of the image with the size given.

    The sub image is centered for width and takes the bottom of the image for height.
    
    Args:
        image (np.ndarray) : The image to crop

    Returns:
        np.ndarray : The sub image
    """
    return sub_centered_low_image(image, WIDTH_SCALE, HEIGHT_SCALE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

keys = [
    "controls",
    "super_odom",
    "image_left"
]
dataset = TorchTKDataset(
    directory="data/tartan_2_kitti/image_model",
    keys = keys
)
image = dataset["image_left", 0]["data"][0]

WIDTH_SCALE = 7*image.shape[1]//8
HEIGHT_SCALE = 2*image.shape[0]//6 


def edge_detection(image: torch.Tensor) -> torch.Tensor:
    """Detect the edges of the image.
    
    Args:
        image (np.ndarray) : The image to process

    Returns:
        np.ndarray : The edges of the image
    """
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]])

    return torch.tensor(cv2.filter2D(image.numpy(), -1, kernel))


def cell_function_mean(cell: np.ndarray):
    """Apply a function to a cell of the image.
    """
    return np.mean(cell)


def cell_function_max_pooling(cell: np.ndarray):
    """Apply a function to a cell of the image.
    
    Take the max value of the cell.
    """
    return np.max(cell)


def grid_image(image: np.ndarray, resolution, cell_function):
    """Apply a function to each cell of the image."""
    height, width = image.shape
    new_height = height // resolution
    new_width = width // resolution
    new_image = np.zeros((new_height, new_width))
    for i in range(new_height):
        for j in range(new_width):
            new_image[i, j] = cell_function(image[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution])
    return new_image


# Image pipeline
sub_image = sub_low_image(image)
print(sub_image.shape)
edges = edge_detection(sub_image)
print("Edges constructed")

low_dim_image_mp = grid_image(edges.numpy(), 8, cell_function_max_pooling)
low_dim_image = grid_image(edges.numpy(), 8, cell_function_mean)


def image_pipeline(image: torch.Tensor) -> torch.Tensor:
    """Apply the pipeline to the image.
    
    Args:
        image (np.ndarray) : The image to process

    Returns:
        np.ndarray : The processed image
    """
    sub_image = sub_low_image(image)
    edges = edge_detection(sub_image)
    low_dim_image = grid_image(edges.numpy(), 8, cell_function_mean)
    low_dim_image = 2*low_dim_image / 255 - 1
    return torch.tensor(low_dim_image)

def control_pipeline(control: torch.Tensor) -> torch.Tensor:
    """Normalize the control value.
    
    Args:
        control (np.ndarray) : The control value to process

    Returns:
        np.ndarray : The processed control value
    """
    print(f"shape: {control}")
    return torch.from_numpy((np.asarray(control) - np.array([15, 200])) / np.array([30, 400]))

class ImageModel(nn.Module):
    """A simple model using current state, image and command to predict the next state.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.keys = [
            "controls",
            "super_odom",
            "image_left"
        ]
        self.input_shape = math.prod(low_dim_image.shape) + 7 + 2
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_shape, self.input_shape),
            nn.ReLU(),
            nn.Linear(self.input_shape, 2)
        )
        self.double()

    def pipeline(self, data):
        print(data.keys())
        print([len(data[key]["data"]) for key in self.keys])
        return torch.cat([
            control_pipeline(data["controls"]["data"][-1]),
            data["super_odom"]["data"][-1],
            image_pipeline(data["image_left"]["data"][-1])
        ])

    def forward(self, x):
        return self.linear_relu_stack(x)


def main():
    model = ImageModel().to(device)
    train_dataset = TorchTKDataset("data/tartan_2_kitti/image_model", keys)
    test_dataset = TorchTKDataset("data/tartan_2_kitti/example1", keys)

    train_sampler = LowFreqUniformSampler(train_dataset.timestamps, sample_size=1, shuffle=False)
    train_batch_sampler = BatchSampler(train_sampler, batch_size=32, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)

    test_sampler = LowFreqUniformSampler(test_dataset.timestamps, sample_size=1, shuffle=False)
    test_batch_sampler = BatchSampler(test_sampler, batch_size=32, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def train(model, train_loader, optimizer, loss_fn):
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = model.pipeline(data)
            outputs = model(inputs)
            loss = loss_fn(outputs, data["super_odom"])
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Loss: {loss.item()}")
        return loss.item()
    
    def test(model, test_loader, loss_fn):
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs = model.pipeline(data)
                outputs = model(inputs)
                loss = loss_fn(outputs, data["super_odom"])
                if i % 100 == 0:
                    print(f"Loss: {loss.item()}")
        return loss.item()
    
    for epoch in range(10):
        print(f"Epoch {epoch}")
        train_loss = train(model, train_loader, optimizer, loss_fn)
        test_loss = test(model, test_loader, loss_fn)
        print(f"Train loss: {train_loss}")
        print(f"Test loss: {test_loss}")

if __name__ == "__main__":
    main()
    