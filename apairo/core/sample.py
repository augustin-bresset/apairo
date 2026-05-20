from dataclasses import dataclass
import torch


@dataclass
class Sample:
    key: str
    data: torch.Tensor
    timestamp: float
