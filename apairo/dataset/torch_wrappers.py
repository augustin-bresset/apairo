from torch.utils.data import Dataset, IterableDataset

from apairo.dataset.kitti import KittiDataset


class TorchKittiDataset(KittiDataset, Dataset):
    r"""Map-style PyTorch Dataset wrapping KittiDataset."""


class TorchKittiIterDataset(KittiDataset, IterableDataset):
    r"""Iterable-style PyTorch Dataset wrapping KittiDataset."""
