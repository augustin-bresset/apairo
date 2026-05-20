import torch
from apairo.core.sample import Sample


def test_temporal_sample():
    s = Sample(data={"image_left": torch.zeros(3, 224, 224)}, timestamp=1234.5)
    assert s.timestamp == 1234.5
    assert "image_left" in s.data
    assert isinstance(s.data["image_left"], torch.Tensor)


def test_synchronous_sample():
    s = Sample(data={"lidar": torch.zeros(1000, 4), "label": torch.zeros(1000)})
    assert s.timestamp is None
    assert "lidar" in s.data
    assert "label" in s.data


def test_sample_data_is_dict():
    s = Sample(data={"x": torch.tensor(1.0)})
    assert isinstance(s.data, dict)


def test_sample_is_dataclass():
    from dataclasses import fields
    names = {f.name for f in fields(Sample)}
    assert names == {"data", "timestamp"}
