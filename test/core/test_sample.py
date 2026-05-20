from apairo.core.sample import Sample
import torch


def test_sample_creation():
    t = torch.tensor([1.0, 2.0])
    s = Sample(key="image_left", data=t, timestamp=1717184814.77)
    assert s.key == "image_left"
    assert torch.equal(s.data, t)
    assert s.timestamp == 1717184814.77


def test_sample_is_dataclass():
    from dataclasses import fields
    f = {field.name for field in fields(Sample)}
    assert f == {"key", "data", "timestamp"}
