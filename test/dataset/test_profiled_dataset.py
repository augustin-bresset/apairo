import pytest
from pathlib import Path
from apairo.core.profiled_dataset import ModalitySpec, _parse_layers, LayerSpec

def test_modality_spec_from_dict_basic():
    spec = ModalitySpec.from_dict("lidar", {"ext": ".bin", "dtype": "float32", "reshape": [-1, 4]})
    assert spec.ext == ".bin"
    assert spec.dtype == "float32"
    assert spec.reshape == [-1, 4]
    assert spec.mask is None
    assert spec.torch_dtype is None
    assert spec.optional is False
    assert spec.effective_subpath("lidar") == ["lidar"]

def test_modality_spec_ext_normalised():
    spec = ModalitySpec.from_dict("labels", {"ext": "label", "dtype": "int32"})
    assert spec.ext == ".label"

def test_modality_spec_with_all_fields():
    spec = ModalitySpec.from_dict("labels", {
        "ext": ".label", "dtype": "int32", "mask": 65535,
        "torch_dtype": "int64", "subpath": ["camera", "left"], "optional": True,
    })
    assert spec.mask == 65535
    assert spec.torch_dtype == "int64"
    assert spec.effective_subpath("labels") == ["camera", "left"]
    assert spec.optional is True

def test_parse_layers_goose():
    raw = [
        {"split": ["train", "val", "test"]},
        "modality",
        {"split": ["train", "val", "test"]},
        "sequence",
    ]
    layers = _parse_layers(raw)
    assert len(layers) == 4
    assert layers[0].type == "split"
    assert layers[0].value == ["train", "val", "test"]
    assert layers[1].type == "modality"
    assert layers[1].value is None
    assert layers[3].type == "sequence"

def test_parse_layers_kitti():
    raw = [
        {"fixed": "sequences"},
        "sequence",
        {"modality": {"lidar": "velodyne", "labels": "labels"}},
    ]
    layers = _parse_layers(raw)
    assert layers[0].type == "fixed"
    assert layers[0].value == "sequences"
    assert layers[2].type == "modality"
    assert layers[2].value == {"lidar": "velodyne", "labels": "labels"}
