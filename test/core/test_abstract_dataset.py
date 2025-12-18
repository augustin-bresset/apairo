
import pytest
from test.utils.create_mock_dataset import create_mock_dataset
from src.core.utils.exceptions import KeysEmptyWarning, KeysDuplicateWarning

@pytest.fixture
def dataset():
    return create_mock_dataset()

def test_keys_setter(dataset):
    assert dataset.keys == ["key"]
    
    with pytest.raises(KeysEmptyWarning): # Assuming Exception type from logic
        dataset.keys = []
        
    with pytest.raises(KeysDuplicateWarning): # Assuming Exception type
        dataset.keys = ["key_a", "key_a"]
        
    dataset.keys = ["key_a", "key_b"]
    assert dataset.keys == ["key_a", "key_b"]
    
    dataset.keys = ["key"]

def test_load(dataset):
    assert dataset.load("key", 0) == "value"

def test_len(dataset):
    assert len(dataset) == 1

def test_shape(dataset):
    assert dataset.shape == (1,)

def test_iter(dataset):
    # Depending on implementation, iter might return a dict or tuple
    # Original test expected {"key": ["value"]}
    assert next(iter(dataset)) == {"key": ["value"]}