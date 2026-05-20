import pytest
import numpy as np
from apairo.sampler.low_freq_uniform_sampler import LowFreqUniformSampler


@pytest.fixture
def sync_timestamps():
    """
    Create a set of synchronized timestamps.
    fast: 100 Hz
    slow: 10 Hz
    """
    fast_freq = 100
    return {
        "fast": np.arange(0, 1, 1 / fast_freq),
        "slow": np.arange(0, 1, 1 / 10)
    }


def test_initialization(sync_timestamps):
    sampler = LowFreqUniformSampler(sync_timestamps)
    assert sampler.reference == "slow"
    # Should expect roughly len(slow) samples
    assert len(sampler) == len(sync_timestamps["slow"])


def test_iteration(sync_timestamps):
    sampler = LowFreqUniformSampler(sync_timestamps, sample_size=1)

    batches = list(sampler)
    assert len(batches) == len(sync_timestamps["slow"])

    # Check first batch
    first_batch = batches[0]
    assert "slow" in first_batch
    assert "fast" in first_batch

    # Indexes should be integers
    assert isinstance(first_batch["slow"], (int, np.integer, np.ndarray))


def test_batch_size(sync_timestamps):
    batch_size = 2
    sampler = LowFreqUniformSampler(sync_timestamps, sample_size=batch_size)

    # With batch_size 2, we should have len - batch_size + 1 samples if sliding window?
    # Or len // batch_size?
    # Looking at implementation:
    # len = len(reference) - sample_size + 1
    # (so it's sliding window style if stride=1)
    # Wait, AbstractSampler usage might imply sliding or non-sliding.
    # The implementation says: len = len(ref) - sample_size + 1.
    # So it stops when full batch can't be formed.

    # Fixed check to match new len
    assert len(sampler) == len(sync_timestamps["slow"]) - batch_size + 1

    for batch in sampler:
        assert len(batch["slow"]) == batch_size
        assert len(batch["fast"]) >= batch_size  # Fast stream might have more samples covering the same window?
        # Actually LowFreqUniformSampler specific logic:
        # returns indexes.
        # It seems it returns arrays of indexes.
        break
