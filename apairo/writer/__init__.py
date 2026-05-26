from .npy_writer import NPYWriter
from .pt_writer import PTWriter
from .bin_writer import BINWriter

WRITERS: dict[str, type] = {
    "npy": NPYWriter,
    "npys": NPYWriter,
    "pt": PTWriter,
    "bin": BINWriter,
}

__all__ = ["NPYWriter", "PTWriter", "BINWriter", "WRITERS"]
