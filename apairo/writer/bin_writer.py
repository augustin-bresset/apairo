from pathlib import Path
import numpy as np


class BINWriter:
    def write(self, arr: np.ndarray, path: Path) -> None:
        """Save arr to path as raw float32 binary. Creates parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(arr).astype(np.float32).tofile(path)
