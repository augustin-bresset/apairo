from pathlib import Path
import numpy as np
import torch


class PTWriter:
    def write(self, arr: np.ndarray, path: Path) -> None:
        """Save arr to path as .pt via torch.save. Creates parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(np.asarray(arr)), path)
