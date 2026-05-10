from __future__ import annotations
import numpy as np

class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 2:
            raise ValueError("Flatten input must include a batch dimension")
        return np.reshape(x, (x.shape[0], -1), order="C")
