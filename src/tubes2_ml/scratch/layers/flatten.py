from __future__ import annotations
import numpy as np

class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 2:
            raise ValueError("Flatten input must include a batch dimension")
        self._input_shape = x.shape
        return np.reshape(x, (x.shape[0], -1), order="C")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_input_shape"):
            raise ValueError("Flatten backward called before forward")
        return np.reshape(np.asarray(grad_output, dtype=np.float32), self._input_shape, order="C")
